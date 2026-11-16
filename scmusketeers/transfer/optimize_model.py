"""Core training workflow for the scMusketeers transfer task.

This module defines the :class:`Workflow` class, which ties together the whole
transfer pipeline:

1. load and preprocess the dataset (:meth:`Workflow.process_dataset`),
2. split it into train / validation / test (:meth:`Workflow.train_val_split`),
3. build the DANN auto-encoder, train it through a sequence of strategies and
   predict cell types (:meth:`Workflow.make_experiment`).

The model has three branches: an auto-encoder (reconstruction), a classifier
(cell type) and a DANN discriminator (batch correction). Training proceeds as a
``training_scheme``: an ordered list of ``(strategy, n_epochs, use_perm)`` steps
(see :meth:`Workflow.get_scheme`).
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import functools
import os
import sys
from unittest import TestResult
import logging
import time

import numpy as np
import pandas as pd
import scanpy as sc
import tensorflow as tf

import keras
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                             adjusted_rand_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix,
                             davies_bouldin_score, f1_score, matthews_corrcoef,
                             normalized_mutual_info_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_class_weight

# Import scmusketeers library
sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..arguments.ae_param import AE_PARAM
    from ..arguments.class_param import CLASS_PARAM
    from ..arguments.dann_param import DANN_PARAM
    from ..transfer.dataset_tf import Dataset, load_dataset
    from ..tools import freeze
    from ..tools.clust_compute import (balanced_cohen_kappa_score,
                                       balanced_f1_score,
                                       balanced_matthews_corrcoef,
                                       batch_entropy_mixing_score, lisi_avg,
                                       nn_overlap)
    from ..tools.models import DANN_AE
    from ..tools.permutation import batch_generator_training_permuted
    from ..tools.utils import (default_value, nan_to_0, scanpy_to_input,
                               str2bool)

except ImportError:
    from scmusketeers.arguments.ae_param import AE_PARAM
    from scmusketeers.arguments.class_param import CLASS_PARAM
    from scmusketeers.arguments.dann_param import DANN_PARAM
    from scmusketeers.transfer.dataset_tf import Dataset, load_dataset
    from scmusketeers.tools import freeze
    from tools.clust_compute import (balanced_cohen_kappa_score,
                                     balanced_f1_score,
                                     balanced_matthews_corrcoef,
                                     batch_entropy_mixing_score, lisi_avg,
                                     nn_overlap)
    from tools.models import DANN_AE
    from tools.permutation import batch_generator_training_permuted
    from tools.utils import default_value, nan_to_0, scanpy_to_input, str2bool


# Use macro-averaged F1 everywhere in this module.
f1_score = functools.partial(f1_score, average="macro")

# Allow TensorFlow to grow GPU memory on demand instead of pre-allocating it.
physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

logger = logging.getLogger("Sc-Musketeers")


# =============================================================================
# Evaluation metrics
#
# Each dict maps a metric name to the callable used to compute it. They are
# iterated over in evaluation_pass() to score the predictions.
# =============================================================================

# Classification metrics computed on the predicted vs. true labels.
PRED_METRICS_LIST = {
    "acc": accuracy_score,
    "mcc": matthews_corrcoef,
    "f1_score": f1_score,
    "KPA": cohen_kappa_score,
    "ARI": adjusted_rand_score,
    "NMI": normalized_mutual_info_score,
    "AMI": adjusted_mutual_info_score,
}

# Class-balanced variants of the classification metrics (robust to imbalance).
PRED_METRICS_LIST_BALANCED = {
    "balanced_acc": balanced_accuracy_score,
    "balanced_mcc": balanced_matthews_corrcoef,
    "balanced_f1_score": balanced_f1_score,
    "balanced_KPA": balanced_cohen_kappa_score,
}

# Metrics evaluating the quality of the latent space clustering.
CLUSTERING_METRICS_LIST = {
    "db_score": davies_bouldin_score
}  #'clisi' : lisi_avg

# Metrics evaluating how well batches are mixed (i.e. batch effect removal).
BATCH_METRICS_LIST = {"batch_mixing_entropy": batch_entropy_mixing_score}


# =============================================================================
# Workflow
# =============================================================================
class Workflow:
    """End-to-end transfer pipeline: preprocess, train and predict.

    A single instance is built from a parsed ``run_file`` (the CLI namespace)
    and then driven in order: ``process_dataset()`` -> ``train_val_split()`` ->
    ``make_experiment()``.
    """

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------
    def __init__(self, run_file):
        """
        run_file : a dictionary outputed by the function load_runfile
        """
        ### Regroup here all parameters given by run_file
        self.run_file = run_file
        self.ae_param = AE_PARAM(run_file)
        self.class_param = CLASS_PARAM(run_file)
        self.dann_param = DANN_PARAM(run_file)
        self.training_scheme = self.run_file.training_scheme
        # normalization parameters
        self.normalize_size_factors = self.run_file.normalize_size_factors
        self.size_factor = self.run_file.size_factor
        self.scale_input = self.run_file.scale_input
        self.logtrans_input = self.run_file.logtrans_input
        self.use_hvg = self.run_file.use_hvg
        self.batch_size = self.run_file.batch_size
        # self.optimizer = self.run_file.optimizer
        # self.verbose = self.run_file[model_training_spec][verbose] # TODO : not implemented yet for DANN_AE
        # self.threads = self.run_file[model_training_spec][threads] # TODO : not implemented yet for DANN_AE
        self.learning_rate = self.run_file.learning_rate
        self.n_perm = 1
        self.semi_sup = False  # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
        # train test split # TODO : Simplify this, or at first only use the case where data is split according to batch
        self.test_split_key = self.run_file.test_split_key
        self.test_obs = self.run_file.test_obs
        self.test_index_name = self.run_file.test_index_name
        self.mode = self.run_file.mode
        self.pct_split = self.run_file.pct_split
        self.obs_key = self.run_file.obs_key
        self.n_keep = self.run_file.n_keep
        self.split_strategy = self.run_file.split_strategy
        self.keep_obs = self.run_file.keep_obs
        self.train_test_random_seed = self.run_file.train_test_random_seed
        # self.use_TEST = self.run_file[dataset_train_split][use_TEST] # TODO : remove, obsolete in the case of DANN_AE
        self.obs_subsample = self.run_file.obs_subsample
        # Create fake annotations
        self.make_fake = self.run_file.make_fake
        self.true_celltype = self.run_file.true_celltype
        self.false_celltype = self.run_file.false_celltype
        self.pct_false = self.run_file.pct_false
        self.clas_loss_name = self.run_file.clas_loss_name
        self.balance_classes = self.run_file.balance_classes
        self.dann_loss_name = self.run_file.dann_loss_name
        self.rec_loss_name = self.run_file.rec_loss_name
        self.weight_decay = self.run_file.weight_decay
        self.optimizer_type = self.run_file.optimizer_type
        self.clas_w = self.run_file.clas_w
        self.dann_w = self.run_file.dann_w
        self.rec_w = self.run_file.rec_w
        self.warmup_epoch = self.run_file.warmup_epoch
        self.dropout = self.run_file.dropout  # alternate way to give dropout
        self.layer1 = self.run_file.layer1
        self.layer2 = self.run_file.layer2
        self.bottleneck = self.run_file.bottleneck
        self.training_scheme = self.run_file.training_scheme

        ### Hyperamateres attributes
        self.hparam_path = self.run_file.hparam_path
        self.hp_params = None
        self.opt_metric = default_value(self.run_file.opt_metric, None)

        ### Regroup here all attributes not in run_file
        self.start_time = time.time()
        self.stop_time = time.time()
        self.run_done = False
        self.predict_done = False
        self.umap_done = False
        self.dataset = None
        self.model = None
        self.predictor = None
        self.training_kwds = {}
        self.network_kwds = {}
        self.clas_loss_fn = None
        self.dann_loss_fn = None
        self.rec_loss_fn = None
        self.num_classes = None
        self.num_batches = None
        self.metrics = []

        # This is a running average : it keeps the previous values in memory when
        # it's called (ie computes the previous and current values)
        self.mean_loss_fn = keras.metrics.Mean(name="total loss")
        self.mean_clas_loss_fn = keras.metrics.Mean(name="classification loss")
        self.mean_dann_loss_fn = keras.metrics.Mean(name="dann loss")
        self.mean_rec_loss_fn = keras.metrics.Mean(name="reconstruction loss")

    def set_hyperparameters(self, params):
        """Override the default training hyperparameters with ``params``.

        Used by the hyperparameter-optimisation path to inject a trial's
        parameter set before building and training the model.
        """
        logger.debug(f"setting hparams {params}")
        self.use_hvg = params["use_hvg"]
        self.batch_size = params["batch_size"]
        self.clas_w = params["clas_w"]
        self.dann_w = params["dann_w"]
        self.rec_w = params["rec_w"]
        self.ae_bottleneck_activation = params["ae_bottleneck_activation"]
        self.clas_loss_name = params["clas_loss_name"]
        self.size_factor = params["size_factor"]
        self.weight_decay = params["weight_decay"]
        self.learning_rate = params["learning_rate"]
        self.warmup_epoch = params["warmup_epoch"]
        self.dropout = params["dropout"]
        self.layer1 = params["layer1"]
        self.layer2 = params["layer2"]
        self.bottleneck = params["bottleneck"]
        self.training_scheme = params["training_scheme"]
        self.hp_params = params

    # -------------------------------------------------------------------------
    # Dataset loading & preprocessing
    # -------------------------------------------------------------------------
    def process_dataset(self):
        """Load the reference (and optional query) data and preprocess it.

        Wraps the raw AnnData in a :class:`Dataset`, ensures a PCA embedding is
        available (computing one if needed) and normalises the counts.
        """
        # Loading dataset
        adata = load_dataset(
            ref_path=self.run_file.ref_path,
            query_path=self.run_file.query_path,
            class_key=self.run_file.class_key,
            unlabeled_category=self.run_file.unlabeled_category,
        )

        self.dataset = Dataset(
            adata=adata,
            class_key=self.run_file.class_key,
            batch_key=self.run_file.batch_key,
            filter_min_counts=self.run_file.filter_min_counts,
            normalize_size_factors=self.run_file.normalize_size_factors,
            size_factor=self.size_factor,
            scale_input=self.scale_input,
            logtrans_input=self.logtrans_input,
            use_hvg=self.use_hvg,
            unlabeled_category=self.run_file.unlabeled_category,
            train_test_random_seed=self.train_test_random_seed,
        )

        if not "X_pca" in self.dataset.adata.obsm:
            logger.debug("Did not find existing PCA, computing it")
            sc.tl.pca(self.dataset.adata)
            self.dataset.adata.obsm["X_pca"] = np.asarray(
                self.dataset.adata.obsm["X_pca"]
            )
        # Processing dataset. Splitting train/test.
        self.dataset.normalize()

    def train_val_split(self):
        """Split the data into train/validation/test and build model inputs."""
        self.dataset.train_val_split()
        self.dataset.create_inputs()

    # -------------------------------------------------------------------------
    # Model building, training & prediction
    # -------------------------------------------------------------------------
    def make_experiment(self):
        """Build the DANN auto-encoder, train it and predict cell types.

        Returns
        -------
        adata_pred : AnnData
            Copy of the full dataset with predictions added: ``obs[class_key +
            "_scMusk"]`` (labels), ``obsm[class_key + "_scMusk_proba"]``
            (per-class probabilities) and ``obsm["X_scMusk"]`` (latent
            embedding).
        dann_ae : DANN_AE
            The trained model.
        history : dict
            Per-epoch training/validation losses and metrics.
        X_scMusk : np.ndarray
            The latent embedding of every cell.
        y_pred : np.ndarray
            The predicted labels for every cell.
        """
        # --- Apply architecture overrides coming from the CLI / hparams -------
        # A single `layer1/layer2/bottleneck` triplet expands to a symmetric
        # 5-layer auto-encoder; a single `dropout` is applied to every branch.
        if self.layer1:
            self.ae_param.ae_hidden_size = [
                self.layer1,
                self.layer2,
                self.bottleneck,
                self.layer2,
                self.layer1,
            ]

        if self.dropout:
            (
                self.dann_param.dann_hidden_dropout,
                self.class_param.class_hidden_dropout,
                self.ae_param.ae_hidden_dropout,
            ) = (self.dropout, self.dropout, self.dropout)

        # --- Gather the data of every split into parallel dictionaries -------
        # Every dict is keyed by split ("full"/"train"/"val"/"test") so the
        # training loop can address the same split across all inputs.
        adata_list = {
            "full": self.dataset.adata,
            "train": self.dataset.adata_train,
            "val": self.dataset.adata_val,
            "test": self.dataset.adata_test,
        }

        X_list = {
            "full": self.dataset.X,
            "train": self.dataset.X_train,
            "val": self.dataset.X_val,
            "test": self.dataset.X_test,
        }

        y_nooh_list = {
            "full": self.dataset.y,
            "train": self.dataset.y_train,
            "val": self.dataset.y_val,
            "test": self.dataset.y_test,
        }

        y_list = {
            "full": self.dataset.y_one_hot,
            "train": self.dataset.y_train_one_hot,
            "val": self.dataset.y_val_one_hot,
            "test": self.dataset.y_test_one_hot,
        }

        batch_list = {
            "full": self.dataset.batch_one_hot,
            "train": self.dataset.batch_train_one_hot,
            "val": self.dataset.batch_val_one_hot,
            "test": self.dataset.batch_test_one_hot,
        }

        X_pca_list = {
            "full": self.dataset.adata.obsm["X_pca"],
            "train": self.dataset.adata_train.obsm["X_pca"],
            "val": self.dataset.adata_val.obsm["X_pca"],
            "test": self.dataset.adata_test.obsm["X_pca"],
        }

        # Initialization of pseudolabels
        knn_cl = KNeighborsClassifier(n_neighbors=5)
        knn_cl.fit(X_pca_list["train"], y_nooh_list["train"])

        pseudo_y_val = pd.Series(
            knn_cl.predict(X_pca_list["val"]),
            index=adata_list["val"].obs_names,
        )
        pseudo_y_test = pd.Series(
            knn_cl.predict(X_pca_list["test"]),
            index=adata_list["test"].obs_names,
        )

        pseudo_y_full = pd.concat(
            [pseudo_y_val, pseudo_y_test, y_nooh_list["train"]]
        )
        pseudo_y_full = pseudo_y_full[
            adata_list["full"].obs_names
        ]  # reordering cells in the right order

        pseudo_y_list = {
            "full": self.dataset.ohe_celltype.transform(
                np.array(pseudo_y_full).reshape(-1, 1)
            )
            .astype(float)
            .todense(),
            "train": y_list["train"],
            "val": self.dataset.ohe_celltype.transform(
                np.array(pseudo_y_val).reshape(-1, 1)
            )
            .astype(float)
            .todense(),
            "test": self.dataset.ohe_celltype.transform(
                np.array(pseudo_y_test).reshape(-1, 1)
            )
            .astype(float)
            .todense(),
        }

        self.num_classes = len(np.unique(self.dataset.y_train))
        self.num_batches = len(np.unique(self.dataset.batch))

        # Correct size of layers depending on the number of classes and
        # on the bottleneck size
        bottleneck_size = int(
            self.ae_param.ae_hidden_size[
                int(len(self.run_file.ae_hidden_size) / 2)
            ]
        )
        self.class_param.class_hidden_size = default_value(
            self.class_param.class_hidden_size,
            (bottleneck_size + self.num_classes) / 2,
        )  # default value [(bottleneck_size + num_classes)/2]
        self.dann_param.dann_hidden_size = default_value(
            self.dann_param.dann_hidden_size,
            (bottleneck_size + self.num_batches) / 2,
        )  # default value [(bottleneck_size + num_batches)/2]

        # Creation of model
        self.dann_ae = DANN_AE(
            ae_hidden_size=self.ae_param.ae_hidden_size,
            ae_hidden_dropout=self.ae_param.ae_hidden_dropout,
            ae_activation=self.ae_param.ae_activation,
            ae_output_activation=self.ae_param.ae_output_activation,
            ae_bottleneck_activation=self.ae_param.ae_bottleneck_activation,
            ae_init=self.ae_param.ae_init,
            ae_batchnorm=self.ae_param.ae_batchnorm,
            ae_l1_enc_coef=self.ae_param.ae_l1_enc_coef,
            ae_l2_enc_coef=self.ae_param.ae_l2_enc_coef,
            num_classes=self.num_classes,
            class_hidden_size=self.class_param.class_hidden_size,
            class_hidden_dropout=self.class_param.class_hidden_dropout,
            class_batchnorm=self.class_param.class_batchnorm,
            class_activation=self.class_param.class_activation,
            class_output_activation=self.class_param.class_output_activation,
            num_batches=self.num_batches,
            dann_hidden_size=self.dann_param.dann_hidden_size,
            dann_hidden_dropout=self.dann_param.dann_hidden_dropout,
            dann_batchnorm=self.dann_param.dann_batchnorm,
            dann_activation=self.dann_param.dann_activation,
            dann_output_activation=self.dann_param.dann_output_activation,
        )

        # --- Losses and training schedule ------------------------------------
        # NB: the optimizer is intentionally not built here. train_scheme()
        # creates a fresh optimizer for every strategy (its state must reset
        # between strategies), so building one now would be dead work.
        self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn = (
            self.get_losses(y_list)
        )
        training_scheme = self.get_scheme()
        start_time = time.time()

        # --- Train the model through the full training scheme ----------------
        history = self.train_scheme(
            training_scheme=training_scheme,
            verbose=False,
            ae=self.dann_ae,
            adata_list=adata_list,
            X_list=X_list,
            y_list=y_list,
            batch_list=batch_list,
            pseudo_y_list=pseudo_y_list,
            #  optimizer= self.optimizer, # not an **loop_param since it resets between strategies
            clas_loss_fn=self.clas_loss_fn,
            dann_loss_fn=self.dann_loss_fn,
            rec_loss_fn=self.rec_loss_fn,
        )

        # --- Predict on every cell with the trained model --------------------
        # Run a forward pass to get the latent code (enc) and the class
        # probabilities (clas); dann/rec outputs are unused at prediction time.
        input_tensor = {
            k: tf.convert_to_tensor(v)
            for k, v in scanpy_to_input(
                adata_list["full"], ["size_factors"]
            ).items()
        }
        enc, clas, dann, rec = self.dann_ae(
            input_tensor, training=False
        ).values()
        # Per-class probabilities, indexed by cell and named by cell type.
        y_pred_proba = pd.DataFrame(
            np.asarray(clas),
            index=adata_list["full"].obs_names,
            columns=self.dataset.ohe_celltype.categories_[0],
        )
        # Hard labels = argmax of the probabilities, decoded back to strings.
        clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
        y_pred = self.dataset.ohe_celltype.inverse_transform(clas).reshape(
            -1,
        )

        # --- Assemble the output AnnData with predictions + embedding --------
        adata_pred = adata_list["full"].copy()

        X_scMusk = np.asarray(enc)
        adata_pred.obsm[f"{self.run_file.class_key}_scMusk_proba"] = y_pred_proba
        adata_pred.obs[f"{self.run_file.class_key}_scMusk"] = y_pred
        # scMusketeers latent embedding (bottleneck of the autoencoder)
        adata_pred.obsm["X_scMusk"] = X_scMusk

        # query_pred = adata_pred.obs[f'{self.class_key}_scMusk'][adata_pred.obs['train_split'] == 'test']

        return adata_pred, self.dann_ae, history, X_scMusk, y_pred

    # -------------------------------------------------------------------------
    # Training loops
    # -------------------------------------------------------------------------
    def train_scheme(self, training_scheme, verbose=True, **loop_params):
        """Run the full training schedule, one strategy after another.

        For every ``(strategy, n_epochs, use_perm)`` step the optimizer is reset
        and the model trained for ``n_epochs`` epochs. Strategies that monitor a
        validation metric also apply early stopping.

        training scheme : dictionary explaining the succession of strategies to use as keys with the corresponding number of epochs and use_perm as values.
                        ex :  training_scheme_3 = {"warmup_dann" : (10, False), "full_model":(10, False)}
        """
        # --- Initialise the per-split history of losses and metrics ----------
        history = {"train": {}, "val": {}}  # initialize history
        for group in history.keys():
            history[group] = {
                "total_loss": [],
                "clas_loss": [],
                "dann_loss": [],
                "rec_loss": [],
            }
            for m in PRED_METRICS_LIST:
                history[group][m] = []
            for m in PRED_METRICS_LIST_BALANCED:
                history[group][m] = []

        # if self.log_neptune:
        #     for group in history:
        #         for par,val in history[group].items():
        #             self.run[f"training/{group}/{par}"] = []
        i = 0

        total_epochs = np.sum([n_epochs for _, n_epochs, _ in training_scheme])
        running_epoch = 0
        scheme_index = 0

        # Up-front summary of the training schedule.
        logger.info(
            f"Running {self.training_scheme}: {len(training_scheme)} stages, "
            f"{total_epochs} epochs total"
        )
        for stage_strategy, stage_epochs, stage_perm in training_scheme:
            logger.info(
                f"  - {stage_strategy} : {stage_epochs} epochs "
                f"(permutation={stage_perm})"
            )

        for strategy, n_epochs, use_perm in training_scheme:
            optimizer = self.get_optimizer(
                self.learning_rate, self.weight_decay, self.optimizer_type
            )  # resetting optimizer state when switching strategy

            logger.info(
            f"##-- {strategy.upper()} - Step {scheme_index}, running {strategy} strategy with permutation = {use_perm} for {n_epochs} epochs"
            )
            time_in = time.time()
            scheme_index += 1
            # Early stopping for those strategies only
            if strategy in [
                "full_model",
                "classifier_branch",
                "permutation_only",
            ]:
                wait = 0
                best_epoch = 0
                patience = 20
                min_delta = 0
                if strategy == "permutation_only":
                    monitored = "rec_loss"
                    es_best = np.inf  # initialize early_stopping
                else:
                    split, metric = self.opt_metric.split("-")
                    monitored = metric
                    es_best = -np.inf
            memory = {}
            if strategy in [
                "warmup_dann_pseudolabels",
                "full_model_pseudolabels",
            ]:  # We use the pseudolabels computed with the model
                input_tensor = {
                    k: tf.convert_to_tensor(v)
                    for k, v in scanpy_to_input(
                        loop_params["adata_list"]["full"], ["size_factors"]
                    ).items()
                }
                enc, clas, dann, rec = self.dann_ae(
                    input_tensor, training=False
                ).values()  # Model predict
                pseudo_full = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
                pseudo_full[
                    loop_params["adata_list"]["full"].obs["train_split"]
                    == "train",
                    :,
                ] = loop_params["pseudo_y_list"][
                    "train"
                ]  # the true labels
                loop_params["pseudo_y_list"]["full"] = pseudo_full
                for group in ["val", "test"]:
                    loop_params["pseudo_y_list"][group] = pseudo_full[
                        loop_params["adata_list"]["full"].obs["train_split"]
                        == group,
                        :,
                    ]  # the predicted labels in test and val

            elif strategy in ["warmup_dann_semisup"]:
                memory = {}
                memory["pseudo_full"] = loop_params["pseudo_y_list"]["full"]
                for group in ["val", "test"]:
                    loop_params["pseudo_y_list"]["full"][
                        loop_params["adata_list"]["full"].obs["train_split"]
                        == group,
                        :,
                    ] = (
                        self.run_file.unlabeled_category
                    )  # set val and test to self.unlabeled_category
                    loop_params["pseudo_y_list"][group] = pseudo_full[
                        loop_params["adata_list"]["full"].obs["train_split"]
                        == group,
                        :,
                    ]

            else:
                if (
                    memory
                ):  # In case we are no longer using semi sup config, we reset to the previous known pseudolabels
                    for group in ["val", "test", "full"]:
                        loop_params["pseudo_y_list"][group] = memory[group]
                    memory = {}

            for epoch in range(1, n_epochs + 1):
                running_epoch += 1
                history, _, _, _, _ = self.training_loop(
                    history=history,
                    training_strategy=strategy,
                    use_perm=use_perm,
                    optimizer=optimizer,
                    **loop_params,
                )

                # Per-epoch summary (visible without --debug). The emoji shows
                # whether each loss decreased (🟢) or increased (🔴) vs. the
                # previous epoch.
                train_hist = history["train"]["total_loss"]
                val_hist = history["val"]["total_loss"]
                train_loss = train_hist[-1] if train_hist else float("nan")
                val_loss = val_hist[-1] if val_hist else float("nan")
                logger.info(
                    f"Epoch {running_epoch}/{total_epochs} - "
                    f"[{strategy}] epoch {epoch}/{n_epochs} - "
                    f"train_loss: {train_loss:.4f} {loss_trend_emoji(train_hist)} - "
                    f"val_loss: {val_loss:.4f} {loss_trend_emoji(val_hist)}"
                )
                if strategy in [
                    "full_model",
                    "classifier_branch",
                    "permutation_only",
                ]:
                    # Early stopping
                    wait += 1
                    monitored_value = history["val"][monitored][-1]

                    if "loss" in monitored:
                        if monitored_value < es_best - min_delta:
                            best_epoch = epoch
                            es_best = monitored_value
                            wait = 0
                            best_model = self.dann_ae.get_weights()
                    else:
                        if monitored_value > es_best + min_delta:
                            best_epoch = epoch
                            es_best = monitored_value
                            wait = 0
                            best_model = self.dann_ae.get_weights()
                    if wait >= patience:
                        logger.info(
                            f"Early stopping [{strategy}]: no improvement for "
                            f"{patience} epochs, restoring best model from "
                            f"epoch {best_epoch}"
                        )
                        self.dann_ae.set_weights(best_model)
                        break
            del optimizer

            if verbose:
                time_out = time.time()
                logger.debug(f"Strategy duration : {time_out - time_in} s")
        logger.debug(f"training/{group}/total_epochs = {running_epoch}")

        # Final full evaluation: the complete metric suite is computed only once,
        # here, then the headline numbers are reported.
        history, _, _, _, _ = self.evaluation_pass(
            history,
            loop_params["ae"],
            loop_params["adata_list"],
            loop_params["X_list"],
            loop_params["y_list"],
            loop_params["batch_list"],
            loop_params["clas_loss_fn"],
            loop_params["dann_loss_fn"],
            loop_params["rec_loss_fn"],
            full_metrics=True,
        )
        logger.info("Final metrics:")
        for group in ["train", "val"]:
            logger.info(
                f"  {group:>5} - accuracy: {history[group]['acc'][-1]:.4f}"
                f" - balanced_accuracy: {history[group]['balanced_acc'][-1]:.4f}"
            )
        return history

    def training_loop(
        self,
        history,
        ae,
        adata_list,
        X_list,
        y_list,
        pseudo_y_list,
        batch_list,
        optimizer,
        clas_loss_fn,
        dann_loss_fn,
        rec_loss_fn,
        use_perm=True,
        training_strategy="full_model",
        verbose=False,
    ):
        """
        A consolidated training loop function that covers common logic used in different training strategies.

        training_strategy : one of ["full", "warmup_dann", "warmup_dann_no_rec", "classifier_branch", "permutation_only"]
            - full_model : trains the whole model, optimizing the 3 losses (reconstruction, classification, anti batch discrimination ) at once
            - warmup_dann : trains the dann, encoder and decoder with reconstruction (no permutation because unsupervised), maximizing the dann loss and minimizing the reconstruction loss
            - warmup_dann_no_rec : trains the dann and encoder without reconstruction, maximizing the dann loss only.
            - dann_with_ae : same as warmup dann but with permutation. Separated in two strategies because this one is supervised
            - classifier_branch : trains the classifier branch only, without the encoder. Use to fine tune the classifier once training is over
            - permutation_only : trains the autoencoder with permutations, optimizing the reconstruction loss without the classifier
        use_perm : True by default except form "warmup_dann" training strategy. Note that for training strategies that don't involve the reconstruction, this parameter has no impact on training
        """

        freeze.unfreeze_all(ae)  # resetting freeze state
        if training_strategy == "full_model":
            group = "train"
        elif training_strategy == "full_model_pseudolabels":
            group = "full"
        elif training_strategy == "encoder_classifier":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "all_but_classifier")
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy in [
            "warmup_dann",
            "warmup_dann_pseudolabels",
            "warmup_dann_semisup",
        ]:
            group = "full"  # semi-supervised setting
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "warmup_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy == "warmup_dann_train":
            group = "train"  # semi-supervised setting
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "warmup_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy == "warmup_dann_no_rec":
            group = "full"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "all_but_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy == "dann_with_ae":
            group = "train"
            self.dann_ae.classifier.trainable = False
        elif training_strategy == "classifier_branch":
            group = "train"
            layers_to_freeze = freeze.freeze_block(
                self.dann_ae, "all_but_classifier_branch"
            )  # training only classifier branch
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy == "permutation_only":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "all_but_autoencoder")
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy == "no_dann":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "freeze_dann")
            freeze.freeze_layers(layers_to_freeze)
        elif training_strategy == "no_decoder":
            group = "train"
            layers_to_freeze = freeze.freeze_block(self.dann_ae, "freeze_dec")
            freeze.freeze_layers(layers_to_freeze)

        batch_generator = batch_generator_training_permuted(
            X=X_list[group],
            y=pseudo_y_list[
                group
            ],  # We use pseudo labels for val and test. y_train are true labels
            batch_ID=batch_list[group],
            sf=adata_list[group].obs["size_factors"],
            ret_input_only=False,
            batch_size=self.batch_size,
            n_perm=1,
            unlabeled_category=self.run_file.unlabeled_category,  # Those cells are matched with themselves during AE training
            use_perm=use_perm,
        )
        n_obs = adata_list[group].n_obs
        steps = n_obs // self.batch_size + 1
        n_steps = steps
        n_samples = 0

        self.mean_loss_fn.reset_state()
        self.mean_clas_loss_fn.reset_state()
        self.mean_dann_loss_fn.reset_state()
        self.mean_rec_loss_fn.reset_state()

        # Batch steps
        for step in range(1, n_steps + 1):
            # self.tr = tracker.SummaryTracker()
            self.batch_step(
                step,
                ae,
                clas_loss_fn,
                dann_loss_fn,
                rec_loss_fn,
                batch_generator,
                training_strategy,
                optimizer,
                n_samples,
                n_obs,
            )

        history, _, clas, dann, rec = self.evaluation_pass(
            history,
            ae,
            adata_list,
            X_list,
            y_list,
            batch_list,
            clas_loss_fn,
            dann_loss_fn,
            rec_loss_fn,
        )
        return history, _, clas, dann, rec

    def batch_step(
        self,
        step,
        ae,
        clas_loss_fn,
        dann_loss_fn,
        rec_loss_fn,
        batch_generator,
        training_strategy,
        optimizer,
        n_samples,
        n_obs,
    ):
        """Run a single gradient-descent step on one mini-batch.

        Pulls the next batch from ``batch_generator``, computes the weighted sum
        of the active losses under a gradient tape and applies the optimizer
        update. Which losses are active depends on ``training_strategy``.
        """
        # self.tr.print_diff()
        input_batch, output_batch = next(batch_generator)
        # print(f"input {type(input_batch)}")
        # X_batch, sf_batch = input_batch.values()
        clas_batch, dann_batch, rec_batch = output_batch.values()
        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])
        with tf.GradientTape() as tape:
            # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])

            input_batch = {
                k: tf.convert_to_tensor(v) for k, v in input_batch.items()
            }
            # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])

            enc, clas, dann, rec = ae(input_batch, training=True).values()
            # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])

            clas_loss = tf.reduce_mean(clas_loss_fn(clas_batch, clas))
            dann_loss = tf.reduce_mean(dann_loss_fn(dann_batch, dann))
            rec_loss = tf.reduce_mean(rec_loss_fn(rec_batch, rec))
            if training_strategy == "full_model":
                loss = tf.add_n(
                    [self.run_file.clas_w * clas_loss]
                    + [self.run_file.dann_w * dann_loss]
                    + [self.run_file.rec_w * rec_loss]
                    + ae.losses
                )
            elif training_strategy == "warmup_dann":
                loss = tf.add_n(
                    [self.run_file.dann_w * dann_loss]
                    + [self.run_file.rec_w * rec_loss]
                    + ae.losses
                )
            elif training_strategy == "warmup_dann_no_rec":
                loss = tf.add_n([self.run_file.dann_w * dann_loss] + ae.losses)
            elif training_strategy == "dann_with_ae":
                loss = tf.add_n(
                    [self.run_file.dann_w * dann_loss]
                    + [self.run_file.rec_w * rec_loss]
                    + ae.losses
                )
            elif training_strategy == "classifier_branch":
                loss = tf.add_n([self.run_file.clas_w * clas_loss] + ae.losses)
            elif training_strategy == "permutation_only":
                loss = tf.add_n([self.run_file.rec_w * rec_loss] + ae.losses)
            elif training_strategy == "no_dann":
                loss = tf.add_n(
                    [self.rec_w * rec_loss]
                    + [self.clas_w * clas_loss]
                    + ae.losses
                )
            elif training_strategy == "no_decoder":
                loss = tf.add_n(
                    [self.dann_w * dann_loss]
                    + [self.clas_w * clas_loss]
                    + ae.losses
                )
        # print(f"loss {asizeof.asizeof(loss)}")
        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])
        n_samples += enc.shape[0]
        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])
        gradients = tape.gradient(loss, ae.trainable_variables)
        # print(f"gradients {asizeof.asizeof(gradients)}")
        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])

        optimizer.apply_gradients(zip(gradients, ae.trainable_variables))
        # print(f"optimizer {asizeof.asizeof(optimizer)}")
        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])
        self.mean_loss_fn(loss.__float__())
        self.mean_clas_loss_fn(clas_loss.__float__())
        self.mean_dann_loss_fn(dann_loss.__float__())
        self.mean_rec_loss_fn(rec_loss.__float__())

        # if self.run_file.verbose:
        #     print_status_bar(
        #         n_samples,
        #         n_obs,
        #         [
        #             self.mean_loss_fn,
        #             self.mean_clas_loss_fn,
        #             self.mean_dann_loss_fn,
        #             self.mean_rec_loss_fn,
        #         ],
        #         self.metrics,
        #     )

    def evaluation_pass(
        self,
        history,
        ae,
        adata_list,
        X_list,
        y_list,
        batch_list,
        clas_loss_fn,
        dann_loss_fn,
        rec_loss_fn,
        full_metrics=False,
    ):
        """Evaluate the model on train/val and append losses & metrics to history.

        The three losses (and their weighted total) are always computed. The
        classification metrics are expensive on large datasets, so by default
        only the metric monitored for early stopping (derived from
        ``self.opt_metric``) is computed each epoch. Pass ``full_metrics=True``
        (done once at the end of training) to compute the complete metric suite.
        """
        # Select which classification metrics to compute on this pass: the full
        # suite only when explicitly requested, otherwise just the metric needed
        # for early stopping (computing all of them every epoch is wasteful).
        if full_metrics:
            pred_metrics = list(PRED_METRICS_LIST)
            balanced_metrics = list(PRED_METRICS_LIST_BALANCED)
        else:
            _, monitored = self.opt_metric.split("-")
            pred_metrics = (
                [monitored] if monitored in PRED_METRICS_LIST else []
            )
            balanced_metrics = (
                [monitored] if monitored in PRED_METRICS_LIST_BALANCED else []
            )

        for group in ["train", "val"]:  # evaluation round
            inp = scanpy_to_input(adata_list[group], ["size_factors"])
            with tf.device("CPU"):
                inp = {k: tf.convert_to_tensor(v) for k, v in inp.items()}
                _, clas, dann, rec = ae(inp, training=False).values()

                #         return _, clas, dann, rec
                clas_loss = tf.reduce_mean(
                    clas_loss_fn(y_list[group], clas)
                ).numpy()
                history[group]["clas_loss"] += [clas_loss]
                dann_loss = tf.reduce_mean(
                    dann_loss_fn(batch_list[group], dann)
                ).numpy()
                history[group]["dann_loss"] += [dann_loss]
                rec_loss = tf.reduce_mean(
                    rec_loss_fn(X_list[group], rec)
                ).numpy()
                history[group]["rec_loss"] += [rec_loss]
                history[group]["total_loss"] += [
                    self.clas_w * clas_loss
                    + self.dann_w * dann_loss
                    + self.rec_w * rec_loss
                    + np.sum(ae.losses)
                ]  # using numpy to prevent memory leaks
                # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

                clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
                for metric in pred_metrics:  # classification metrics
                    history[group][metric] += [
                        PRED_METRICS_LIST[metric](
                            np.asarray(y_list[group].argmax(axis=1)).reshape(
                                -1,
                            ),
                            clas.argmax(axis=1),
                        )
                    ]  # y_list are onehot encoded
                for metric in balanced_metrics:  # balanced classification metrics
                    history[group][metric] += [
                        PRED_METRICS_LIST_BALANCED[metric](
                            np.asarray(y_list[group].argmax(axis=1)).reshape(
                                -1,
                            ),
                            clas.argmax(axis=1),
                        )
                    ]  # y_list are onehot encoded
        del inp
        return history, _, clas, dann, rec

    # -------------------------------------------------------------------------
    # Configuration helpers (training schedule, losses, optimizer)
    # -------------------------------------------------------------------------
    def get_scheme(self):
        """Return the ordered training schedule for ``self.training_scheme``.

        Each named scheme maps to a list of ``(strategy, n_epochs, use_perm)``
        steps consumed by :meth:`train_scheme`. Epoch counts are taken from the
        ``run_file`` (``warmup_epoch``, ``fullmodel_epoch``, ...).
        """
        logger.debug(
            f"Training scheme : {self.training_scheme}, warmup {self.run_file.warmup_epoch}"
        )
        if self.training_scheme == "training_scheme_1":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                ("full_model", self.run_file.fullmodel_epoch, True),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_2":
            training_scheme = [
                ("warmup_dann_no_rec", self.run_file.warmup_epoch, False),
                ("full_model", self.run_file.fullmodel_epoch, True),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_3":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                ("full_model", self.run_file.fullmodel_epoch, True),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_4":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                (
                    "permutation_only",
                    100,
                    True,
                ),  # This will end with a callback
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_5":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                ("full_model", self.run_file.fullmodel_epoch, False),
            ]  # This will end with a callback, NO PERMUTATION HERE
        if self.training_scheme == "training_scheme_6":
            training_scheme = [
                (
                    "warmup_dann",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", self.run_file.fullmodel_epoch, True),
            ]

        if self.training_scheme == "training_scheme_7":
            training_scheme = [
                (
                    "warmup_dann",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", self.run_file.fullmodel_epoch, False),
            ]

        if self.training_scheme == "training_scheme_8":
            training_scheme = [
                (
                    "warmup_dann",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback]

        if self.training_scheme == "training_scheme_9":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                ("full_model", self.run_file.fullmodel_epoch, True),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback]

        if self.training_scheme == "training_scheme_10":
            training_scheme = [
                (
                    "warmup_dann",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
                (
                    "warmup_dann_pseudolabels",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels from the current model state
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback

        if self.training_scheme == "training_scheme_11":
            training_scheme = [
                (
                    "warmup_dann",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
                (
                    "full_model_pseudolabels",
                    100,
                    True,
                ),  # using permutations on plabels for full training
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback

        if self.training_scheme == "training_scheme_12":
            training_scheme = [
                (
                    "permutation_only",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_13":
            training_scheme = [
                ("full_model", self.run_file.fullmodel_epoch, True),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_14":
            training_scheme = [
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_15":
            training_scheme = [
                ("warmup_dann_train", self.run_file.warmup_epoch, True),
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_16":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, True),
                ("full_model", self.run_file.fullmodel_epoch, True),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_17":
            training_scheme = [
                ("no_dann", self.run_file.fullmodel_epoch, True),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_18":
            training_scheme = [
                ("no_dann", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_19":
            training_scheme = [
                (
                    "warmup_dann",
                    self.run_file.warmup_epoch,
                    False,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_20":
            training_scheme = [
                (
                    "warmup_dann_semisup",
                    self.run_file.warmup_epoch,
                    True,
                ),  # Permutating in semisup fashion ie unknown cells reconstruc themselves
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_21":
            training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                ("no_dann", 100, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_22":
            training_scheme = [
                ("permutation_only", self.run_file.warmup_epoch, True),
                ("warmup_dann", self.run_file.warmup_epoch, True),
                ("full_model", self.run_file.fullmodel_epoch, False),
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]

        if self.training_scheme == "training_scheme_23":
            training_scheme = [("full_model", 100, True)]

        if self.training_scheme == "training_scheme_24":
            training_scheme = [
                ("full_model", self.run_file.fullmodel_epoch, False),
            ]

        if self.training_scheme == "training_scheme_25":
            training_scheme = [
                ("no_decoder", 100, False),
            ]

        return training_scheme

    def get_losses(self, y_list):
        """Build the three loss functions (reconstruction, classifier, DANN).

        When ``balance_classes`` is set, the classifier focal loss is weighted
        by inverse class frequency to better handle rare cell types.
        """
        if self.rec_loss_name == "MSE":
            self.rec_loss_fn = tf.keras.losses.MSE
        else:
            logger.debug(self.rec_loss_name + " loss not supported for rec")

        if self.balance_classes:
            y_integers = np.argmax(np.asarray(y_list["train"]), axis=1)
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_integers),
                y=y_integers,
            )

        if self.clas_loss_name == "categorical_crossentropy":
            self.clas_loss_fn = tf.keras.losses.categorical_crossentropy
        elif self.clas_loss_name == "categorical_focal_crossentropy":

            self.clas_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
                alpha=class_weights, gamma=3
            )
        else:
            logger.debug(self.clas_loss_name + " loss not supported for classif")

        if self.dann_loss_name == "categorical_crossentropy":
            self.dann_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            logger.debug(self.dann_loss_name + " loss not supported for dann")
        return self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn

    def get_optimizer(
        self, learning_rate, weight_decay, optimizer_type, momentum=0.9
    ):
        """
        This function takes a  learning rate, weight decay and optionally momentum and returns an optimizer object
        Args:
            learning_rate: The optimizer's learning rate
            weight_decay: The optimizer's weight decay
            optimizer_type: The optimizer's type [adam or sgd]
            momentum: The optimizer's momentum
        Returns:
            an optimizer object
        """
        logger.debug(f"Set optimizer to {optimizer_type}")
        if optimizer_type == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                #  decay=weight_decay
            )
        elif optimizer_type == "adamw":
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            )
        elif optimizer_type == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        elif optimizer_type == "adafactor":
            optimizer = tf.keras.optimizers.Adafactor(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            optimizer = tf.keras.optimizers(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                momentum=momentum,
            )
        return optimizer


# =============================================================================
# Module-level helpers
# =============================================================================
def loss_trend_emoji(values):
    """Return a trend arrow comparing the last value to the previous one.

    ``🟢➘`` when the loss decreased (good), ``🔴➚`` when it increased: a green
    circle with a heavy diagonal-down arrow, or a red circle with a heavy
    diagonal-up arrow. The heavy dingbat arrows (➘/➚) are bigger and bolder than
    the thin ``↘``/``↗`` and render in the normal text colour. Returns an empty
    string when there is no previous value to compare against (first epoch).
    """
    if len(values) < 2:
        return ""
    return "🟢 ➘" if values[-1] < values[-2] else "🔴 ➚"


def print_status_bar(iteration, total, loss, metrics=None):
    """Print a carriage-return progress bar with current losses and metrics."""
    metrics = " - ".join(
        [
            "{}: {:.4f}".format(m.name, m.result())
            for m in loss + (metrics or [])
        ]
    )

    end = "" if int(iteration) < int(total) else "\n"
    #     print(f"{iteration}/{total} - "+metrics ,end="\r")
    #     print(f"\r{iteration}/{total} - " + metrics, end=end)
    print("\r{}/{} - ".format(iteration, total) + metrics, end=end)
