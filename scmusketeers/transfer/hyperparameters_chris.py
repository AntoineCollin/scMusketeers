# except ImportError:
#     from dataset import Dataset, load_dataset
#     from scpermut.tools.utils import scanpy_to_input, default_value, str2bool
#     from scpermut.tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg
# from dca.utils import str2bool,tuple_to_scalar
import argparse
import functools
import os
import sys

import keras
from pympler import asizeof, tracker
from pympler.classtracker import ClassTracker
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import compute_class_weight

# try :
#     from .dataset import Dataset, load_dataset
#     from ..tools.utils import scanpy_to_input, default_value, str2bool
#     from ..tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg


sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from .dataset_tf import Dataset, load_dataset
except ImportError:
    from dataset_tf import Dataset, load_dataset

try:
    from ..tools.clust_compute import (
        balanced_cohen_kappa_score,
        balanced_f1_score,
        balanced_matthews_corrcoef,
        batch_entropy_mixing_score,
        lisi_avg,
        nn_overlap,
    )
    from ..tools.models import DANN_AE
    from ..tools.permutation import batch_generator_training_permuted
    from ..tools.utils import (
        default_value,
        nan_to_0,
        scanpy_to_input,
        str2bool,
    )

except ImportError:
    from tools.clust_compute import (
        balanced_cohen_kappa_score,
        balanced_f1_score,
        balanced_matthews_corrcoef,
        batch_entropy_mixing_score,
        lisi_avg,
        nn_overlap,
    )
    from tools.models import DANN_AE
    from tools.permutation import batch_generator_training_permuted
    from tools.utils import default_value, nan_to_0, scanpy_to_input, str2bool


f1_score = functools.partial(f1_score, average="macro")
import gc
import json
import os
import subprocess
import sys
import time

import matplotlib.pyplot as plt
import neptune
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf
from ax.service.managed_loop import optimize

# from numba import cuda
from neptune.utils import stringify_unsupported

# from ax import RangeParameter, SearchSpace, ParameterType, FixedParameter, ChoiceParameter

physical_devices = tf.config.list_physical_devices("GPU")
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


# Reset Keras Session
def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier  # this is from global space - change this as you need
    except:
        pass

    # print(gc.collect())

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


class Workflow:
    def __init__(self, run_file):
        """
        run_file : a dictionary outputed by the function load_runfile
        """
        print("NOOOOO EAGERRRRR")
        ## tf.compat.v1.disable_eager_execution()
        # self.sess = tf.compat.v1.Session()
        self.run_file = run_file
        # dataset identifiers
        self.ref_path = (
            self.run_file.ref_path
        )  # If providing only one dataset, the unlabeled_category argument is mandatory du distinguish ref from query
        self.query_path = self.run_file.query_path
        self.class_key = self.run_file.class_key
        self.batch_key = self.run_file.batch_key
        # normalization parameters
        self.filter_min_counts = (
            self.run_file.filter_min_counts
        )  # TODO :remove, we always want to do that
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

        self.start_time = time.time()
        self.stop_time = time.time()
        # self.runtime_path = self.result_path + '/runtime.txt'

        self.run_done = False
        self.predict_done = False
        self.umap_done = False

        self.dataset = None
        self.model = None
        self.predictor = None

        self.training_kwds = {}
        self.network_kwds = {}

        ##### TODO : Add to runfile

        self.clas_loss_name = self.run_file.clas_loss_name
        self.clas_loss_name = default_value(
            self.clas_loss_name, "categorical_crossentropy"
        )
        self.balance_classes = self.run_file.balance_classes
        self.dann_loss_name = self.run_file.dann_loss_name
        self.dann_loss_name = default_value(
            self.dann_loss_name, "categorical_crossentropy"
        )
        self.rec_loss_name = self.run_file.rec_loss_name
        self.rec_loss_name = default_value(self.rec_loss_name, "MSE")

        self.clas_loss_fn = None
        self.dann_loss_fn = None
        self.rec_loss_fn = None

        self.weight_decay = self.run_file.weight_decay
        self.weight_decay = default_value(self.clas_loss_name, None)
        self.optimizer_type = self.run_file.optimizer_type
        self.optimizer_type = default_value(self.optimizer_type, "adam")

        self.clas_w = self.run_file.clas_w
        self.dann_w = self.run_file.dann_w
        self.rec_w = self.run_file.rec_w
        self.warmup_epoch = self.run_file.warmup_epoch

        self.num_classes = None
        self.num_batches = None

        self.ae_hidden_size = self.run_file.ae_hidden_size
        self.ae_hidden_size = default_value(
            self.ae_hidden_size, (128, 64, 128)
        )
        self.ae_hidden_dropout = self.run_file.ae_hidden_dropout

        self.dropout = self.run_file.dropout  # alternate way to give dropout
        self.layer1 = (
            self.run_file.layer1
        )  # alternate way to give model dimensions
        self.layer2 = self.run_file.layer2
        self.bottleneck = self.run_file.bottleneck

        # self.ae_hidden_dropout = default_value(self.ae_hidden_dropout , None)
        self.ae_activation = self.run_file.ae_activation
        self.ae_activation = default_value(self.ae_activation, "relu")
        self.ae_bottleneck_activation = self.run_file.ae_bottleneck_activation
        self.ae_bottleneck_activation = default_value(
            self.ae_bottleneck_activation, "linear"
        )
        self.ae_output_activation = self.run_file.ae_output_activation
        self.ae_output_activation = default_value(
            self.ae_output_activation, "relu"
        )
        self.ae_init = self.run_file.ae_init
        self.ae_init = default_value(self.ae_init, "glorot_uniform")
        self.ae_batchnorm = self.run_file.ae_batchnorm
        self.ae_batchnorm = default_value(self.ae_batchnorm, True)
        self.ae_l1_enc_coef = self.run_file.ae_l1_enc_coef
        self.ae_l1_enc_coef = default_value(self.ae_l1_enc_coef, 0)
        self.ae_l2_enc_coef = self.run_file.ae_l2_enc_coef
        self.ae_l2_enc_coef = default_value(self.ae_l2_enc_coef, 0)

        self.class_hidden_size = self.run_file.class_hidden_size
        self.class_hidden_size = default_value(
            self.class_hidden_size, None
        )  # default value will be initialize as [(bottleneck_size + num_classes)/2] once we'll know num_classes
        self.class_hidden_dropout = self.run_file.class_hidden_dropout
        self.class_batchnorm = self.run_file.class_batchnorm
        self.class_batchnorm = default_value(self.class_batchnorm, True)
        self.class_activation = self.run_file.class_activation
        self.class_activation = default_value(self.class_activation, "relu")
        self.class_output_activation = self.run_file.class_output_activation
        self.class_output_activation = default_value(
            self.class_output_activation, "softmax"
        )

        self.dann_hidden_size = self.run_file.dann_hidden_size
        self.dann_hidden_size = default_value(
            self.dann_hidden_size, None
        )  # default value will be initialize as [(bottleneck_size + num_batches)/2] once we'll know num_classes
        self.dann_hidden_dropout = self.run_file.dann_hidden_dropout
        self.dann_batchnorm = self.run_file.dann_batchnorm
        self.dann_batchnorm = default_value(self.dann_batchnorm, True)
        self.dann_activation = self.run_file.dann_activation
        self.dann_activation = default_value(self.dann_activation, "relu")
        self.dann_output_activation = self.run_file.dann_output_activation
        self.dann_output_activation = default_value(
            self.dann_output_activation, "softmax"
        )

        self.dann_ae = None

        self.pred_metrics_list = {
            "acc": accuracy_score,
            "mcc": matthews_corrcoef,
            "f1_score": f1_score,
            "KPA": cohen_kappa_score,
            "ARI": adjusted_rand_score,
            "NMI": normalized_mutual_info_score,
            "AMI": adjusted_mutual_info_score,
        }

        self.pred_metrics_list_balanced = {
            "balanced_acc": balanced_accuracy_score,
            "balanced_mcc": balanced_matthews_corrcoef,
            "balanced_f1_score": balanced_f1_score,
            "balanced_KPA": balanced_cohen_kappa_score,
        }

        self.clustering_metrics_list = {  #'clisi' : lisi_avg,
            "db_score": davies_bouldin_score
        }

        self.batch_metrics_list = {
            "batch_mixing_entropy": batch_entropy_mixing_score,
            #'ilisi': lisi_avg
        }
        self.metrics = []

        self.mean_loss_fn = keras.metrics.Mean(
            name="total_loss"
        )  # This is a running average : it keeps the previous values in memory when it's called ie computes the previous and current values
        self.mean_clas_loss_fn = keras.metrics.Mean(name="classification_loss")
        self.mean_dann_loss_fn = keras.metrics.Mean(name="dann_loss")
        self.mean_rec_loss_fn = keras.metrics.Mean(name="reconstruction_loss")

        self.training_scheme = self.run_file.training_scheme

        self.log_neptune = self.run_file.log_neptune
        self.run = None

        self.hparam_path = self.run_file.hparam_path
        self.hp_params = None
        self.opt_metric = default_value(self.run_file.opt_metric, None)

    def set_hyperparameters(self, params):

        print(f"setting hparams {params}")
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

    def process_dataset(self):
        # Loading dataset
        adata = load_dataset(
            ref_path=self.ref_path,
            query_path=self.query_path,
            class_key=self.class_key,
            unlabeled_category=self.run_file.unlabeled_category,
        )

        if not "X_pca" in adata.obsm:
            print("Did not find existing PCA, computing it")
            sc.tl.pca(adata)
            adata.obsm["X_pca"] = np.asarray(adata.obsm["X_pca"])
        # Processing dataset. Splitting train/test.

        self.dataset = Dataset(
            adata=adata,
            class_key=self.class_key,
            batch_key=self.batch_key,
            filter_min_counts=self.filter_min_counts,
            normalize_size_factors=self.normalize_size_factors,
            size_factor=self.size_factor,
            scale_input=self.scale_input,
            logtrans_input=self.logtrans_input,
            use_hvg=self.use_hvg,
            unlabeled_category=self.run_file.unlabeled_category,
            train_test_random_seed=self.train_test_random_seed,
        )

        self.dataset.normalize()

    def train_val_split(self):
        self.dataset.train_val_split()
        self.dataset.create_inputs()

    def make_experiment(self):
        if self.layer1:
            self.ae_hidden_size = [
                self.layer1,
                self.layer2,
                self.bottleneck,
                self.layer2,
                self.layer1,
            ]

        if self.dropout:
            (
                self.dann_hidden_dropout,
                self.class_hidden_dropout,
                self.ae_hidden_dropout,
            ) = (self.dropout, self.dropout, self.dropout)

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

        bottleneck_size = int(
            self.ae_hidden_size[int(len(self.ae_hidden_size) / 2)]
        )

        self.class_hidden_size = default_value(
            self.class_hidden_size, (bottleneck_size + self.num_classes) / 2
        )  # default value [(bottleneck_size + num_classes)/2]
        self.dann_hidden_size = default_value(
            self.dann_hidden_size, (bottleneck_size + self.num_batches) / 2
        )  # default value [(bottleneck_size + num_batches)/2]

        # Creation of model
        self.dann_ae = DANN_AE(
            ae_hidden_size=self.ae_hidden_size,
            ae_hidden_dropout=self.ae_hidden_dropout,
            ae_activation=self.ae_activation,
            ae_output_activation=self.ae_output_activation,
            ae_bottleneck_activation=self.ae_bottleneck_activation,
            ae_init=self.ae_init,
            ae_batchnorm=self.ae_batchnorm,
            ae_l1_enc_coef=self.ae_l1_enc_coef,
            ae_l2_enc_coef=self.ae_l2_enc_coef,
            num_classes=self.num_classes,
            class_hidden_size=self.class_hidden_size,
            class_hidden_dropout=self.class_hidden_dropout,
            class_batchnorm=self.class_batchnorm,
            class_activation=self.class_activation,
            class_output_activation=self.class_output_activation,
            num_batches=self.num_batches,
            dann_hidden_size=self.dann_hidden_size,
            dann_hidden_dropout=self.dann_hidden_dropout,
            dann_batchnorm=self.dann_batchnorm,
            dann_activation=self.dann_activation,
            dann_output_activation=self.dann_output_activation,
        )

        self.optimizer = self.get_optimizer(
            self.learning_rate, self.weight_decay, self.optimizer_type
        )
        self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn = (
            self.get_losses(y_list)
        )  # redundant
        training_scheme = self.get_scheme()
        start_time = time.time()

        # Training
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

        # predicting
        input_tensor = {
            k: tf.convert_to_tensor(v)
            for k, v in scanpy_to_input(
                adata_list["full"], ["size_factors"]
            ).items()
        }
        enc, clas, dann, rec = self.dann_ae(
            input_tensor, training=False
        ).values()
        y_pred_proba = pd.DataFrame(
            np.asarray(clas),
            index=adata_list["full"].obs_names,
            columns=self.dataset.ohe_celltype.categories_[0],
        )
        y_pred = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]

        adata_pred = adata_list["full"].copy()

        X_scCER = enc
        adata_pred.obsm[f"{self.class_key}_pred_proba"] = y_pred_proba
        adata_pred.obs[f"{self.class_key}_pred"] = y_pred
        adata_pred.obsm["X_scCER"] = X_scCER

        query_pred = adata_pred.obs[f"{self.class_key}_pred"][
            adata_pred.obs["train_split"] == "test"
        ]

        return adata_pred, self.dann_ae, history, X_scCER, query_pred

    def train_scheme(self, training_scheme, verbose=True, **loop_params):
        """
        training scheme : dictionnary explaining the succession of strategies to use as keys with the corresponding number of epochs and use_perm as values.
                        ex :  training_scheme_3 = {"warmup_dann" : (10, False), "full_model":(10, False)}
        """
        history = {"train": {}, "val": {}}  # initialize history
        for group in history.keys():
            history[group] = {
                "total_loss": [],
                "clas_loss": [],
                "dann_loss": [],
                "rec_loss": [],
            }
            for m in self.pred_metrics_list:
                history[group][m] = []
            for m in self.pred_metrics_list_balanced:
                history[group][m] = []

        # if self.log_neptune:
        #     for group in history:
        #         for par,val in history[group].items():
        #             self.run[f"training/{group}/{par}"] = []
        i = 0

        total_epochs = np.sum([n_epochs for _, n_epochs, _ in training_scheme])
        running_epoch = 0

        for strategy, n_epochs, use_perm in training_scheme:
            optimizer = self.get_optimizer(
                self.learning_rate, self.weight_decay, self.optimizer_type
            )  # resetting optimizer state when switching strategy
            if verbose:
                print(
                    f"Step number {i}, running {strategy} strategy with permuation = {use_perm} for {n_epochs} epochs"
                )
                time_in = time.time()

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

            # Pseudolabels
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
                        self.unlabeled_category
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
                # self.tr.print_diff()
                # tracker_class.create_snapshot()
                # tracker_class.stats.print_summary()
                running_epoch += 1
                print(
                    f"Epoch {running_epoch}/{total_epochs}, Current strat Epoch {epoch}/{n_epochs}"
                )
                history, _, _, _, _ = self.training_loop(
                    history=history,
                    training_strategy=strategy,
                    use_perm=use_perm,
                    optimizer=optimizer,
                    **loop_params,
                )
                # tracker_class.create_snapshot()
                # tracker_class.stats.print_summary()
                # print(f"size {asizeof.asizeof(self.dann_ae)}")
                # print(f"size {asizeof.asizeof(self.dataset)}")
                # print(f"size {asizeof.asizeof(self.history)}")
                # Look at workflow size
                # attribute_dict = self.__dict__
                # type_attributes = {}
                # for key, value in attribute_dict.items():
                #     type_attr = type(value)
                #     if type_attr in type_attributes:
                #         list_attr = type_attributes[type_attr]
                #         list_attr.append(key)
                #         type_attributes[type_attr] = list_attr
                #     else:
                #         type_attributes[type_attr] = [key]
                # print(type_attributes.keys())
                # import sys
                # from pympler import asizeof
                # size_attributes = {}
                # for key, value in type_attributes.items():
                #     size_of = 0
                #     for variable in value:
                #         size_of += asizeof.asizeof(variable)
                #     size_attributes[key] = size_of
                # print(size_attributes)
                # if self.log_neptune:
                #     for key, value in size_attributes.items():
                #         self.run['memory/'+str(key)].append(value)
                #         print(value)
                # all objects
                # all_objects = muppy.get_objects()
                # print("allobjects "+str(len(all_objects)))
                # my_types = muppy.filter(all_objects, Type=type)
                # print(len(my_types))
                # for t in my_types:
                #     print(t)

                """ if self.log_neptune:
                    for group in history:
                        for par,value in history[group].items():
                            self.run[f"training/{group}/{par}"].append(value[-1])
                            if physical_devices :
                                # print(f"memory {tf.config.experimental.get_memory_info('GPU:0')['current']}")
                                self.run['training/train/tf_GPU_memory'].append(tf.config.experimental.get_memory_info('GPU:0')['current']/1e6)
                                """
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
                        print(
                            f"Early stopping at epoch {best_epoch}, restoring model parameters from this epoch"
                        )
                        self.dann_ae.set_weights(best_model)
                        break
            del optimizer

            if verbose:
                time_out = time.time()
                print(f"Strategy duration : {time_out - time_in} s")
        if self.log_neptune:
            self.run[f"training/{group}/total_epochs"] = running_epoch
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
        use_perm=None,
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

        self.unfreeze_all(ae)  # resetting freeze state
        if training_strategy == "full_model":
            group = "train"
        elif training_strategy == "full_model_pseudolabels":
            group = "full"
        elif training_strategy in [
            "warmup_dann",
            "warmup_dann_pseudolabels",
            "warmup_dann_semisup",
        ]:
            group = "full"  # semi-supervised setting
            ae.classifier.trainable = False  # Freezing classifier just to be sure but should not be necessary since gradient won't be propagating in this branch
        elif training_strategy == "warmup_dann_train":
            group = "train"  # semi-supervised setting
            ae.classifier.trainable = False  # Freezing classifier just to be sure but should not be necessary since gradient won't be propagating in this branch
        elif training_strategy == "warmup_dann_no_rec":
            group = "full"
            self.freeze_block(ae, "all_but_dann")
        elif training_strategy == "dann_with_ae":
            group = "train"
            ae.classifier.trainable = False
        elif training_strategy == "classifier_branch":
            group = "train"
            self.freeze_block(
                ae, "all_but_classifier_branch"
            )  # training only classifier branch
        elif training_strategy == "permutation_only":
            group = "train"
            self.freeze_block(ae, "all_but_autoencoder")
        elif training_strategy == "no_dann":
            group = "train"
            self.freeze_block(ae, "freeze_dann")
        elif training_strategy == "no_decoder":
            group = "train"
            self.freeze_block(ae, "freeze_dec")

        print(f"use_perm = {use_perm}")
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
            # self.run['training/train/tf_GPU_memory'].append(tf.config.experimental.get_memory_info('GPU:0')['current']/1e6)
            # self.tr.print_diff()
            input_batch, output_batch = next(batch_generator)
            # X_batch, sf_batch = input_batch.values()
            clas_batch, dann_batch, rec_batch = output_batch.values()

            with tf.GradientTape() as tape:
                input_batch = {
                    k: tf.convert_to_tensor(v) for k, v in input_batch.items()
                }
                enc, clas, dann, rec = ae(
                    input_batch, training=True
                ).values()  # Forward pass
                clas_loss = tf.reduce_mean(clas_loss_fn(clas_batch, clas))
                dann_loss = tf.reduce_mean(dann_loss_fn(dann_batch, dann))
                rec_loss = tf.reduce_mean(rec_loss_fn(rec_batch, rec))

                if training_strategy in [
                    "full_model",
                    "full_model_pseudolabels",
                ]:
                    loss = tf.add_n(
                        [self.clas_w * clas_loss]
                        + [self.dann_w * dann_loss]
                        + [self.rec_w * rec_loss]
                        + ae.losses
                    )
                elif training_strategy in [
                    "warmup_dann",
                    "warmup_dann_pseudolabels",
                    "warmup_dann_train",
                    "warmup_dann_semisup",
                ]:
                    loss = tf.add_n(
                        [self.dann_w * dann_loss]
                        + [self.rec_w * rec_loss]
                        + ae.losses
                    )
                elif training_strategy == "warmup_dann_no_rec":
                    loss = tf.add_n([self.dann_w * dann_loss] + ae.losses)
                elif training_strategy == "dann_with_ae":
                    loss = tf.add_n(
                        [self.dann_w * dann_loss]
                        + [self.rec_w * rec_loss]
                        + ae.losses
                    )
                elif training_strategy == "classifier_branch":
                    loss = tf.add_n([self.clas_w * clas_loss] + ae.losses)
                elif training_strategy == "permutation_only":
                    loss = tf.add_n([self.rec_w * rec_loss] + ae.losses)
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

            n_samples += enc.shape[0]

            # Backpropagation
            gradients = tape.gradient(loss, ae.trainable_variables)
            optimizer.apply_gradients(zip(gradients, ae.trainable_variables))

            self.mean_loss_fn(loss)
            self.mean_clas_loss_fn(clas_loss)
            self.mean_dann_loss_fn(dann_loss)
            self.mean_rec_loss_fn(rec_loss)

            if verbose:
                self.print_status_bar(
                    n_samples,
                    n_obs,
                    [
                        self.mean_loss_fn,
                        self.mean_clas_loss_fn,
                        self.mean_dann_loss_fn,
                        self.mean_rec_loss_fn,
                    ],
                    self.metrics,
                )
        # self.print_status_bar(n_samples, n_obs, [self.mean_loss_fn, self.mean_clas_loss_fn, self.mean_dann_loss_fn, self.mean_rec_loss_fn], self.metrics)
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
        del input_batch
        return history, _, clas, dann, rec

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
    ):
        """
        evaluate model and logs metrics. Depending on "on parameter, computes it on train and val or train,val and test.

        on : "epoch_end" to evaluate on train and val, "training_end" to evaluate on train, val and "test".
        """
        for group in ["train", "val"]:  # evaluation round
            inp = scanpy_to_input(adata_list[group], ["size_factors"])
            with tf.device("CPU"):
                inp = {k: tf.convert_to_tensor(v) for k, v in inp.items()}
                _, clas, dann, rec = ae(inp, training=False).values()

                #         return _, clas, dann, rec
                clas_loss = tf.reduce_mean(clas_loss_fn(y_list[group], clas))
                history[group]["clas_loss"] += [clas_loss]
                dann_loss = tf.reduce_mean(
                    dann_loss_fn(batch_list[group], dann)
                )
                history[group]["dann_loss"] += [dann_loss]
                rec_loss = tf.reduce_mean(rec_loss_fn(X_list[group], rec))
                history[group]["rec_loss"] += [rec_loss]
                history[group]["total_loss"] += [
                    self.clas_w * clas_loss
                    + self.dann_w * dann_loss
                    + self.rec_w * rec_loss
                    + tf.reduce_sum(ae.losses)
                ]  # using numpy to prevent memory leaks
                # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

                # Perform ArgMax operation
                # argmax_op = tf.math.argmax(clas, axis=1)
                # print(argmax_op)
                # print(clas.shape)
                # Create the fetch operation to retrieve the ArgMax result
                # fetch_op = argmax_op
                """ if self.sess is not None:
                    input_data = np.random.randint(0, 100, size=(clas.shape[0],clas.shape[1]))  # Sample input data
                    numpy_argmax = self.sess.run(fetch_op, feed_dict={clas: input_data})
                 """

                # tensor_list = numpy_tensor(tensor_arg)
                # print(f"numpy argmax : {numpy_argmax}")
                # clas_tf = np.eye(clas.shape[1])[numpy_argmax]
                clas_tf = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]

                # clas = tf.eye(clas.shape[1])[tf.math.argmax(clas, axis=0)]
                for (
                    metric
                ) in self.pred_metrics_list:  # only classification metrics ATM
                    history[group][metric] += [
                        self.pred_metrics_list[metric](
                            np.asarray(y_list[group].argmax(axis=1)).reshape(
                                -1,
                            ),
                            clas_tf.argmax(axis=1),
                        )
                    ]  # y_list are onehot encoded
                for (
                    metric
                ) in (
                    self.pred_metrics_list_balanced
                ):  # only classification metrics ATM
                    history[group][metric] += [
                        self.pred_metrics_list_balanced[metric](
                            np.asarray(y_list[group].argmax(axis=1)).reshape(
                                -1,
                            ),
                            clas_tf.argmax(axis=1),
                        )
                    ]  # y_list are onehot encoded
        del inp
        return history, _, clas, dann, rec

    def freeze_layers(self, ae, layers_to_freeze):
        """
        Freezes specified layers in the model.

        ae: Model to freeze layers in.
        layers_to_freeze: List of layers to freeze.
        """
        for layer in layers_to_freeze:
            layer.trainable = False

    def freeze_block(self, ae, strategy):
        if strategy == "all_but_classifier_branch":
            layers_to_freeze = [
                ae.dann_discriminator,
                ae.enc,
                ae.dec,
                ae.ae_output_layer,
            ]
        elif strategy == "all_but_classifier":
            layers_to_freeze = [
                ae.dann_discriminator,
                ae.dec,
                ae.ae_output_layer,
            ]
        elif strategy == "all_but_dann_branch":
            layers_to_freeze = [
                ae.classifier,
                ae.enc,
                ae.dec,
                ae.ae_output_layer,
            ]
        elif strategy == "all_but_dann":
            layers_to_freeze = [ae.classifier, ae.dec, ae.ae_output_layer]
        elif strategy == "all_but_autoencoder":
            layers_to_freeze = [ae.classifier, ae.dann_discriminator]
        elif strategy == "freeze_dann":
            layers_to_freeze = [ae.dann_discriminator]
        elif strategy == "freeze_dec":
            layers_to_freeze = [ae.dec]
        else:
            raise ValueError("Unknown freeze strategy: " + strategy)

        self.freeze_layers(ae, layers_to_freeze)

    def freeze_all(self, ae):
        for l in ae.layers:
            l.trainable = False

    def unfreeze_all(self, ae):
        for l in ae.layers:
            l.trainable = True

    def get_scheme(self):
        if self.training_scheme == "training_scheme_1":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, False),
                ("full_model", 100, True),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_2":
            training_scheme = [
                ("warmup_dann_no_rec", self.warmup_epoch, False),
                ("full_model", 100, True),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_3":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, False),
                ("full_model", 100, True),
                ("classifier_branch", 50, False),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_4":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, False),
                (
                    "permutation_only",
                    100,
                    True,
                ),  # This will end with a callback
                ("classifier_branch", 50, False),
            ]  # This will end with a callback
        if self.training_scheme == "training_scheme_5":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, False),
                ("full_model", 100, False),
            ]  # This will end with a callback, NO PERMUTATION HERE
        if self.training_scheme == "training_scheme_6":
            training_scheme = [
                (
                    "warmup_dann",
                    self.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", 100, True),
            ]

        if self.training_scheme == "training_scheme_7":
            training_scheme = [
                (
                    "warmup_dann",
                    self.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", 100, False),
            ]

        """ if self.training_scheme == 'training_scheme_8':
            training_scheme = [("warmup_dann", self.warmup_epoch, True), # Permutating with pseudo labels during warmup 
                                ("full_model", 100, False),
                                ("classifier_branch", 50, False)] # This will end with a callback] """
        if self.training_scheme == "training_scheme_8":
            training_scheme = [
                (
                    "warmup_dann",
                    1,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", 1, False),
                ("classifier_branch", 1, False),
            ]  # This will end with a callback]

        if self.training_scheme == "training_scheme_9":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, False),
                ("full_model", 100, True),
                ("classifier_branch", 50, False),
            ]  # This will end with a callback]

        if self.training_scheme == "training_scheme_10":
            training_scheme = [
                (
                    "warmup_dann",
                    self.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
                (
                    "warmup_dann_pseudolabels",
                    self.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels from the current model state
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
            ]  # This will end with a callback

        if self.training_scheme == "training_scheme_11":
            training_scheme = [
                (
                    "warmup_dann",
                    self.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
                (
                    "full_model_pseudolabels",
                    100,
                    True,
                ),  # using permutations on plabels for full training
                ("classifier_branch", 50, False),
            ]  # This will end with a callback

        if self.training_scheme == "training_scheme_12":
            training_scheme = [
                (
                    "permutation_only",
                    self.warmup_epoch,
                    True,
                ),  # Permutating with pseudo labels during warmup
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_13":
            training_scheme = [
                ("full_model", 50, True),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_14":
            training_scheme = [
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_15":
            training_scheme = [
                ("warmup_dann_train", self.warmup_epoch, True),
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_16":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, True),
                ("full_model", 100, True),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_17":
            training_scheme = [
                ("no_dann", 100, True),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_18":
            training_scheme = [
                ("no_dann", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_19":
            training_scheme = [
                (
                    "warmup_dann",
                    self.warmup_epoch,
                    False,
                ),  # Permutating with pseudo labels during warmup
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_20":
            training_scheme = [
                (
                    "warmup_dann_semisup",
                    self.warmup_epoch,
                    True,
                ),  # Permutating in semisup fashion ie unknown cells reconstruc themselves
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_21":
            training_scheme = [
                ("warmup_dann", self.warmup_epoch, False),
                ("no_dann", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_22":
            training_scheme = [
                ("permutation_only", self.warmup_epoch, True),
                ("warmup_dann", self.warmup_epoch, True),
                ("full_model", 100, False),
                ("classifier_branch", 50, False),
            ]

        if self.training_scheme == "training_scheme_23":
            training_scheme = [("full_model", 100, True)]

        if self.training_scheme == "training_scheme_24":
            training_scheme = [
                ("full_model", 100, False),
            ]

        if self.training_scheme == "training_scheme_25":
            training_scheme = [
                ("no_decoder", 100, False),
            ]

        return training_scheme

    def get_losses(self, y_list):
        if self.rec_loss_name == "MSE":
            self.rec_loss_fn = tf.keras.losses.MSE
        else:
            print(self.rec_loss_name + " loss not supported for rec")

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
            print(self.clas_loss_name + " loss not supported for classif")

        if self.dann_loss_name == "categorical_crossentropy":
            self.dann_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            print(self.dann_loss_name + " loss not supported for dann")
        return self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn

    def print_status_bar(self, iteration, total, loss, metrics=None):
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
        # TODO Add more optimizers
        print(optimizer_type)
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
