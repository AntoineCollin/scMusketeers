# except ImportError:
#     from dataset import Dataset, load_dataset
#     from scpermut.tools.utils import scanpy_to_input, default_value, str2bool
#     from scpermut.tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg
# from dca.utils import str2bool,tuple_to_scalar
import argparse
import functools
import gc
import os
import sys
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import tensorflow as tf
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

# try :
#     from .dataset import Dataset, load_dataset
#     from ..tools.utils import scanpy_to_input, default_value, str2bool
#     from ..tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg


sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ae_param import AE_PARAM
    from class_param import CLASS_PARAM
    from dann_param import DANN_PARAM
    from dataset import Dataset, load_dataset

    from . import freeze
except ImportError:
    from ..arguments.ae_param import AE_PARAM
    from ..arguments.class_param import CLASS_PARAM
    from ..arguments.dann_param import DANN_PARAM
    from ..workflow.dataset import Dataset, load_dataset
    from . import freeze

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


# from ax import RangeParameter, SearchSpace, ParameterType, FixedParameter, ChoiceParameter

physical_devices = tf.config.list_physical_devices("GPU")
print("tf", physical_devices)
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)


PRED_METRICS_LIST = {
    "acc": accuracy_score,
    "mcc": matthews_corrcoef,
    "f1_score": f1_score,
    "KPA": cohen_kappa_score,
    "ARI": adjusted_rand_score,
    "NMI": normalized_mutual_info_score,
    "AMI": adjusted_mutual_info_score,
}

PRED_METRICS_LIST_BALANCED = {
    "balanced_acc": balanced_accuracy_score,
    "balanced_mcc": balanced_matthews_corrcoef,
    "balanced_f1_score": balanced_f1_score,
    "balanced_KPA": balanced_cohen_kappa_score,
}

CLUSTERING_METRICS_LIST = {
    "db_score": davies_bouldin_score
}  #'clisi' : lisi_avg

BATCH_METRICS_LIST = {"batch_mixing_entropy": batch_entropy_mixing_score}


class Workflow:
    def __init__(self, run_file, working_dir):
        """
        run_file : a dictionary outputed by the function load_run_file
        """
        # print("NOOOOO EAGERRRRR")
        # tf.compat.v1.disable_eager_execution()
        self.process = run_file.process
        # self.sess = tf.compat.v1.Session()
        self.run_file = run_file
        self.ae_param = AE_PARAM(run_file)
        self.class_param = CLASS_PARAM(run_file)
        self.dann_param = DANN_PARAM(run_file)

        self.run_neptune = None
        self.hp_params = None

        self.dataset = None
        self.model = None
        self.predictor = None
        self.dann_ae = None

        self.metrics = []
        self.working_dir = working_dir
        self.data_dir = working_dir + "/data"

        ### USED ?
        self.n_perm = 1
        self.semi_sup = False  # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
        self.unlabeled_category = "UNK"  # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves

        self.training_kwds = {}
        self.network_kwds = {}

        ##### TODO : Add to run_file

        self.clas_loss_fn = None
        self.dann_loss_fn = None
        self.rec_loss_fn = None
        self.num_classes = None
        self.num_batches = None
        self.mean_loss_fn = keras.metrics.Mean(
            name="total_loss"
        )  # This is a running average : it keeps the previous values in memory when it's called ie computes the previous and current values
        self.mean_clas_loss_fn = keras.metrics.Mean(name="classification_loss")
        self.mean_dann_loss_fn = keras.metrics.Mean(name="dann_loss")
        self.mean_rec_loss_fn = keras.metrics.Mean(name="reconstructionloss")

    def make_workflow(self):
        if self.run_file.layer1:
            self.ae_param.ae_hidden_size = [
                self.run_file.layer1,
                self.run_file.layer2,
                self.run_file.bottleneck,
                self.run_file.layer2,
                self.run_file.layer1,
            ]

        if self.run_file.dropout:
            self.dann_param.dann_hidden_dropout, self.class_param.class_hidden_dropout,
            self.ae_param.ae_hidden_dropout = (self.run_file.dropout,)
            self.run_file.dropout, self.run_file.dropout

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

        # print({i:adata_list[i] for i in adata_list})
        # print({i:len(y_list[i]) for i in y_list})
        # print(f"sum : {len(y_list['train']) + len(y_list['test']) + len(y_list['val'])}")
        # print(f"full: {len(y_list['full'])}")

        self.num_classes = len(np.unique(self.dataset.y_train))
        self.num_batches = len(np.unique(self.dataset.batch))

        bottleneck_size = int(
            self.ae_param.ae_hidden_size[
                int(len(self.ae_param.ae_hidden_size) / 2)
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
        ######## TO DO fix params with AE_PARAM, DANN_PARAM and CLASS_PARAM
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

        # Get optimizer object
        self.optimizer = get_optimizer(
            self.run_file.learning_rate,
            self.run_file.weight_decay,
            self.run_file.optimizer_type,
        )

        # Get loss functions
        self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn = (
            self.get_losses()
        )  # redundant

        # Training

        self.run_file.training_scheme = self.get_scheme()
        start_time = time.time()
        history = self.train_scheme(
            training_scheme=self.run_file.training_scheme,
            verbose=self.run_file.verbose,
            ae=self.dann_ae,
            adata_list=adata_list,
            X_list=X_list,
            y_list=y_list,
            batch_list=batch_list,
            #  optimizer= self.optimizer, # not an **loop_param since it resets between strategies
            clas_loss_fn=self.clas_loss_fn,
            dann_loss_fn=self.dann_loss_fn,
            rec_loss_fn=self.rec_loss_fn,
        )
        stop_time = time.time()

        # Reporting
        self.reporting(stop_time, start_time, adata_list, batch_list)

        # Get optimization metric
        split, metric = self.run_file.opt_metric.split("-")
        self.run_neptune.wait()
        opt_metric = self.run_neptune[f"evaluation/{split}/{metric}"].fetch()
        if self.run_file.verbose:
            print(f"opt_metric {opt_metric}")

        tf.keras.backend.clear_session()

        return opt_metric

    def reporting(self, stop_time, start_time, adata_list, batch_list):
        # Reporting after training
        if self.run_file.log_neptune:
            self.run_neptune["evaluation/training_time"] = (
                stop_time - start_time
            )
            neptune_run_id = self.run_neptune["sys/id"].fetch()
            save_dir = (
                self.working_dir
                + "experiment_script/results/"
                + str(neptune_run_id)
                + "/"
            )
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            y_true_full = adata_list["full"].obs[
                f"true_{self.run_file.class_key}"
            ]
            ct_prop = (
                pd.Series(y_true_full).value_counts()
                / pd.Series(y_true_full).value_counts().sum()
            )
            sizes = {
                "xxsmall": list(ct_prop[ct_prop < 0.001].index),
                "small": list(
                    ct_prop[(ct_prop >= 0.001) & (ct_prop < 0.01)].index
                ),
                "medium": list(
                    ct_prop[(ct_prop >= 0.01) & (ct_prop < 0.1)].index
                ),
                "large": list(ct_prop[ct_prop >= 0.1].index),
            }

            for group in ["full", "train", "val", "test"]:
                with tf.device("GPU"):
                    input_tensor = {
                        k: tf.convert_to_tensor(v)
                        for k, v in scanpy_to_input(
                            adata_list[group], ["size_factors"]
                        ).items()
                    }
                    enc, clas, dann, rec = self.dann_ae(
                        input_tensor, training=False
                    ).values()  # Model predict

                    if (
                        group == "full"
                    ):  # saving full predictions as probability output from the classifier
                        y_pred_proba = pd.DataFrame(
                            np.asarray(clas),
                            index=adata_list["full"].obs_names,
                            columns=self.dataset.ohe_celltype.categories_[0],
                        )
                        y_pred_proba.to_csv(
                            save_dir + f"y_pred_proba_full.csv"
                        )
                        self.run_neptune[
                            f"evaluation/{group}/y_pred_proba_full"
                        ].track_files(save_dir + f"y_pred_proba_full.csv")

                    # In Eager Mode
                    numpy_argmax = np.argmax(clas, axis=1)

                    # In Graph mode
                    """ argmax_op = tf.math.argmax(clas, axis=1)
                    fetch_op = argmax_op
                    if self.sess is not None:
                        input_data = np.random.randint(0, 100, size=(clas.shape[0],clas.shape[1]))  # Sample input data
                        numpy_argmax = self.sess.run(fetch_op, feed_dict={clas: input_data}) """

                    # Get Classif hot encoding
                    class_tf = np.eye(clas.shape[1])[numpy_argmax]

                    y_pred = self.dataset.ohe_celltype.inverse_transform(
                        class_tf
                    ).reshape(
                        -1,
                    )
                    y_true = adata_list[group].obs[
                        f"true_{self.run_file.class_key}"
                    ]
                    batches = np.asarray(
                        batch_list[group].argmax(axis=1)
                    ).reshape(
                        -1,
                    )
                    split = adata_list[group].obs[f"train_split"]

                    # Saving confusion matrices
                    labels = list(
                        set(np.unique(y_true)).union(set(np.unique(y_pred)))
                    )
                    cm_no_label = confusion_matrix(y_true, y_pred)
                    print(f"no label : {cm_no_label.shape}")
                    cm = confusion_matrix(y_true, y_pred, labels=labels)
                    cm_norm = cm / cm.sum(axis=1, keepdims=True)
                    print(f"label : {cm.shape}")
                    cm_to_plot = pd.DataFrame(
                        cm_norm, index=labels, columns=labels
                    )
                    cm_to_save = pd.DataFrame(cm, index=labels, columns=labels)
                    cm_to_plot = cm_to_plot.fillna(value=0)
                    cm_to_save = cm_to_save.fillna(value=0)
                    cm_to_save.to_csv(
                        save_dir + f"confusion_matrix_{group}.csv"
                    )
                    self.run_neptune[
                        f"evaluation/{group}/confusion_matrix_file"
                    ].track_files(save_dir + f"confusion_matrix_{group}.csv")
                    size = len(labels)
                    f, ax = plt.subplots(figsize=(size / 1.5, size / 1.5))
                    sns.heatmap(
                        cm_to_plot,
                        annot=True,
                        ax=ax,
                        fmt=".2f",
                        vmin=0,
                        vmax=1,
                    )
                    show_mask = np.asarray(cm_to_plot > 0.01)
                    print(f"label df : {cm_to_plot.shape}")
                    for text, show_annot in zip(
                        ax.texts,
                        (element for row in show_mask for element in row),
                    ):
                        text.set_visible(show_annot)

                    self.run_neptune[
                        f"evaluation/{group}/confusion_matrix"
                    ].upload(f)

                    # Computing batch mixing metrics
                    if (
                        len(
                            np.unique(
                                np.asarray(batch_list[group].argmax(axis=1))
                            )
                        )
                        >= 2
                    ):  # If there are more than 2 batches in this group
                        for metric in BATCH_METRICS_LIST:
                            self.run_neptune[
                                f"evaluation/{group}/{metric}"
                            ] = BATCH_METRICS_LIST[metric](enc, batches)
                            print(
                                type(BATCH_METRICS_LIST[metric](enc, batches))
                            )

                    # Computing classification metrics
                    for metric in PRED_METRICS_LIST:
                        self.run_neptune[f"evaluation/{group}/{metric}"] = (
                            PRED_METRICS_LIST[metric](y_true, y_pred)
                        )

                    for metric in PRED_METRICS_LIST_BALANCED:
                        self.run_neptune[f"evaluation/{group}/{metric}"] = (
                            PRED_METRICS_LIST_BALANCED[metric](y_true, y_pred)
                        )

                    # Metrics by size of ct
                    for s in sizes:
                        idx_s = np.isin(
                            y_true, sizes[s]
                        )  # Boolean array, no issue to index y_pred
                        y_true_sub = y_true[idx_s]
                        y_pred_sub = y_pred[idx_s]
                        # print(s)
                        for metric in PRED_METRICS_LIST:
                            self.run_neptune[
                                f"evaluation/{group}/{s}/{metric}"
                            ] = nan_to_0(
                                PRED_METRICS_LIST[metric](
                                    y_true_sub, y_pred_sub
                                )
                            )

                        for metric in PRED_METRICS_LIST_BALANCED:
                            self.run_neptune[
                                f"evaluation/{group}/{s}/{metric}"
                            ] = nan_to_0(
                                PRED_METRICS_LIST_BALANCED[metric](
                                    y_true_sub, y_pred_sub
                                )
                            )

                    # Computing clustering metrics
                    for metric in CLUSTERING_METRICS_LIST:
                        self.run_neptune[f"evaluation/{group}/{metric}"] = (
                            CLUSTERING_METRICS_LIST[metric](enc, y_pred)
                        )

                    if group == "full":
                        y_pred_df = pd.DataFrame(
                            {"pred": y_pred, "true": y_true, "split": split},
                            index=adata_list[group].obs_names,
                        )
                        split = pd.DataFrame(
                            split, index=adata_list[group].obs_names
                        )
                        np.save(
                            save_dir + f"latent_space_{group}.npy", enc.numpy()
                        )
                        y_pred_df.to_csv(save_dir + f"predictions_{group}.csv")
                        split.to_csv(save_dir + f"split_{group}.csv")
                        self.run_neptune[
                            f"evaluation/{group}/latent_space"
                        ].track_files(save_dir + f"latent_space_{group}.npy")
                        self.run_neptune[
                            f"evaluation/{group}/predictions"
                        ].track_files(save_dir + f"predictions_{group}.csv")

                        # Saving umap representation
                        pred_adata = sc.AnnData(
                            X=adata_list[group].X,
                            obs=adata_list[group].obs,
                            var=adata_list[group].var,
                        )
                        pred_adata.obs[f"{self.run_file.class_key}_pred"] = (
                            y_pred_df["pred"]
                        )
                        pred_adata.obsm["latent_space"] = enc.numpy()
                        sc.pp.neighbors(pred_adata, use_rep="latent_space")
                        sc.tl.umap(pred_adata)
                        np.save(
                            save_dir + f"umap_{group}.npy",
                            pred_adata.obsm["X_umap"],
                        )
                        self.run_neptune[
                            f"evaluation/{group}/umap"
                        ].track_files(save_dir + f"umap_{group}.npy")
                        sc.set_figure_params(figsize=(15, 10), dpi=300)
                        fig_class = sc.pl.umap(
                            pred_adata,
                            color=f"true_{self.run_file.class_key}",
                            size=10,
                            return_fig=True,
                        )
                        fig_pred = sc.pl.umap(
                            pred_adata,
                            color=f"{self.run_file.class_key}_pred",
                            size=10,
                            return_fig=True,
                        )
                        fig_batch = sc.pl.umap(
                            pred_adata,
                            color=self.run_file.batch_key,
                            size=10,
                            return_fig=True,
                        )
                        fig_split = sc.pl.umap(
                            pred_adata,
                            color="train_split",
                            size=10,
                            return_fig=True,
                        )
                        self.run_neptune[
                            f"evaluation/{group}/true_umap"
                        ].upload(fig_class)
                        self.run_neptune[
                            f"evaluation/{group}/pred_umap"
                        ].upload(fig_pred)
                        self.run_neptune[
                            f"evaluation/{group}/batch_umap"
                        ].upload(fig_batch)
                        self.run_neptune[
                            f"evaluation/{group}/split_umap"
                        ].upload(fig_split)

    def train_scheme(self, training_scheme, verbose, **loop_params):
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
            for m in PRED_METRICS_LIST:
                history[group][m] = []

        # if self.log_neptune:
        #     for group in history:
        #         for par,val in history[group].items():
        #             self.run_neptune[f"training/{group}/{par}"] = []
        i = 0

        total_epochs = np.sum([n_epochs for _, n_epochs, _ in training_scheme])
        running_epoch = 0

        for strategy, n_epochs, use_perm in training_scheme:
            optimizer = get_optimizer(
                self.run_file.learning_rate,
                self.run_file.weight_decay,
                self.run_file.optimizer_type,
            )  # resetting optimizer state when switching strategy
            if self.run_file.verbose:
                print(
                    f"Step number {i}, running {strategy} strategy with permutation = {use_perm} for {n_epochs} epochs"
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
                patience = 10
                min_delta = 0.01
                if strategy == "permutation_only":
                    monitored = "rec_loss"
                    es_best = np.inf  # initialize early_stopping
                else:
                    monitored = "mcc"
                    es_best = -np.inf

            # tracker_class = ClassTracker()
            # self.tr = tracker.SummaryTracker()
            # tracker_class.track_class(Workflow, resolution_level=3, trace=2)
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

                if self.run_file.log_neptune:
                    for group in history:
                        for par, value in history[group].items():
                            self.run_neptune[f"training/{group}/{par}"].append(
                                value[-1]
                            )
                            """ if physical_devices:
                                # print(f"memory {tf.config.experimental.get_memory_info('GPU:0')['current']}")
                                self.run_neptune["training/train/tf_GPU_memory_epoch"].append(
                                    tf.config.experimental.get_memory_info("GPU:0")[
                                        "current"
                                    ]
                                    / 1e6
                                ) """
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
        if self.run_file.log_neptune:
            self.run_neptune[f"training/{group}/total_epochs"] = running_epoch
        return history

    def training_loop(
        self,
        history,
        ae,
        adata_list,
        X_list,
        y_list,
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

        freeze.unfreeze_all(ae)  # resetting freeze state
        if training_strategy == "full_model":
            group = "train"
        elif training_strategy == "warmup_dann":
            group = "full"  # unsupervised setting
            ae.classifier.trainable = False  # Freezing classifier just to be sure but should not be necessary since gradient won't be propagating in this branch
            use_perm = False  # no permutation for warming up the dann. No need to specify it in the no rec version since we don't reconstruct
        elif training_strategy == "warmup_dann_no_rec":
            group = "full"
            freeze.freeze_block(ae, "all_but_dann")
        elif training_strategy == "dann_with_ae":
            group = "train"
            ae.classifier.trainable = False
            use_perm = True
        elif training_strategy == "classifier_branch":
            group = "train"
            freeze.freeze_block(
                ae, "all_but_classifier_branch"
            )  # traning only classifier branch
        elif training_strategy == "permutation_only":
            group = "train"
            freeze.freeze_block(ae, "all_but_autoencoder")
            use_perm = True

        if not use_perm:
            use_perm = True

        batch_generator = batch_generator_training_permuted(
            X=X_list[group],
            y=y_list[group],
            batch_ID=batch_list[group],
            sf=adata_list[group].obs["size_factors"],
            ret_input_only=False,
            batch_size=self.run_file.batch_size,
            n_perm=1,
            use_perm=use_perm,
        )
        n_obs = adata_list[group].n_obs
        # print(f"obs {n_obs}")
        # print(adata_list[group])
        steps = n_obs // self.run_file.batch_size + 1
        n_steps = steps
        # print(f"obs {n_steps}")

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
        #         self.run_neptune['memory/'+str(key)].append(value)
        #         print(value)
        # all objects
        # all_objects = muppy.get_objects()
        # print("allobjects "+str(len(all_objects)))
        # my_types = muppy.filter(all_objects, Type=type)
        # print(len(my_types))
        # for t in my_types:
        #     print(t)
        # tracker_class = ClassTracker()
        # tracker_class.track_class(Workflow, resolution_level=3, trace=2)
        # tracker_class.create_snapshot()
        # tracker_class.stats.print_summary()
        # print(f"New step : {step}")
        # gpu_mem = []
        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])
        self.run_neptune["training/train/tf_GPU_memory_step"].append(
            tf.config.experimental.get_memory_info("GPU:0")["current"] / 1e6
        )
        self.run_neptune["training/train/step"].append(step)
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
        """ self.mean_loss_fn(loss.__float__())
        self.mean_clas_loss_fn(clas_loss.__float__())
        self.mean_dann_loss_fn(dann_loss.__float__())
        self.mean_rec_loss_fn(rec_loss.__float__()) """

        if self.run_file.verbose:
            """print_status_bar(
                n_samples,
                n_obs,
                [
                    self.mean_loss_fn,
                    self.mean_clas_loss_fn,
                    self.mean_dann_loss_fn,
                    self.mean_rec_loss_fn,
                ],
                self.metrics,
            )"""

        # gpu_mem.append(tf.config.experimental.get_memory_info("GPU:0")["current"])

        # gpu_mem_diff = []
        # for i in range(len(gpu_mem)-1):
        #     gpu_mem_diff.append(gpu_mem[i+1] - gpu_mem[i])
        # print(gpu_mem_diff)
        # print(np.sum(gpu_mem_diff))
        # self.tr.print_diff()

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
                """ clas_loss = tf.reduce_mean(clas_loss_fn(y_list[group], clas)).numpy()
                history[group]['clas_loss'] += [clas_loss]
                dann_loss = tf.reduce_mean(dann_loss_fn(batch_list[group], dann)).numpy()
                history[group]['dann_loss'] += [dann_loss]
                rec_loss = tf.reduce_mean(rec_loss_fn(X_list[group], rec)).numpy()
                history[group]['rec_loss'] += [rec_loss] """
                history[group]["total_loss"] += [
                    self.run_file.clas_w * clas_loss
                    + self.run_file.dann_w * dann_loss
                    + self.run_file.rec_w * rec_loss
                    + ae.losses
                ]  # using numpy to prevent memory leaks
                # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

                # In Eager Mode
                numpy_argmax = np.argmax(clas, axis=1)

                # In Graph mode
                """ argmax_op = tf.math.argmax(clas, axis=1)
                fetch_op = argmax_op
                if self.sess is not None:
                    input_data = np.random.randint(0, 100, size=(clas.shape[0],clas.shape[1]))  # Sample input data
                    numpy_argmax = self.sess.run(fetch_op, feed_dict={clas: input_data}) """

                # Get Classif hot encoding
                class_tf = np.eye(clas.shape[1])[numpy_argmax]

                # clas = tf.eye(clas.shape[1])[tf.math.argmax(clas, axis=0)]
                for (
                    metric
                ) in PRED_METRICS_LIST:  # only classification metrics ATM
                    history[group][metric] += [
                        PRED_METRICS_LIST[metric](
                            np.asarray(y_list[group].argmax(axis=1)).reshape(
                                -1,
                            ),
                            class_tf.argmax(axis=1),
                        )
                    ]  # y_list are onehot encoded
        del inp
        return history, _, clas, dann, rec

    def get_scheme(self):
        if self.run_file.training_scheme == "training_scheme_1":
            print(f"WARMUPDANNNNNNNN {self.run_file.warmup_epoch}")
            print(f"COMPLETE {self.run_file.fullmodel_epoch}")
            self.run_file.training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                ("full_model", self.run_file.fullmodel_epoch, False),
            ]  # This will end with a callback
        if self.run_file.training_scheme == "training_scheme_2":
            self.run_file.training_scheme = [
                ("warmup_dann", self.run_file.warmup_epoch, False),
                (
                    "permutation_only",
                    self.run_file.permonly_epoch,
                    True,
                ),  # This will end with a callback
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]  # This will end with a callback
        if self.run_file.training_scheme == "training_scheme_3":
            self.run_file.training_scheme = [
                (
                    "permutation_only",
                    self.run_file.permonly_epoch,
                    True,
                ),  # This will end with a callback
                ("classifier_branch", self.run_file.classifier_epoch, False),
            ]
        return self.run_file.training_scheme

    def get_losses(self):
        if self.run_file.rec_loss_name == "MSE":
            self.rec_loss_fn = tf.keras.losses.mean_squared_error
        else:
            print(self.run_file.rec_loss_name + " loss not supported for rec")

        if self.run_file.clas_loss_name == "categorical_crossentropy":
            self.clas_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            print(
                self.run_file.clas_loss_name
                + " loss not supported for classif"
            )

        if self.run_file.dann_loss_name == "categorical_crossentropy":
            self.dann_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            print(
                self.run_file.dann_loss_name + " loss not supported for dann"
            )
        return self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn


def get_optimizer(learning_rate, weight_decay, optimizer_type, momentum=0.9):
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


def print_status_bar(iteration, total, loss, metrics=None):
    """metrics = ' - '.join(['{}: {:.4f}'.format(m.name, m.result())
    for m in loss + (metrics or [])])"""

    end = "" if int(iteration) < int(total) else "\n"
    #     print(f"{iteration}/{total} - "+metrics ,end="\r")
    #     print(f"\r{iteration}/{total} - " + metrics, end=end)
    print("\r{}/{} - ".format(iteration, total) + str(metrics), end=end)


if __name__ == "__main__":
    print("Load optimize_hp")
