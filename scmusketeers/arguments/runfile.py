import argparse

import str2bool

"""Parses command-line arguments for a workflow involving model training and hyperparameter optimization."""

PROCESS_TYPE = ["transfer", "optim"]


def get_runfile():
    parser = create_argparser()
    return parser.parse_args()


def create_argparser():
    """Parses command-line arguments and returns a namespace containing parsed arguments.

    Returns:
        argparse.Namespace: A namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="sc-musketeers",
        usage="sc-musketeers [OPTIONS] process atlas_name your_search_folder/",
        description="musketeers....",
        epilog="Enjoy scMusketeers!",
    )

    parser.add_argument(
        "process",
        type=str,
        help="Type of process to run : Training, Hyperparameter optimization"
        f" among {PROCESS_TYPE}",
        default="",
    )

    parser.add_argument(
        "ref_path",
        type=str,
        help="Path of the referent adata file (example : data/ajrccm.h5ad",
        default="",
    )

    parser.add_argument(
        "--class_key",
        type=str,
        help="Key of the celltype to classify",
        default="celltype",
    )

    parser.add_argument(
        "--batch_key",
        type=str,
        help="Key of the batches",
        default="manip",
    )

    # Working dir arguments
    workflow_group = parser.add_argument_group("Worklow parameters")
    workflow_group.add_argument(
        "--query_path",
        type=str,
        nargs="?",
        default=None,
        help="Optional query dataset",
    )
    workflow_group.add_argument(
        "--out_dir",
        type=str,
        nargs="?",
        default=".",
        help="The output directory",
    )
    workflow_group.add_argument(
        "--out_name",
        type=str,
        nargs="?",
        default=".",
        help="The output naming",
    )
    workflow_group.add_argument(
        "--training_scheme",
        type=str,
        nargs="?",
        default="training_scheme_13",
        help="",
    )
    workflow_group.add_argument(
        "--log_neptune",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    workflow_group.add_argument(
        "--hparam_path", type=str, nargs="?", default=None, help=""
    )
    workflow_group.add_argument(
        "--opt_metric",
        type=str,
        nargs="?",
        default="val-balanced_mcc",
        help="The metric top optimize in hp search as it appears in neptune (split-metricname)",
    )
    workflow_group.add_argument(
        "--verbose", type=str2bool.str2bool, default=True, help=""
    )

    # Dataset arguments
    dataset_group = parser.add_argument_group("Dataset Parameters")
    dataset_group.add_argument(
        "--unlabeled_category",
        type=str,
        nargs="?",
        default="UNK",
        help="Mandatory if only one dataset is passed. Tag of the cells to predict.",
    )
    dataset_group.add_argument(
        "--filter_min_counts",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Filters genes with <1 counts",
    )  # TODO :remove, we always want to do that
    dataset_group.add_argument(
        "--normalize_size_factors",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Weither to normalize dataset or not",
    )
    dataset_group.add_argument(
        "--size_factor",
        type=str,
        nargs="?",
        const="default",
        default="default",
        help='Which size factor to use. "default" computes size factor on the chosen level of preprocessing. "raw" uses size factor computed on raw data as n_counts/median(n_counts). "constant" uses a size factor of 1 for every cells',
    )
    dataset_group.add_argument(
        "--scale_input",
        type=str2bool.str2bool,
        nargs="?",
        const=False,
        default=False,
        help="Weither to scale input the count values",
    )
    dataset_group.add_argument(
        "--logtrans_input",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Weither to log transform count values",
    )
    dataset_group.add_argument(
        "--use_hvg",
        type=int,
        nargs="?",
        const=5000,
        default=None,
        help="Number of hvg to use. If no tag, don't use hvg.",
    )

    # Training Parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default=128,
        help="Training batch size",
    )  # Default identified with hp optimization
    training_group.add_argument(
        "--test_split_key",
        type=str,
        default="TRAIN_TEST_split",
        help="key of obs containing the test split",
    )
    training_group.add_argument(
        "--test_obs",
        type=str,
        nargs="+",
        default=None,
        help="batches from batch_key to use as test",
    )
    training_group.add_argument(
        "--test_index_name",
        type=str,
        nargs="+",
        default=None,
        help="indexes to be used as test. Overwrites test_obs",
    )
    training_group.add_argument(
        "--mode",
        type=str,
        default="percentage",
        help="Train test split mode to be used by Dataset.train_split",
    )
    training_group.add_argument(
        "--pct_split", type=float, nargs="?", default=0.9, help=""
    )
    training_group.add_argument(
        "--obs_key", type=str, nargs="?", default="manip", help=""
    )
    training_group.add_argument(
        "--n_keep",
        type=int,
        nargs="?",
        default=None,
        help="batches from obs_key to use as train",
    )
    training_group.add_argument(
        "--split_strategy", type=str, nargs="?", default=None, help=""
    )
    training_group.add_argument(
        "--keep_obs", type=str, nargs="+", default=None, help=""
    )
    training_group.add_argument(
        "--train_test_random_seed", type=float, nargs="?", default=0, help=""
    )
    training_group.add_argument(
        "--obs_subsample", type=str, nargs="?", default=None, help=""
    )
    training_group.add_argument(
        "--make_fake",
        type=str2bool.str2bool,
        nargs="?",
        const=False,
        default=False,
        help="",
    )
    training_group.add_argument(
        "--true_celltype", type=str, nargs="?", default=None, help=""
    )
    training_group.add_argument(
        "--false_celltype", type=str, nargs="?", default=None, help=""
    )
    training_group.add_argument(
        "--pct_false", type=float, nargs="?", default=None, help=""
    )
    training_group.add_argument(
        "--weight_decay",
        type=float,
        nargs="?",
        default=2e-6,
        help="Weight decay applied by th optimizer",
    )  # Default identified with hp optimization
    training_group.add_argument(
        "--learning_rate",
        type=float,
        nargs="?",
        default=0.001,
        help="Starting learning rate for training",
    )  # Default identified with hp optimization
    training_group.add_argument(
        "--optimizer_type",
        type=str,
        nargs="?",
        choices=["adam", "adamw", "rmsprop"],
        default="adam",
        help="Name of the optimizer to use",
    )

    # epoch groups
    epoch_group = parser.add_argument_group("Epoch Parameters")
    epoch_group.add_argument(
        "--warmup_epoch",
        type=int,
        nargs="?",
        default=100,
        help="Number of epoch to warmup DANN",
    )
    epoch_group.add_argument(
        "--fullmodel_epoch",
        type=int,
        nargs="?",
        default=100,
        help="Number of epoch to train full model",
    )
    epoch_group.add_argument(
        "--permonly_epoch",
        type=int,
        nargs="?",
        default=100,
        help="Number of epoch to train in permutation only mode",
    )
    epoch_group.add_argument(
        "--classifier_epoch",
        type=int,
        nargs="?",
        default=50,
        help="Number of epoch to train te classifier only",
    )

    # Loss function Arguments
    loss_group = parser.add_argument_group("Loss function Parameters")
    loss_group.add_argument(
        "--balance_classes",
        type=str2bool.str2bool,
        nargs="?",
        default=False,
        help="Add balance to weight to the loss",
    )
    loss_group.add_argument(
        "--clas_loss_name",
        type=str,
        nargs="?",
        choices=["MSE", "categorical_crossentropy"],
        default="categorical_crossentropy",
        help="Loss of the classification branch",
    )
    loss_group.add_argument(
        "--dann_loss_name",
        type=str,
        nargs="?",
        choices=["categorical_crossentropy"],
        default="categorical_crossentropy",
        help="Loss of the DANN branch",
    )
    loss_group.add_argument(
        "--rec_loss_name",
        type=str,
        nargs="?",
        choices=["MSE"],
        default="MSE",
        help="Reconstruction loss of the autoencoder",
    )
    loss_group.add_argument(
        "--clas_w",
        type=float,
        nargs="?",
        default=0.1,
        help="Weight of the classification loss",
    )
    loss_group.add_argument(
        "--dann_w",
        type=float,
        nargs="?",
        default=0.1,
        help="Weight of the DANN loss",
    )
    loss_group.add_argument(
        "--rec_w",
        type=float,
        nargs="?",
        default=0.8,
        help="Weight of the reconstruction loss",
    )

    # Model Architecture arguments
    architecture_group = parser.add_argument_group("Model Architecture")
    architecture_group.add_argument(
        "--dropout",
        type=int,
        nargs="?",
        default=None,
        help="dropout applied to every layers of the model. If specified, overwrites other dropout arguments",
    )
    architecture_group.add_argument(
        "--layer1",
        type=int,
        nargs="?",
        default=None,
        help="size of the first layer for a 2-layers model. If specified, overwrites ae_hidden_size",
    )
    architecture_group.add_argument(
        "--layer2",
        type=int,
        nargs="?",
        default=None,
        help="size of the second layer for a 2-layers model. If specified, overwrites ae_hidden_size",
    )
    architecture_group.add_argument(
        "--bottleneck",
        type=int,
        nargs="?",
        default=None,
        help="size of the bottleneck layer. If specified, overwrites ae_hidden_size",
    )

    # Autoencoder arguments
    ae_group = parser.add_argument_group("Autoencoder Architecture")
    ae_group.add_argument(
        "--ae_hidden_size",
        type=int,
        nargs="+",
        default=[128, 64, 128],
        help="Hidden sizes of the successive ae layers",
    )
    ae_group.add_argument(
        "--ae_hidden_dropout", type=float, nargs="?", default=None, help=""
    )
    ae_group.add_argument(
        "--ae_activation", type=str, nargs="?", default="relu", help=""
    )
    ae_group.add_argument(
        "--ae_bottleneck_activation",
        type=str,
        nargs="?",
        default="linear",
        help="activation of the bottleneck layer",
    )
    ae_group.add_argument(
        "--ae_output_activation", type=str, nargs="?", default="relu", help=""
    )
    ae_group.add_argument(
        "--ae_init", type=str, nargs="?", default="glorot_uniform", help=""
    )
    ae_group.add_argument(
        "--ae_batchnorm",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    ae_group.add_argument(
        "--ae_l1_enc_coef", type=float, nargs="?", default=None, help=""
    )
    ae_group.add_argument(
        "--ae_l2_enc_coef", type=float, nargs="?", default=None, help=""
    )

    # Classification model arguments
    class_group = parser.add_argument_group("Classification Architecture")
    class_group.add_argument(
        "--class_hidden_size",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden sizes of the successive classification layers",
    )
    class_group.add_argument(
        "--class_hidden_dropout", type=float, nargs="?", default=None, help=""
    )
    class_group.add_argument(
        "--class_batchnorm",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    class_group.add_argument(
        "--class_activation", type=str, nargs="?", default="relu", help=""
    )
    class_group.add_argument(
        "--class_output_activation",
        type=str,
        nargs="?",
        default="softmax",
        help="",
    )

    # Dann model arguments
    dann_group = parser.add_argument_group("DANN Architecture")
    dann_group.add_argument(
        "--dann_hidden_size", type=int, nargs="?", default=[64], help=""
    )
    dann_group.add_argument(
        "--dann_hidden_dropout", type=float, nargs="?", default=None, help=""
    )
    dann_group.add_argument(
        "--dann_batchnorm",
        type=str2bool.str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    dann_group.add_argument(
        "--dann_activation", type=str, nargs="?", default="relu", help=""
    )
    dann_group.add_argument(
        "--dann_output_activation",
        type=str,
        nargs="?",
        default="softmax",
        help="",
    )

    return parser


def set_hyperparameters(workflow, params):
    workflow.run_file.use_hvg = params["use_hvg"]
    workflow.run_file.batch_size = params["batch_size"]
    workflow.run_file.clas_w = params["clas_w"]
    workflow.run_file.dann_w = params["dann_w"]
    if "rec_w" in params:
        workflow.run_file.rec_w = params["rec_w"]
    if "ae_bottleneck_activation" in params:
        workflow.ae_param.ae_bottleneck_activation = params[
            "ae_bottleneck_activation"
        ]
    if "size_factor" in params:
        workflow.run_file.size_factor = params["size_factor"]
    if "weight_decay" in params:
        workflow.run_file.weight_decay = params["weight_decay"]
    workflow.run_file.learning_rate = params["learning_rate"]
    workflow.run_file.warmup_epoch = params["warmup_epoch"]
    workflow.run_file.dropout = params["dropout"]
    workflow.run_file.layer1 = params["layer1"]
    workflow.run_file.layer2 = params["layer2"]
    workflow.run_file.bottleneck = params["bottleneck"]
    workflow.run_file.hp_params = params
