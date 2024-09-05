import argparse

from workflow.hyperparameters import *

JSON_PATH_DEFAULT = "/home/acollin/scPermut/experiment_script/hp_ranges/"


def load_json(json_path):
    with open(json_path, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


total_trial = 50
random_seed = 40


class MakeExperiment:
    def __init__(self, run_file, working_dir):
        # super().__init__()
        self.run_file = run_file
        self.working_dir = working_dir
        self.workflow = None
        self.trial_count = 0
        project = neptune.init_project(
            project="becavin-lab/benchmark",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
            mode="read-only",
        )  # For checkpoint
        self.runs_table_df = project.fetch_runs_table().to_pandas()
        project.stop()

    def train(self, params):
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        # import tensorflow as tf

        self.trial_count += 1
        # print('params')
        # print(params)
        checkpoint = {"parameters/" + k: i for k, i in params.items()}
        checkpoint["parameters/dataset_name"] = self.run_file.dataset_name
        checkpoint["parameters/opt_metric"] = self.run_file.opt_metric
        checkpoint["parameters/task"] = "hp_optim_V2"
        print(checkpoint)
        # checkpoint = {'parameters/dataset_name': self.run_file.dataset_name,
        #               'parameters/total_trial': total_trial, 'parameters/trial_count': self.trial_count,
        #               'parameters/opt_metric': self.opt_metric, 'parameters/hp_random_seed': random_seed}
        result = self.runs_table_df[
            self.runs_table_df[list(checkpoint.keys())]
            .eq(list(checkpoint.values()))
            .all(axis=1)
        ]
        print(result)
        split, metric = self.run_file.opt_metric.split("-")
        if result.empty or pd.isna(
            result.loc[:, f"evaluation/{split}/{metric}"].iloc[-1]
        ):  # we run the trial
            print(f"Trial {self.trial_count} does not exist, running trial")
            self.workflow = Workflow(
                run_file=self.run_file, working_dir=self.working_dir
            )
            self.workflow.set_hyperparameters(params)
            self.workflow.start_neptune_log()
            self.workflow.process_dataset()
            self.workflow.split_train_test()
            self.workflow.split_train_val()
            opt_metric = (
                self.workflow.make_experiment()
            )  # This starts the logging
            self.workflow.add_custom_log("task", "hp_optim_V2")
            self.workflow.add_custom_log("total_trial", total_trial)
            self.workflow.add_custom_log("hp_random_seed", random_seed)
            self.workflow.add_custom_log("trial_count", self.trial_count)
            self.workflow.add_custom_log(
                "opt_metric", self.run_file.opt_metric
            )
            self.workflow.stop_neptune_log()
            # del self.workflow  # Should not be necessary
            return opt_metric
        else:  # we return the already computed value
            print(f"Trial {self.trial_count} already exists, retrieving value")
            return result.loc[:, f"evaluation/{split}/{metric}"].iloc[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--working_dir",
        type=str,
        nargs="?",
        default="",
        help="The working directory",
    )

    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="htap_final_by_batch",
        help="Name of the dataset to use, should indicate a raw h5ad AnnData file",
    )
    parser.add_argument(
        "--class_key",
        type=str,
        default="celltype",
        help="Key of the class to classify",
    )
    parser.add_argument(
        "--batch_key", type=str, default="donor", help="Key of the batches"
    )
    parser.add_argument(
        "--filter_min_counts",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Filters genes with <1 counts",
    )  # TODO :remove, we always want to do that
    parser.add_argument(
        "--normalize_size_factors",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Weither to normalize dataset or not",
    )
    parser.add_argument(
        "--size_factor",
        type=str,
        nargs="?",
        const="default",
        default="default",
        help='Which size factor to use. "default" computes size factor on the chosen level of preprocessing. "raw" uses size factor computed on raw data as n_counts/median(n_counts). "constant" uses a size factor of 1 for every cells',
    )
    parser.add_argument(
        "--scale_input",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="Weither to scale input the count values",
    )
    parser.add_argument(
        "--logtrans_input",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Weither to log transform count values",
    )
    parser.add_argument(
        "--use_hvg",
        type=int,
        nargs="?",
        const=5000,
        default=None,
        help="Number of hvg to use. If no tag, don't use hvg.",
    )
    # parser.add_argument('--reduce_lr', type = , default = , help ='')
    # parser.add_argument('--early_stop', type = , default = , help ='')
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        default=128,
        help="Training batch size",
    )  # Default identified with hp optimization
    # parser.add_argument('--verbose', type = , default = , help ='')
    # parser.add_argument('--threads', type = , default = , help ='')
    parser.add_argument(
        "--test_split_key",
        type=str,
        default="TRAIN_TEST_split",
        help="key of obs containing the test split",
    )
    parser.add_argument(
        "--test_obs",
        type=str,
        nargs="+",
        default=None,
        help="batches from batch_key to use as test",
    )
    parser.add_argument(
        "--test_index_name",
        type=str,
        nargs="+",
        default=None,
        help="indexes to be used as test. Overwrites test_obs",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="percentage",
        help="Train test split mode to be used by Dataset.train_split",
    )
    parser.add_argument(
        "--pct_split", type=float, nargs="?", default=0.9, help=""
    )
    parser.add_argument(
        "--obs_key", type=str, nargs="?", default="manip", help=""
    )
    parser.add_argument(
        "--n_keep",
        type=int,
        nargs="?",
        default=None,
        help="batches from obs_key to use as train",
    )
    parser.add_argument(
        "--split_strategy", type=str, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--keep_obs", type=str, nargs="+", default=None, help=""
    )
    parser.add_argument(
        "--train_test_random_seed", type=float, nargs="?", default=0, help=""
    )
    parser.add_argument(
        "--obs_subsample", type=str, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--make_fake",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="",
    )
    parser.add_argument(
        "--true_celltype", type=str, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--false_celltype", type=str, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--pct_false", type=float, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--clas_loss_name",
        type=str,
        nargs="?",
        choices=["categorical_crossentropy", "categorical_focal_crossentropy"],
        default="categorical_crossentropy",
        help="Loss of the classification branch",
    )
    parser.add_argument(
        "--balance_classes",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Wether to weight the classification loss by inverse of classes weight",
    )
    parser.add_argument(
        "--dann_loss_name",
        type=str,
        nargs="?",
        choices=["categorical_crossentropy"],
        default="categorical_crossentropy",
        help="Loss of the DANN branch",
    )
    parser.add_argument(
        "--rec_loss_name",
        type=str,
        nargs="?",
        choices=["MSE"],
        default="MSE",
        help="Reconstruction loss of the autoencoder",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        nargs="?",
        default=2e-6,
        help="Weight decay applied by th optimizer",
    )  # Default identified with hp optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        nargs="?",
        default=0.001,
        help="Starting learning rate for training",
    )  # Default identified with hp optimization
    parser.add_argument(
        "--optimizer_type",
        type=str,
        nargs="?",
        choices=["adam", "adamw", "rmsprop"],
        default="adam",
        help="Name of the optimizer to use",
    )
    parser.add_argument(
        "--clas_w",
        type=float,
        nargs="?",
        default=0.1,
        help="Weight of the classification loss",
    )
    parser.add_argument(
        "--dann_w",
        type=float,
        nargs="?",
        default=0.1,
        help="Weight of the DANN loss",
    )
    parser.add_argument(
        "--rec_w",
        type=float,
        nargs="?",
        default=0.8,
        help="Weight of the reconstruction loss",
    )
    parser.add_argument(
        "--warmup_epoch",
        type=float,
        nargs="?",
        default=50,
        help="Number of epoch to warmup DANN",
    )

    parser.add_argument(
        "--dropout",
        type=int,
        nargs="?",
        default=None,
        help="dropout applied to every layers of the model. If specified, overwrites other dropout arguments",
    )
    parser.add_argument(
        "--layer1",
        type=int,
        nargs="?",
        default=None,
        help="size of the first layer for a 2-layers model. If specified, overwrites ae_hidden_size",
    )
    parser.add_argument(
        "--layer2",
        type=int,
        nargs="?",
        default=None,
        help="size of the second layer for a 2-layers model. If specified, overwrites ae_hidden_size",
    )
    parser.add_argument(
        "--bottleneck",
        type=int,
        nargs="?",
        default=None,
        help="size of the bottleneck layer. If specified, overwrites ae_hidden_size",
    )

    parser.add_argument(
        "--ae_hidden_size",
        type=int,
        nargs="+",
        default=[128, 64, 128],
        help="Hidden sizes of the successive ae layers",
    )
    parser.add_argument(
        "--ae_hidden_dropout", type=float, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--ae_activation", type=str, nargs="?", default="relu", help=""
    )
    parser.add_argument(
        "--ae_bottleneck_activation",
        type=str,
        nargs="?",
        default="linear",
        help="activation of the bottleneck layer",
    )
    parser.add_argument(
        "--ae_output_activation",
        type=str,
        nargs="?",
        const="relu",
        default="relu",
        help="",
    )
    parser.add_argument(
        "--ae_init", type=str, nargs="?", default="glorot_uniform", help=""
    )
    parser.add_argument(
        "--ae_batchnorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    parser.add_argument(
        "--ae_l1_enc_coef", type=float, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--ae_l2_enc_coef", type=float, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--class_hidden_size",
        type=int,
        nargs="+",
        default=[64],
        help="Hidden sizes of the successive classification layers",
    )
    parser.add_argument(
        "--class_hidden_dropout", type=float, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--class_batchnorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    parser.add_argument(
        "--class_activation", type=str, nargs="?", default="relu", help=""
    )
    parser.add_argument(
        "--class_output_activation",
        type=str,
        nargs="?",
        default="softmax",
        help="",
    )
    parser.add_argument(
        "--dann_hidden_size", type=int, nargs="?", default=[64], help=""
    )
    parser.add_argument(
        "--dann_hidden_dropout", type=float, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--dann_batchnorm",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    parser.add_argument(
        "--dann_activation", type=str, nargs="?", default="relu", help=""
    )
    parser.add_argument(
        "--dann_output_activation",
        type=str,
        nargs="?",
        default="softmax",
        help="",
    )
    parser.add_argument(
        "--training_scheme",
        type=str,
        nargs="?",
        default="training_scheme_1",
        help="",
    )
    parser.add_argument(
        "--log_neptune",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="",
    )
    parser.add_argument(
        "--hparam_path", type=str, nargs="?", default=None, help=""
    )
    parser.add_argument(
        "--opt_metric",
        type=str,
        nargs="?",
        default="val-balanced_mcc",
        help="The metric used for early stopping as well as optimizes in hp search. Should be formatted as it appears in neptune (split-metricname)",
    )

    # parser.add_argument('--epochs', type=int, nargs='?', default=100, help ='')

    run_file = parser.parse_args()
    print(run_file.class_key, run_file.batch_key)
    experiment = MakeExperiment(
        run_file=run_file, working_dir=run_file.working_dir
    )

    if not run_file.hparam_path:
        hparam_path = JSON_PATH_DEFAULT + "generic_r1.json"
    else:
        hparam_path = run_file.hparam_path

    hparams = load_json(hparam_path)

    # else:
    #     hparams = [{"name": "use_hvg", "type": "range", "bounds": [5000, 10000], "log_scale": False},
    #         {"name": "clas_w", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
    #         {"name": "dann_w", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
    #         {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
    #         {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-4], "log_scale": True},
    #         {"name": "warmup_epoch", "type": "range", "bounds": [1, 50]},
    #         {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
    #         {"name": "bottleneck", "type": "range", "bounds": [32, 64]},
    #         {"name": "layer2", "type": "range", "bounds": [64, 512]},
    #         {"name": "layer1", "type": "range", "bounds": [512, 2048]},

    #     ]

    # workflow.make_experiment(hparams)

    best_parameters, values, experiment, model = optimize(
        parameters=hparams,
        evaluation_function=experiment.train,
        objective_name=run_file.opt_metric,
        minimize=False,
        total_trials=total_trial,
        random_seed=random_seed,
    )

    print(best_parameters)
