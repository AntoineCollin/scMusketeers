import json
import os

from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                stop_neptune_log)
from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)
from scmusketeers.workflow.benchmark import Workflow
from scmusketeers.transfer.experiment import MakeExperiment

try:
    from ax.service.ax_client import AxClient, ObjectiveProperties
except ImportError:
    print("Import scmusketeers.workflow - AX Platform not installed")
    print("Please consider installing AxPlatform for hyperparameters optimization")
    print("poetry install --with workflow")
    

# JSON_PATH_DEFAULT = '/home/acollin/scMusketeers/experiment_script/hp_ranges/'
JSON_PATH_DEFAULT = "/home/becavin/scMusketeers/experiment_script/hp_ranges/"

TOTAL_TRIAL = 10
RANDOM_SEED = 40


def load_json(json_path):
    with open(json_path, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


def run_workflow(run_file):
    experiment = MakeExperiment(
        run_file=run_file,
        working_dir=run_file.working_dir,
        total_trial=TOTAL_TRIAL,
        random_seed=RANDOM_SEED,
    )

    if not run_file.hparam_path:
        hparam_path = JSON_PATH_DEFAULT + "generic_r1.json"
    else:
        hparam_path = run_file.hparam_path

    hparams = load_json(hparam_path)

    ### Loop API
    best_parameters, values, experiment, model = optimize(
        parameters=hparams,
        evaluation_function=experiment.train,
        objective_name=run_file.opt_metric,
        minimize=False,
        total_trials=experiment.total_trial,
        random_seed=experiment.random_seed,
    )

    ### Service API
    ax_client = AxClient()
    ax_client.create_experiment(
        name="scmusketeers",
        parameters=hparams,
        objectives={"opt_metric": ObjectiveProperties(minimize=False)},
    )
    for i in range(experiment.total_trial):
        parameterization, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=experiment.train(parameterization),
        )

    best_parameters, values = ax_client.get_best_parameters()
    print(best_parameters)
