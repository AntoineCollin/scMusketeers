import json
import os
import pickle

from ax.service.ax_client import AxClient, ObjectiveProperties

from scmusketeers.arguments.neptune_log import (
    start_neptune_log,
    stop_neptune_log,
)
from scmusketeers.arguments.runfile import (
    PROCESS_TYPE,
    create_argparser,
    get_runfile,
)
from scmusketeers.transfer.experiment import MakeExperiment
from scmusketeers.transfer.optimize_model import Workflow

# JSON_PATH_DEFAULT = '/home/acollin/scMusketeers/experiment_script/hp_ranges/'
JSON_PATH_DEFAULT = "/home/becavin/scMusketeers/experiment_script/hp_ranges/"

TOTAL_TRIAL = 10
RANDOM_SEED = 40


def load_json(json_path):
    with open(json_path, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


def run_sc_musketeers():

    # Get all arguments
    run_file = get_runfile()

    """ # Save runfile for running in python mode
    with open('tutorial/runfile_tuto_2.pkl', 'wb') as outp:
        pickle.dump(run_file, outp, pickle.HIGHEST_PROTOCOL)
    
    # Load run_file for python mode
    with open('tutorial/runfile_tuto_2.pkl', 'rb') as inp:
        run_file = pickle.load(inp)
    run_file.train_scheme = "training_scheme_13" """

    if run_file.process == PROCESS_TYPE[0]:
        # Transfer data
        print(run_file.ref_path, run_file.class_key, run_file.batch_key)
        workflow = Workflow(run_file=run_file)
        start_neptune_log(workflow)
        workflow.process_dataset()
        workflow.train_val_split()
        adata_pred, model, history, X_scCER, query_pred = (
            workflow.make_experiment()
        )
        stop_neptune_log(workflow)
        print(query_pred)
        adata_pred_path = os.path.join(
            run_file.out_dir, f"{run_file.out_name}.h5ad"
        )
        print((f"Save adata_pred to {adata_pred_path}"))
        adata_pred.write_h5ad(adata_pred_path)

    elif run_file.process == PROCESS_TYPE[1]:
        # Hyperparameter optimization
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
            name="scpermut",
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
    else:
        # No process
        print("Process not recognized")


if __name__ == "__main__":

    run_sc_musketeers()
