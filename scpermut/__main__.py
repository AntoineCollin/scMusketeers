import json
from scpermut.workflow.runfile import get_runfile
from scpermut.workflow.runfile import PROCESS_TYPE
from scpermut.workflow.optimize_hp import Workflow
from scpermut.workflow.experiment import MakeExperiment
from scpermut.workflow.neptune_log import start_neptune_log, stop_neptune_log
from scpermut.workflow.dataset import process_dataset, split_train_test, split_train_val


from ax.service.ax_client import AxClient, ObjectiveProperties

# JSON_PATH_DEFAULT = '/home/acollin/scPermut/experiment_script/hp_ranges/'
JSON_PATH_DEFAULT = '/home/becavin/scPermut/experiment_script/hp_ranges/'

TOTAL_TRIAL = 10
RANDOM_SEED = 40

def load_json(json_path):
    with open(json_path, 'r') as fichier_json:
        dico = json.load(fichier_json)
    return dico
    

if __name__ == '__main__':
    
    run_file = get_runfile()
    if run_file.process == PROCESS_TYPE[0]:
        # Single training
        print(run_file.dataset_name, run_file.class_key, run_file.batch_key)
        workflow = Workflow(run_file=run_file, working_dir=run_file.working_dir)
        start_neptune_log(workflow)
        process_dataset(workflow)
        split_train_test(workflow)
        split_train_val(workflow)
        mcc = workflow.make_workflow()
        stop_neptune_log(workflow)
    
    elif run_file.process == PROCESS_TYPE[1]:
        # Hyperparameter optimization
        experiment = MakeExperiment(run_file=run_file, working_dir=run_file.working_dir,
                                    total_trial=TOTAL_TRIAL, random_seed=RANDOM_SEED)

        if not run_file.hparam_path:
            hparam_path = JSON_PATH_DEFAULT + 'generic_r1.json'
        else:
            hparam_path = run_file.hparam_path

        hparams = load_json(hparam_path)

        ### Loop API
        """ best_parameters, values, experiment, model = optimize(
            parameters=hparams,
            evaluation_function=experiment.train,
            objective_name=run_file.opt_metric,
            minimize=False,
            total_trials=experiment.total_trial,
            random_seed=experiment.random_seed,

        ) """

        ### Service API
        ax_client = AxClient()
        ax_client.create_experiment(
            name = "scpermut",
            parameters=hparams,
            objectives={"opt_metric": ObjectiveProperties(minimize=False)},

        )
        for i in range(experiment.total_trial):
            parameterization, trial_index = ax_client.get_next_trial()
            # Local evaluation here can be replaced with deployment to external system.
            ax_client.complete_trial(trial_index=trial_index, raw_data=experiment.train(parameterization))

        
        best_parameters, values = ax_client.get_best_parameters()
        print(best_parameters)
    else:
        # No process
        print("Process not regnoozied")
        
    
    