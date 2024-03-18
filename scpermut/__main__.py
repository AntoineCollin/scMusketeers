from workflow.optimize_hp import *
from scpermut.workflow.runfile import get_runfile, set_hyperparameters
from scpermut.workflow.runfile import PROCESS_TYPE

from scpermut.workflow.neptune_log import start_neptune_log, stop_neptune_log
from scpermut.workflow.dataset import process_dataset, split_train_test, split_train_val

# JSON_PATH_DEFAULT = '/home/acollin/scPermut/experiment_script/hp_ranges/'
JSON_PATH_DEFAULT = '/home/becavin/scPermut/experiment_script/hp_ranges/'

def load_json(json_path):
    with open(json_path, 'r') as fichier_json:
        dico = json.load(fichier_json)
    return dico
    
total_trial = 50
random_seed=40

class MakeExperiment:
    def __init__(self, run_file, working_dir):
        # super().__init__()
        self.run_file = run_file
        self.working_dir = working_dir
        self.workflow = None
        self.trial_count = 0
        # project = neptune.init_project(
        #     project="becavin-lab/benchmark",
        # api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        # mode="read-only",
        #     )
        project = neptune.init_project(
            project="becavin-lab/sc-permut-packaging",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
        mode="read-only",
            )
        # For checkpoint
        self.runs_table_df = project.fetch_runs_table().to_pandas()
        
        project.stop()
        

    def train(self, params):
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        #import tensorflow as tf

        self.trial_count += 1
        # print('params')
        # print(params)
        checkpoint = {'parameters/' + k: i for k,i in params.items()}
        checkpoint['parameters/dataset_name'] = self.run_file.dataset_name
        checkpoint['parameters/opt_metric'] = self.run_file.opt_metric
        checkpoint['parameters/ae_bottleneck_activation'] = self.run_file.ae_bottleneck_activation
        checkpoint['parameters/size_factor'] = self.run_file.size_factor
        # checkpoint = {'parameters/dataset_name': self.run_file.dataset_name,
        #               'parameters/total_trial': total_trial, 'parameters/trial_count': self.trial_count, 
        #               'parameters/opt_metric': self.opt_metric, 'parameters/hp_random_seed': random_seed}
        result = self.runs_table_df[self.runs_table_df[list(checkpoint.keys())].eq(list(checkpoint.values())).all(axis=1)]
        # print(result)
        split, metric = self.run_file.opt_metric.split('-')
        if result.empty or pd.isna(result.loc[:,f'evaluation/{split}/{metric}'].iloc[0]): # we run the trial
            self.workflow = Workflow(run_file=self.run_file, working_dir=self.working_dir)
            set_hyperparameters(self.workflow, params)
            start_neptune_log(self.workflow)
            process_dataset(self.workflow)
            split_train_test(self.workflow)
            split_train_val(self.workflow)
            opt_metric = self.workflow.make_experiment() # This starts the logging
            self.workflow.add_custom_log('task', 'hp_optim')
            self.workflow.add_custom_log('total_trial', total_trial)
            self.workflow.add_custom_log('hp_random_seed', random_seed)
            self.workflow.add_custom_log('trial_count', self.trial_count)
            self.workflow.add_custom_log('opt_metric', self.run_file.opt_metric)
            stop_neptune_log(workflow)
            # del self.workflow  # Should not be necessary
            return opt_metric
        else: # we return the already computed value
            return result.loc[:,f'evaluation/{split}/{metric}'].iloc[0]



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
        mcc = workflow.make_experiment()
        stop_neptune_log(workflow)
    
    elif run_file.process == PROCESS_TYPE[1]:
        # Hyperparameter optimization
        experiment = MakeExperiment(run_file=run_file, working_dir=run_file.working_dir)

        if not run_file.hparam_path:
            hparam_path = JSON_PATH_DEFAULT + 'generic_r1.json'
        else:
            hparam_path = run_file.hparam_path

        hparams = load_json(hparam_path)

        best_parameters, values, experiment, model = optimize(
            parameters=hparams,
            evaluation_function=experiment.train,
            objective_name=run_file.opt_metric,
            minimize=False,
            total_trials=total_trial,
            random_seed=random_seed,

        )

        print(best_parameters)
    else:
        # No process
        print("Process not regnoozied")
        
    
    