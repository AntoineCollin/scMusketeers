import neptune
import pandas as pd

from ..arguments.neptune_log import (
    add_custom_log,
    start_neptune_log,
    stop_neptune_log,
)
from ..arguments.runfile import set_hyperparameters
from ..workflow import dataset
from .optimize_hp import Workflow


class MakeExperiment:
    def __init__(self, run_file, working_dir, total_trial, random_seed):
        # super().__init__()
        self.run_file = run_file
        self.working_dir = working_dir
        self.workflow = None
        self.trial_count = 0
        self.total_trial = total_trial
        self.random_seed = random_seed
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
        # import tensorflow as tf

        self.trial_count += 1
        # print('params')
        # print(params)
        checkpoint = {"parameters/" + k: i for k, i in params.items()}
        checkpoint["parameters/dataset_name"] = self.run_file.dataset_name
        checkpoint["parameters/opt_metric"] = self.run_file.opt_metric

        """ for column in self.runs_table_df.columns:
            if "parameters/" in column:
                print(column)
        for k, v  in params.items(): 
            print(k, v) """
        # checkpoint = {'parameters/dataset_name': self.run_file.dataset_name,
        #               'parameters/total_trial': total_trial, 'parameters/trial_count': self.trial_count,
        #               'parameters/opt_metric': self.opt_metric, 'parameters/hp_random_seed': random_seed}
        # result = self.runs_table_df[self.runs_table_df[list(checkpoint.keys())].eq(list(checkpoint.values())).all(axis=1)]
        result = pd.DataFrame()
        # print(result)
        split, metric = self.run_file.opt_metric.split("-")
        if result.empty or pd.isna(
            result.loc[:, f"evaluation/{split}/{metric}"].iloc[0]
        ):  # we run the trial
            self.workflow = Workflow(
                run_file=self.run_file, working_dir=self.working_dir
            )
            set_hyperparameters(self.workflow, params)
            start_neptune_log(self.workflow)
            dataset.process_dataset(self.workflow)
            dataset.split_train_test(self.workflow)
            dataset.split_train_val(self.workflow)
            opt_metric = (
                self.workflow.make_workflow()
            )  # This starts the logging
            add_custom_log(self.workflow, "task", "hp_optim")
            add_custom_log(self.workflow, "total_trial", self.total_trial)
            add_custom_log(self.workflow, "hp_random_seed", self.random_seed)
            add_custom_log(self.workflow, "trial_count", self.trial_count)
            add_custom_log(
                self.workflow, "opt_metric", self.run_file.opt_metric
            )
            stop_neptune_log(self.workflow)
            # del self.workflow  # Should not be necessary
            return opt_metric
        else:  # we return the already computed value
            return result.loc[:, f"evaluation/{split}/{metric}"].iloc[0]
