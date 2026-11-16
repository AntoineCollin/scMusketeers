import pandas as pd
import logging

from ..arguments.runfile import set_hyperparameters
from . import dataset_tf
from .optimize_model import Workflow

logger = logging.getLogger("Sc-Musketeers")


class MakeExperiment:
    def __init__(self, run_file, working_dir, total_trial, random_seed):
        # super().__init__()
        self.run_file = run_file
        self.working_dir = working_dir
        self.workflow = None
        self.trial_count = 0
        self.total_trial = total_trial
        self.random_seed = random_seed

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
            dataset_tf.process_dataset(self.workflow)
            dataset_tf.split_train_test(self.workflow)
            dataset_tf.split_train_val(self.workflow)
            opt_metric = self.workflow.make_workflow()
            # del self.workflow  # Should not be necessary
            return opt_metric
        else:  # we return the already computed value
            return result.loc[:, f"evaluation/{split}/{metric}"].iloc[0]
