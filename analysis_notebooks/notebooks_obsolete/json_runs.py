import pandas as pd
import numpy as np
# import decoupler as dc
import neptune
import sys

working_dir = '/data/analysis/data_becavin/scMusketeers-data'
fig_dir = '/data/analysis/data_becavin/scMusketeers/analysis_notebooks/figures_review/'

import os
os.chdir(working_dir)

def load_run_df(neptune_name):
    if neptune_name=="benchmark":
        project = neptune.init_project(
                project="becavin-lab/benchmark",  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
            mode="read-only",
                )# For checkpoint
    elif neptune_name=="scmusk-hp":
        project = neptune.init_project(
                project="becavin-lab/scmusk-hp",api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
            mode="read-only",
                )# For checkpoint
    runs_table_df = project.fetch_runs_table().to_pandas()
    project.stop()

    f =  lambda x : x.replace('evaluation/', '').replace('parameters/', '').replace('/', '_')
    runs_table_df.columns = np.array(list(map(f, runs_table_df.columns)))
    return runs_table_df
    
runs_table_df_old = load_run_df("benchmark")
runs_table_df = load_run_df("scmusk-hp")

import os
import pandas as pd
fig_dir = '/data/analysis/data_becavin/scMusketeers/analysis_notebooks/figures_review/'
hp_old_path = os.path.join(fig_dir,"hp_search_old.pkl")
hp_path = os.path.join(fig_dir,"hp_search.pkl")
#if os.path.exists(task1_path):
#    print("tet")
# runs_table_df = pd.read_pickle(task1_path)    
#else:
# runs_table_df = load_run_df("task_1")
runs_table_df_old.to_pickle(hp_old_path)
runs_table_df.to_pickle(hp_path)