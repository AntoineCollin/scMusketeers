import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import decoupler as dc
import sys
import ast
import functools
import neptune
working_dir = '/home/acollin/scPermut/'

sys.path.append(working_dir)

from scmusketeers.tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg, balanced_matthews_corrcoef, balanced_f1_score, balanced_cohen_kappa_score

from sklearn.metrics import accuracy_score,balanced_accuracy_score,matthews_corrcoef, f1_score,cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,davies_bouldin_score,adjusted_rand_score,confusion_matrix
f1_score = functools.partial(f1_score, average = 'macro')

from scmusketeers.workflow.dataset import load_dataset
from scmusketeers.tools.utils import ann_subset, check_raw,save_json, load_json, rgb2hex,hex2rgb
from tqdm import trange, tqdm
import warnings


def load_run_df():
    project = neptune.init_project(
            project="becavin-lab/benchmark",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        mode="read-only",
            )# For checkpoint

    runs_table_df = project.fetch_runs_table().to_pandas()
    project.stop()

    f =  lambda x : x.replace('evaluation/', '').replace('parameters/', '').replace('/', '_')
    runs_table_df.columns = np.array(list(map(f, runs_table_df.columns)))
    return runs_table_df

def result_dir(neptune_id, working_dir = None):
    if working_dir :
        save_dir = working_dir + 'experiment_script/results/' + str(neptune_id) + '/'
    else :
        save_dir = './experiment_script/results/' + str(neptune_id) + '/'
    return save_dir
    
def load_pred(neptune_id, working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(save_dir + f'predictions_full.csv', index_col =0).squeeze()
    
pred_metrics_list = {'acc' : accuracy_score, 
                    'mcc' : matthews_corrcoef,
                    'f1_score': f1_score,
                    'KPA' : cohen_kappa_score,
                    'ARI': adjusted_rand_score,
                    'NMI': normalized_mutual_info_score,
                    'AMI':adjusted_mutual_info_score}

pred_metrics_list_balanced = {'balanced_acc' : balanced_accuracy_score, 
                    'balanced_mcc' : balanced_matthews_corrcoef,
                    'balanced_f1_score': balanced_f1_score,
                            'balanced_KPA' : balanced_cohen_kappa_score,
                            }

clustering_metrics_list = {#'clisi' : lisi_avg, 
                            'db_score' : davies_bouldin_score
                            }

batch_metrics_list = {'batch_mixing_entropy': batch_entropy_mixing_score}


def nan_to_0(val):
    if np.isnan(val) or pd.isna(val) or type(val) == type(None) :
        return 0.0
    else :
        return val

runs_table_df = load_run_df()
# runs_table_df = runs_table_df.query('model != "scPermut"').query('task == "task_1"')
runs_table_df = runs_table_df.query('model != "scPermut"').query('task == "task_2"')
# runs_table_df = runs_table_df.query('model != "scPermut"').query('task == "task_3"')

for ID, dataset_name, class_key in zip(runs_table_df['sys_id'],runs_table_df['dataset_name'],runs_table_df['class_key']):
    if not pd.isna(dataset_name) and ID != 'BEN-1':
        run = neptune.init_run(project="becavin-lab/benchmark",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
                 with_id= ID)
        
        y_full = load_pred(ID,working_dir=working_dir)
        y_true_full = y_full['true']
        split_full = y_full['split']
        y_pred_full = y_full['pred']
        print(ID,dataset_name,class_key)
        
        ct_prop = y_true_full.value_counts() / y_true_full.value_counts().sum()

        sizes = {'xxsmall' : list(ct_prop[ct_prop < 0.001].index), 
                 'small': list(ct_prop[(ct_prop >= 0.001) & (ct_prop < 0.01)].index) ,
                 'medium': list(ct_prop[(ct_prop >= 0.01) & (ct_prop < 0.1)].index),
                 'large': list(ct_prop[ct_prop >= 0.1].index)}
        for group in ['train', 'test', 'val', 'full']:
            # print(group)
            if group != 'full':
                y_true = y_true_full.loc[split_full == group]
                y_pred = y_pred_full.loc[split_full == group]
            else :
                y_true = y_true_full
                y_pred = y_pred_full
            for s in sizes : 
                idx_s = np.isin(y_true, sizes[s]) # Boolean array, no issue to index y_pred
                y_true_sub = y_true[idx_s]
                y_pred_sub = y_pred[idx_s]
                # print(s)
                for metric in pred_metrics_list: 
                    # print(metric)
                    run[f"evaluation/{group}/{s}/{metric}"] = nan_to_0(pred_metrics_list[metric](y_true_sub, y_pred_sub))
                    # print(nan_to_0(pred_metrics_list[metric](y_true_sub, y_pred_sub)))
                
                for metric in pred_metrics_list_balanced:
                    # print(metric)
                    # print(nan_to_0(pred_metrics_list_balanced[metric](y_true_sub, y_pred_sub)))
                    run[f"evaluation/{group}/{s}/{metric}"] = nan_to_0(pred_metrics_list_balanced[metric](y_true_sub, y_pred_sub))
        run.stop()




