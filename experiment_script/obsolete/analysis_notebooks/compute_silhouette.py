import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import decoupler as dc
import sys
import neptune
working_dir = '/home/acollin/scPermut/'

sys.path.append(working_dir)

from sklearn.metrics import silhouette_samples

from scmusketeers.tools.utils import ann_subset, check_raw,save_json, load_json, rgb2hex,hex2rgb, check_dir
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
    
def result_dir(neptune_id, working_dir = None):
    if working_dir :
        save_dir = working_dir + 'experiment_script/results/' + str(neptune_id) + '/'
    else :
        save_dir = './experiment_script/results/' + str(neptune_id) + '/'
    return save_dir
    
def load_confusion_matrix(neptune_id,train_split= 'val', working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(save_dir + f'confusion_matrix_{train_split}.csv', index_col =0)

def load_pred(neptune_id, working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(save_dir + f'predictions_full.csv', index_col =0).squeeze()

def load_proba_pred(neptune_id, working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(save_dir + f'y_pred_proba_full.csv', index_col =0).squeeze()

def load_split(neptune_id, working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(save_dir + f'split_full.csv', index_col =0).squeeze()
    
def load_latent_space(neptune_id, working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return np.load(save_dir + f'latent_space_full.npy')

def load_umap(neptune_id, working_dir = None):
    save_dir = result_dir(neptune_id, working_dir)
    return np.load(save_dir + f'umap_full.npy')

def load_expe(neptune_id, working_dir):
    save_dir = result_dir(neptune_id, working_dir)
    X = load_latent_space(neptune_id, working_dir)
    pred = load_pred(neptune_id, working_dir)
    adata = sc.AnnData(X = X, obs = pred)
    # proba_pred = load_proba_pred(neptune_id, working_dir)
    umap = load_umap(neptune_id, working_dir)
    # adata.obsm['proba_pred'] = proba_pred
    adata.obsm['X_umap'] = umap
    return adata

test_fold_selection = load_json(working_dir + 'experiment_script/benchmark/hp_test_folds')
test_obs = load_json(working_dir + 'experiment_script/benchmark/hp_test_obs')

split = 'test'
met = 'balanced_acc'


def load_best_t1(runs_table_df,dataset_name):
    task_1 = runs_table_df.query("task == 'task_1'").query(f"dataset_name == '{dataset_name}'").query(f"test_fold_nb == {test_fold_selection[dataset_name]}").query('deprecated_status == False').query('use_hvg == 3000')
    task_1 = task_1.loc[~((task_1['model'] == 'scPermut_default') & (task_1['training_scheme'] != 'training_scheme_8')),:]
    ad_list = {}
    for model in np.unique(task_1['model']):
        # if model == 'scPermut':
        #     sub = task_1.query(f'model == "{model}"').query('training_scheme == "training_scheme_8"')
        # else:
        sub = task_1.query(f'model == "{model}"')
        if not sub.loc[sub[f'{split}_{met}'] == sub[f'{split}_{met}'].max(),'sys_id'].empty :
            best_id = sub.loc[sub[f'{split}_{met}'] == sub[f'{split}_{met}'].max(),'sys_id'].values[0]
            ad = load_expe(best_id,working_dir)
            ad_list[model] = ad
    return ad_list

def get_sizes(ad_list):
    ct_prop = list(ad_list.values())[0].obs['true'].value_counts()/ list(ad_list.values())[0].n_obs
    sizes = {'xxsmall' : list(ct_prop[ct_prop < 0.001].index), 
            'small': list(ct_prop[(ct_prop >= 0.001) & (ct_prop < 0.01)].index),
            'medium': list(ct_prop[(ct_prop >= 0.01) & (ct_prop < 0.1)].index),
            'large': list(ct_prop[ct_prop >= 0.1].index)}
    return sizes

dataset_list = ['lake_2021', 'htap', 'yoshida_2021', 'hlca_trac_dataset_harmonized',
               'dominguez_2022_spleen', 'ajrccm_by_batch',
               'tosti_2021', 'tran_2021', 'litvinukova_2020','koenig_2022',
               'hlca_par_dataset_harmonized', 'dominguez_2022_lymph',
                'tabula_2022_spleen']

runs_table_df = load_run_df()

for dataset_name in dataset_list :
    adata_list = load_best_t1(runs_table_df, dataset_name)
    sizes = get_sizes(adata_list)

    for model in adata_list.keys() :
        ad = adata_list[model]
        X = ad.X
        labels_true = ad.obs['true']
        labels_pred = ad.obs['pred']
        sil_true = silhouette_samples(X, labels_true)
        sil_pred = silhouette_samples(X, labels_pred)
        ad.obs['sil_true'] = sil_true
        ad.obs['sil_pred'] = sil_pred
        save_dir = f'/home/acollin/scPermut/analysis_notebooks/results/silhouettes/{dataset_name}/'
        check_dir(save_dir)
        ad.write(f'{save_dir}{dataset_name}_{model}.h5ad')

