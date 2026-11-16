import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
# import decoupler as dc
import sys
import ast
import functools
import neptune

from scmusketeers.tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg, balanced_matthews_corrcoef, balanced_f1_score, balanced_cohen_kappa_score

from sklearn.metrics import accuracy_score,balanced_accuracy_score,matthews_corrcoef, f1_score,cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,davies_bouldin_score,adjusted_rand_score,confusion_matrix
f1_score = functools.partial(f1_score, average = 'macro')

from scmusketeers.workflow.dataset import load_dataset
from scmusketeers.tools.utils import ann_subset, check_raw,save_json, load_json, rgb2hex,hex2rgb, check_dir, df_to_dict,dict_to_df
from scmusketeers.tools.plot import get_scanpy_cmap

#working_dir = '/data/analysis/data_becavin/scMusketeers/'
#working_dir = '/workspace/cell/scPermut_Antoine/'
working_dir = '/workspace/cell/scMusketeers/'


dataset_names = {'htap':'htap',
                "ajrccm_by_batch":"ajrccm_by_batch",
                'hlca_par_dataset_harmonized':'hlca_par_dataset_harmonized',
                'hlca_trac_dataset_harmonized':'hlca_trac_dataset_harmonized' ,
                'koenig_2022' : 'celltypist_dataset/koenig_2022/koenig_2022_healthy',
                'tosti_2021' : 'celltypist_dataset/tosti_2021/tosti_2021',
                'yoshida_2021' : 'celltypist_dataset/yoshida_2021/yoshida_2021',
                'tran_2021' : 'celltypist_dataset/tran_2021/tran_2021',
                'dominguez_2022_lymph' : 'celltypist_dataset/dominguez_2022/dominguez_2022_lymph',
                'dominguez_2022_spleen' : 'celltypist_dataset/dominguez_2022/dominguez_2022_spleen',
                'tabula_2022_spleen' : 'celltypist_dataset/tabula_2022/tabula_2022_spleen',
                'litvinukova_2020' : 'celltypist_dataset/litvinukova_2020/litvinukova_2020',
                 'lake_2021': 'celltypist_dataset/lake_2021/lake_2021'
                }

dataset_list = ['htap', 'yoshida_2021', 'hlca_trac_dataset_harmonized',
               'lake_2021', 'dominguez_2022_spleen', 'ajrccm_by_batch',
               'tosti_2021', 'litvinukova_2020','koenig_2022', #'tran_2021', 
               'hlca_par_dataset_harmonized', 'dominguez_2022_lymph',
                'tabula_2022_spleen']


diff_dict = {'homo': ['tosti_2021',  'yoshida_2021', 'htap', 'ajrccm_by_batch'], #'tran_2021',
           'assays': ['dominguez_2022_lymph', 'dominguez_2022_spleen', 'tabula_2022_spleen'],
           'suspension': ['koenig_2022', 'litvinukova_2020', 'lake_2021'],
           'datasets': ['hlca_par_dataset_harmonized','hlca_trac_dataset_harmonized']}

aestetic_data_name = {'tosti_2021': 'Tosti 2021',
                      # 'tran_2021': 'Tran 2021', 
                      'yoshida_2021': 'Yoshida 2021',
                      'htap': 'Mbouamboua 2024', 
                      'ajrccm_by_batch': 'Deprez 2020',
                      'dominguez_2022_lymph': 'Dominguez 2022 - lymph',
                      'dominguez_2022_spleen': 'Dominguez 2022 - spleen',
                      'tabula_2022_spleen': 'Tabula Sapiens - spleen',
                      'koenig_2022': 'Koenig 2022', 
                      'litvinukova_2020': 'Litvinukova 2020', 
                      'lake_2021': 'Lake 2021',
                      'hlca_par_dataset_harmonized': 'HLCA - Parenchyma',
                      'hlca_trac_dataset_harmonized': 'HLCA - Airway'}

aestetic_model_name = {'scPermut' : 'scMusketeers',
                       'scPermut_default' : 'scMusketeers - default',
                      'scanvi' : 'scANVI',
                       'uce' : 'UCE', 
                       'harmony_svm' : 'Harmony',
                      'pca_svm' : 'PCA',
                       'celltypist' : 'Celltypist',
                       'scmap_cells' : 'scmap - cells',
                       'scmap_cluster' : 'scmap - cluster'}

hp_list=['use_hvg',
'batch_size',
'clas_w',
'dann_w',
'rec_w',
'ae_bottleneck_activation',
         'clas_loss_name',
'size_factor',
'weight_decay',
'learning_rate',
'warmup_epoch',
'dropout',
'layer1',
'layer2',
'bottleneck',
        'training_scheme']

test_fold_selection = load_json(working_dir + 'experiment_script/benchmark/hp_test_folds')
test_obs = load_json(working_dir + 'experiment_script/benchmark/hp_test_obs')

def load_run_df(task):
    project = neptune.init_project(
            project="becavin-lab/benchmark",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        mode="read-only",
            )# For checkpoint

    runs_table_df = project.fetch_runs_table(query = f'`parameters/task`:string = "{task}"').to_pandas()
    project.stop()

    f =  lambda x : x.replace('evaluation/', '').replace('parameters/', '').replace('/', '_')
    runs_table_df.columns = np.array(list(map(f, runs_table_df.columns)))
    return runs_table_df


def load_pred(neptune_id, working_dir = None):
    if working_dir :
        save_dir = working_dir + 'experiment_script/results/' + str(neptune_id) + '/'
    else :
        save_dir = './experiment_script/results/' + str(neptune_id) + '/'
    return pd.read_csv(save_dir + f'predictions_full.csv', index_col =0).squeeze()

def load_split(neptune_id, working_dir = None):
    if working_dir :
        save_dir = working_dir + 'experiment_script/results/' + str(neptune_id) + '/'
    else :
        save_dir = './experiment_script/results/' + str(neptune_id) + '/'
    return pd.read_csv(save_dir + f'split_full.csv', index_col =0).squeeze()
    
def nan_to_0(val):
    if np.isnan(val) or pd.isna(val) or type(val) == type(None) :
        return 0.0
    else :
        return val
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
    print(f"Neptune: {neptune_id}")
    #umap = load_umap(neptune_id, working_dir)
    # adata.obsm['proba_pred'] = proba_pred
    #adata.obsm['X_umap'] = umap
    return adata

def plot_umap_proba(adata, celltype, **kwargs):
    adata.obs[celltype] = adata.obsm['proba_pred'][celltype]
    sc.pl.umap(adata, color = celltype, **kwargs)


def plot_size_conf_correlation(adata):
    proba_pred = adata.obsm['proba_pred']
    class_df_dict = {ct : proba_pred.loc[adata.obs['true'] == ct, :] for ct in adata.obs['true'].cat.categories} # The order of the plot is defined here (adata.obs['true_louvain'].cat.categories)
    mean_acc_dict = {ct : df.mean(axis = 0) for ct, df in class_df_dict.items()}

    f, axes = plt.subplots(1,2, figsize = (10,5))
    f.suptitle('correlation between confidence and class size')
    pd.Series({ct : class_df_dict[ct].shape[0] for ct in mean_acc_dict.keys()}).plot.bar(ax = axes[0])
    pd.Series({ct : mean_acc_dict[ct][ct] for ct in mean_acc_dict.keys()}).plot.bar(ax =axes[1])
    
def plot_class_accuracy(adata,layout = True, **kwargs):
    '''
    mode is either bar (average) or box (boxplot)
    '''
    adata = self.latent_spaces[ID]
    workflow = self.workflow_list[ID]
    true_key = f'true_{workflow.class_key}'
    pred_key = f'{workflow.class_key}_pred'
    labels = adata.obs[true_key].cat.categories
    conf_mat = pd.DataFrame(confusion_matrix(adata.obs[true_key], adata.obs[pred_key], labels=labels),index = labels, columns = labels)

    n = math.ceil(np.sqrt(len(labels)))
    f, axes = plt.subplots(n,n, constrained_layout=layout)
    f.suptitle("Accuracy & confusion by celltype")
#     plt.constrained_layout()
    i = 0
    for ct in labels:
        r = i // n
        c = i % n
        ax = axes[r,c]
        df = conf_mat.loc[ct,:]/conf_mat.loc[ct,:].sum()  
        df.plot.bar(ax = ax, figsize = (20,15), ylim = (0,1), **kwargs)
        ax.tick_params(axis='x', labelrotation=90 )
        ax.set_title(ct + f'- {conf_mat.loc[ct,:].sum()} cells')
        i+=1

def plot_confusion_matrix(sub, sub_small, return_mat = False):
    # labels = list(set(np.unique(y_true)).union(set(np.unique(y_pred))))
    y_true = sub.obs['true']
    y_pred = sub.obs['pred']
    y_true_small = sub_small.obs['true']
    y_pred_small = sub_small.obs['pred']
    
    # true_labels = np.unique(y_true)
    # pred_labels = np.unique(y_pred)
    # true_labels_small = np.unique(y_true_small)
    # pred_labels_small = np.unique(y_pred_small)

    true_labels = y_true.value_counts().index[::-1]
    pred_labels = y_pred.value_counts().index[::-1]
    true_labels_small = y_true_small.value_counts().index[::-1]
    pred_labels_small = y_pred_small.value_counts().index[::-1]

    labels = sub.obs['true'].unique()
    cm = confusion_matrix(y_true, y_pred, labels = labels)
    cm_norm = cm / cm.sum(axis = 1, keepdims=True)
    cm_to_plot=pd.DataFrame(cm_norm, index = labels, columns=labels)
    cm_to_save=pd.DataFrame(cm, index = labels, columns=labels)
    
    cm_to_plot = cm_to_plot.fillna(value=0)
    cm_to_save = cm_to_save.fillna(value=0)
    
    size = len(labels)
    true_order = list(true_labels_small) + [i for i in cm_to_plot.columns if i not in true_labels_small]
    cm_to_plot = cm_to_plot.loc[true_labels_small,true_order]
    if return_mat:
        cm_to_save = cm_to_save.loc[true_labels_small,true_order]
        return cm_to_save, cm_to_plot
    f, ax = plt.subplots(figsize = (cm_to_plot.shape[1]/1.5, cm_to_plot.shape[0]/1.5))  #figsize=(size/1.5,size/1.5))
    sns.heatmap(cm_to_plot, annot=True,fmt='.2f', vmin = 0, vmax = 1)
    show_mask = np.asarray(cm_to_plot>0.01)
    for text, show_annot in zip(ax.texts, (element for row in show_mask for element in row)):
        text.set_visible(show_annot)
    return f,ax

def umap_subset(adata, obs_key, subset, **kwargs):
    '''
    Plot individual subset, the rest is grey
    '''
    if type(subset) == str:
        subset = [subset]
    ax=sc.pl.umap(adata,color=obs_key, groups=['subset'], show=False, **kwargs)


    legend_texts=ax.get_legend().get_texts()

    for legend_text in legend_texts:
        if legend_text.get_text()=="NA":
            legend_text.set_text('other cell types')
    return ax

def load_best_t1(runs_table_df,dataset_name, split, met):
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
            print(model)
            ad_list[model] = ad
    return ad_list

def load_best_model_t1(runs_table_df,dataset_name, best_model, split, met):
    task_1 = runs_table_df.query("task == 'task_1'").query(f"dataset_name == '{dataset_name}'").query(f"test_fold_nb == {test_fold_selection[dataset_name]}").query('deprecated_status == False').query('use_hvg == 3000')
    task_1 = task_1.loc[~((task_1['model'] == 'scPermut_default') & (task_1['training_scheme'] != 'training_scheme_8')),:]
    ad_list = {}
    for model in np.unique(task_1['model']):
        if model == best_model:
        
            # if model == 'scPermut':
            #     sub = task_1.query(f'model == "{model}"').query('training_scheme == "training_scheme_8"')
            # else:
            sub = task_1.query(f'model == "{model}"')
            if not sub.loc[sub[f'{split}_{met}'] == sub[f'{split}_{met}'].max(),'sys_id'].empty :
                best_id = sub.loc[sub[f'{split}_{met}'] == sub[f'{split}_{met}'].max(),'sys_id'].values[0]
                ad = load_expe(best_id,working_dir)
                print(model)
                ad_list[model] = ad
    return ad_list

def load_fold_t1(runs_table_df,dataset_name, fold, split, met):
    task_1 = runs_table_df.query("task == 'task_1'").query(f"dataset_name == '{dataset_name}'").query(f"test_fold_nb == {test_fold_selection[dataset_name]}").query('deprecated_status == False').query('use_hvg == 3000')
    task_1 = task_1.loc[~((task_1['model'] == 'scPermut_default') & (task_1['training_scheme'] != 'training_scheme_8')),:]
    task_1 = task_1.query(f"val_fold_nb == {fold}")
    ad_list = {}
    for model in np.unique(task_1['model']):
        # if model == 'scPermut':
        #     sub = task_1.query(f'model == "{model}"').query('training_scheme == "training_scheme_8"')
        # else:
        sub = task_1.query(f'model == "{model}"')
        if not sub.loc[sub[f'{split}_{met}'] == sub[f'{split}_{met}'].max(),'sys_id'].empty :
            print(sub.value[f'{split}_{met}'])
            id = sub.loc[sub[f'{split}_{met}'] == sub[f'{split}_{met}'].max(),'sys_id'].values[0]
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

#split = 'test'
#met = 'balanced_acc'

def uniform_colors(ad):
    scanpy_102 = get_scanpy_cmap()
    ad.obs['true'] = pd.Categorical(ad.obs['true'])
    ad.obs['pred'] = pd.Categorical(ad.obs['pred'])
    # Defining colors
    n_col = len(np.unique(ad.obs['true']))
    colors = scanpy_102[:n_col]
    # colors = sns.color_palette(n_colors= n_col)
    color_dict = {k: v for k, v in zip(ad.obs['true'].cat.categories, colors)}
    color_dict['Unassigned'] = scanpy_102[-1]
    color_dict['Unassigne'] = scanpy_102[-2]
    color_dict['Unassign'] = scanpy_102[-3]
    color_dict['Unassig'] = scanpy_102[-4]
    ad.uns['true_colors'] = [color_dict[ct] for ct in ad.obs['true'].cat.categories]
    ad.uns['pred_colors'] = [color_dict[ct] for ct in ad.obs['pred'].cat.categories]