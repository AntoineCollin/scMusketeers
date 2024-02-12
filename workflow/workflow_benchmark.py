import sys
import os
WD_PATH = '/home/acollin/dca_permuted_workflow/workflow'
sys.path.append(WD_PATH)
try :
    from .dataset import Dataset, load_dataset
    from .utils import str2bool
    from .clust_compute import batch_entropy_mixing_score,lisi_avg
    from .benchmark_models import pca_svm, harmony_svm, scanvi

except ImportError:
    from dataset import Dataset, load_dataset
    from utils import str2bool
    from clust_compute import batch_entropy_mixing_score,lisi_avg
    from benchmark_models import pca_svm, harmony_svm, scanvi
# from dca.utils import str2bool,tuple_to_scalar
import argparse
import functools
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef, f1_score,cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,davies_bouldin_score,adjusted_rand_score
f1_score = functools.partial(f1_score, average = 'weighted')

import pandas as pd
import scanpy as sc
import numpy as np
import os
import sys
import neptune
# from numba import cuda
from neptune.utils import stringify_unsupported
import time

def train_dummy(X_list, y_list, batch_list, train_plit, **kwargs):
    latent_list = {k:X[:,:2] for k,X in X_list.items()}
    print(kwargs)
    return latent_list,y_list




class Workflow:
    def __init__(self, run_file, working_dir):
        '''
        run_file : a dictionary outputed by the function load_runfile
        '''
        self.run_file = run_file
        self.working_dir = working_dir
        self.data_dir = working_dir + '/data'
        # dataset identifiers
        self.dataset_name = self.run_file.dataset_name
        self.class_key = self.run_file.class_key
        self.batch_key = self.run_file.batch_key
        # normalization parameters
        self.filter_min_counts = self.run_file.filter_min_counts # TODO :remove, we always want to do that
        self.normalize_size_factors = self.run_file.normalize_size_factors
        self.scale_input = self.run_file.scale_input
        self.logtrans_input = self.run_file.logtrans_input
        self.use_hvg = self.run_file.use_hvg

        # train test split # TODO : Simplify this, or at first only use the case where data is split according to batch
        self.test_split_key = self.run_file.test_split_key
        self.mode = self.run_file.mode
        self.pct_split = self.run_file.pct_split
        self.obs_key = self.run_file.obs_key
        self.n_keep = self.run_file.n_keep
        self.split_strategy = self.run_file.split_strategy
        self.keep_obs = self.run_file.keep_obs
        self.train_test_random_seed = self.run_file.train_test_random_seed
        self.obs_subsample = self.run_file.obs_subsample
        
        self.unlabeled_category = 'UNK' # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves

        self.models_fn = {'pca_svm':pca_svm, 'harmony_svm':harmony_svm, 'scanvi':scanvi}#

        self.training_kwds = {}
        self.network_kwds = {}

        self.adata_list = {}
        self.X_list = {}
        self.y_list = {}
        self.batch_list = {}

        self.y_pred_list = {}
        self.latent_space_list = {}
        
        self.pred_metrics_list = {'balanced_acc' : balanced_accuracy_score, 
                    'mcc' : matthews_corrcoef,
                    'f1_score': f1_score,
                    'KPA' : cohen_kappa_score,
                    'ARI': adjusted_rand_score,
                    'NMI': normalized_mutual_info_score,
                    'AMI':adjusted_mutual_info_score}

        self.clustering_metrics_list = {'clisi' : lisi_avg, 
                                    'db_score' : davies_bouldin_score}

        self.batch_metrics_list = {'batch_mixing_entropy': batch_entropy_mixing_score,
                            'ilisi': lisi_avg}
        self.metrics = []

        self.log_neptune = self.run_file.log_neptune
        self.run = None

    def process_dataset(self):
        # Loading dataset
        adata = load_dataset(dataset_dir = self.data_dir,
                               dataset_name = self.dataset_name)
        
        self.dataset = Dataset(adata = adata,
                                class_key = self.class_key,
                                batch_key= self.batch_key,
                                filter_min_counts = self.filter_min_counts,
                                normalize_size_factors = self.normalize_size_factors,
                                scale_input = self.scale_input,
                                logtrans_input = self.logtrans_input,
                                use_hvg = self.use_hvg,
                                test_split_key = self.test_split_key,
                                unlabeled_category = self.unlabeled_category)
        
        # Processing dataset. Splitting train/test. 
        self.dataset.normalize()

    def split_train_test_val(self):
        self.dataset.train_split(mode = self.mode,
                            pct_split = self.pct_split,
                            obs_key = self.obs_key,
                            n_keep = self.n_keep,
                            keep_obs = self.keep_obs,
                            split_strategy = self.split_strategy,
                            obs_subsample = self.obs_subsample,
                            train_test_random_seed = self.train_test_random_seed)
       
        print('dataset has been preprocessed')
        self.dataset.create_inputs()


        self.adata_list = {'full': self.dataset.adata,
                      'train': self.dataset.adata_train,
                      'val': self.dataset.adata_val,
                      'test': self.dataset.adata_test}

        self.X_list = {'full': self.dataset.X,
                'train': self.dataset.X_train,
                'val': self.dataset.X_val,
                'test': self.dataset.X_test}

        self.y_list = {'full': self.dataset.y,
                      'train': self.dataset.y_train,
                      'val': self.dataset.y_val,
                      'test': self.dataset.y_test}

        self.batch_list = {'full': self.dataset.batch,
                      'train': self.dataset.batch_train,
                      'val': self.dataset.batch_val,
                      'test': self.dataset.batch_test}
        
    def train_model(self, model, **kwds):
        start_time = time.time()
        if self.log_neptune :
            self.run = neptune.init_run(
                    project="becavin-lab/sc-permut",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
)
            for par,val in self.run_file.__dict__.items():
                self.run[f"parameters/{par}"] = stringify_unsupported(getattr(self, par)) # getattr(self, par) in case the parameter changed somehow

        self.latent_space_list, self.y_pred_list,= self.models_fn[model](self.X_list, self.y_list, self.batch_list,self.dataset.adata.obs['train_split'], **kwds) # include the self.run object
        stop_time = time.time()
        if self.log_neptune :
            self.run['evaluation/training_time'] = stop_time - start_time

    def compute_metrics(self):
        if self.log_neptune:
            neptune_run_id = self.run['sys/id'].fetch()
            for group in ['full', 'train', 'val', 'test']:
                if group in ['train', 'val', 'test']:
                    for metric in self.pred_metrics_list: # only classification metrics ATM
                            self.run[f"evaluation/{group}/{metric}"] = self.pred_metrics_list[metric](self.adata_list[group].obs[f'true_{self.class_key}'], self.y_pred_list[group])
                    for metric in self.clustering_metrics_list: # only classification metrics ATM
                        self.run[f"evaluation/{group}/{metric}"] = self.clustering_metrics_list[metric](self.latent_space_list[group], self.y_pred_list[group])
                    if len(np.unique(self.batch_list[group])) >= 2: # If there are more than 2 batches in this group
                        for metric in self.batch_metrics_list:
                            self.run[f'evaluation/{group}/{metric}'] = self.batch_metrics_list[metric](self.latent_space_list[group], self.batch_list[group])
                if group == 'full':
                    if len(np.unique(self.batch_list[group])) >= 2: # If there are more than 2 batches in this group
                        for metric in self.batch_metrics_list:
                            self.run[f'evaluation/{group}/{metric}'] = self.batch_metrics_list[metric](self.latent_space_list[group], self.batch_list[group])
                    save_dir = self.working_dir + 'experiment_script/results/' + str(neptune_run_id) + '/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                        
                    # Saving latent space and predictions
                    y_pred = pd.DataFrame(self.y_pred_list[group], index = self.adata_list[group].obs_names)
                    np.save(save_dir + f'latent_space_{group}.npy', self.latent_space_list[group])
                    y_pred.to_csv(save_dir + f'predictions_{group}.csv')
                    self.run[f'evaluation/{group}/latent_space'].track_files(save_dir + f'latent_space_{group}.npy')
                    self.run[f'evaluation/{group}/predictions'].track_files(save_dir + f'predictions_{group}.csv')

                    # Saving umap representation
                    pred_adata = sc.AnnData(X = self.adata_list[group].X, obs = self.adata_list[group].obs, var = self.adata_list[group].var)
                    pred_adata.obs[f'{self.class_key}_pred'] = y_pred
                    pred_adata.obsm['latent_space'] = self.latent_space_list[group]
                    sc.pp.neighbors(pred_adata, use_rep = 'latent_space')
                    sc.tl.umap(pred_adata)
                    np.save(save_dir + f'umap_{group}.npy', pred_adata.obsm['X_umap'])
                    self.run[f'evaluation/{group}/umap'].track_files(save_dir + f'umap_{group}.npy')
                    sc.set_figure_params(figsize=(15, 10), dpi = 300)
                    fig_class = sc.pl.umap(pred_adata, color = f'true_{self.class_key}', size = 10,return_fig = True)
                    fig_pred = sc.pl.umap(pred_adata, color = f'{self.class_key}_pred', size = 10,return_fig = True)
                    fig_batch = sc.pl.umap(pred_adata, color = self.batch_key, size = 10,return_fig = True)
                    fig_split = sc.pl.umap(pred_adata, color = 'train_split', size = 10,return_fig = True)
                    self.run[f'evaluation/{group}/true_umap'].upload(fig_class)
                    self.run[f'evaluation/{group}/pred_umap'].upload(fig_pred)
                    self.run[f'evaluation/{group}/batch_umap'].upload(fig_batch)
                    self.run[f'evaluation/{group}/split_umap'].upload(fig_split)

            self.run.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument('--dataset_name', type = str, default = 'htap_final_by_batch', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--class_key', type = str, default = 'celltype', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'donor', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')# TODO :remove, we always want to do that
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')
    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=5000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")

    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = None, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    
    
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')

    run_file = parser.parse_args()
    print(run_file.class_key, run_file.batch_key)
    working_dir = '/home/acollin/dca_permuted_workflow/'
    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    experiment.process_dataset()

    kf = GroupKFold(5)
    for i, (train_index, val_index) in enumerate(kf.split(experiment.dataset.adata_train_extended.X, experiment.dataset.adata_train_extended.obs[experiment.class_key], experiment.dataset.adata_train_extended.obs[experiment.batch_key])):
        for model in ['pca_svm','harmony_svm']:
            print(i)
            experiment.keep_obs = list(experiment.dataset.adata_train_extended.obs[experiment.batch_key][train_index].unique()) # keeping only train idx
            experiment.split_train_test_val()
            experiment.train_model(model)
            experiment.compute_metrics()


    
