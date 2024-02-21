try :
    from .load import load_runfile
    from ...scpermut.workflow.dataset import Dataset
    from .predictor import MLP_Predictor
    from .model import DCA_Permuted,Scanvi,DCA_into_Perm, ScarchesScanvi_LCA
    
except ImportError:
    from obsolete.old_workflow.load import load_runfile
    from dataset import Dataset
    from obsolete.old_workflow.predictor import MLP_Predictor
    from obsolete.old_workflow.model import DCA_Permuted,Scanvi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import time
import yaml
import pickle
import anndata
import pandas as pd
import scanpy as sc
import anndata
import numpy as np 
import os
import sys

workflow_ID = 'workflow_ID'

dataset = 'dataset'
dataset_name = 'dataset_name'
class_key = 'class_key'

dataset_normalize = 'dataset_normalize'
filter_min_counts = 'filter_min_counts'
normalize_size_factors = 'normalize_size_factors'
scale_input = 'scale_input'
logtrans_input = 'logtrans_input'
use_hvg = 'use_hvg'
    
model_spec = 'model_spec'
model_name = 'model_name'
ae_type = 'ae_type'
hidden_size = 'hidden_size'
hidden_dropout = 'hidden_dropout'
batchnorm = 'batchnorm'
activation = 'activation'
init = 'init'
batch_removal_weight = 'batch_removal_weight'

model_training_spec = 'model_training_spec'
epochs = 'epochs'
reduce_lr = 'reduce_lr'
early_stop = 'early_stop'
batch_size = 'batch_size'
optimizer = 'optimizer'
verbose = 'verbose'
threads = 'threads'
learning_rate = 'learning_rate'
n_perm = 'n_perm'
permute = 'permute'
change_perm = 'change_perm'
semi_sup = 'semi_sup'
unlabeled_category = 'unlabeled_category'
save_zinb_param = 'save_zinb_param'
use_raw_as_output = 'use_raw_as_output'
contrastive_margin = 'contrastive_margin'
same_class_pct = 'same_class_pct'

dataset_train_split = 'dataset_train_split'
mode = 'mode'
pct_split = 'pct_split'
obs_key = 'obs_key'
n_keep = 'n_keep'
split_strategy = 'split_strategy'
keep_obs = 'keep_obs'
train_test_random_seed = 'train_test_random_seed'
use_TEST = 'use_TEST'
obs_subsample = 'obs_subsample'
    
dataset_fake_annotation = 'dataset_fake_annotation'
make_fake = 'make_fake'
true_celltype = 'true_celltype'
false_celltype = 'false_celltype'
pct_false = 'pct_false'

predictor_spec = 'predictor_spec'
predictor_model = 'predictor_model' 
predict_key = 'predict_key' 
predictor_hidden_sizes = 'predictor_hidden_sizes'
predictor_epochs = 'predictor_epochs'
predictor_batch_size = 'predictor_batch_size'
predictor_activation = 'predictor_activation'



class Workflow:
    def __init__(self, run_file, working_dir): 
        '''
        run_file : a dictionary outputed by the function load_runfile
        '''
        self.run_file = run_file
        self.workflow_ID = self.run_file[workflow_ID]
        # dataset identifiers
        self.dataset_name = self.run_file[dataset][dataset_name]
        self.class_key = self.run_file[dataset][class_key]
        # normalization parameters
        self.filter_min_counts = self.run_file[dataset_normalize][filter_min_counts]
        self.normalize_size_factors = self.run_file[dataset_normalize][normalize_size_factors]
        self.scale_input = self.run_file[dataset_normalize][scale_input]
        self.logtrans_input = self.run_file[dataset_normalize][logtrans_input]
        self.use_hvg = self.run_file[dataset_normalize][use_hvg]
        # model parameters
        self.model_name = self.run_file[model_spec][model_name]
        self.ae_type = self.run_file[model_spec][ae_type]
        self.hidden_size = self.run_file[model_spec][hidden_size] # If hidden size is a integer, gives the size of the center layer. Otherwise, gives all layer sizes
        if type(self.hidden_size) == int:
            self.hidden_size = [2*self.hidden_size, self.hidden_size, 2*self.hidden_size]
        self.hidden_dropout = self.run_file[model_spec][hidden_dropout]
        if not self.hidden_dropout:
            self.hidden_dropout = 0 
        self.hidden_dropout = len(self.hidden_size) * [self.hidden_dropout]
        self.batchnorm = self.run_file[model_spec][batchnorm]
        self.activation = self.run_file[model_spec][activation]
        self.init = self.run_file[model_spec][init]
        self.batch_removal_weight = self.run_file[model_spec][batch_removal_weight]
        # model training parameters
        self.epochs = self.run_file[model_training_spec][epochs]
        self.reduce_lr = self.run_file[model_training_spec][reduce_lr]
        self.early_stop = self.run_file[model_training_spec][early_stop]
        self.batch_size = self.run_file[model_training_spec][batch_size]
        self.optimizer = self.run_file[model_training_spec][optimizer]
        self.verbose = self.run_file[model_training_spec][verbose]
        self.threads = self.run_file[model_training_spec][threads]
        self.learning_rate = self.run_file[model_training_spec][learning_rate]
        self.n_perm = self.run_file[model_training_spec][n_perm]
        self.permute = self.run_file[model_training_spec][permute]
        self.change_perm = self.run_file[model_training_spec][change_perm]
        self.semi_sup = self.run_file[model_training_spec][semi_sup]
        self.unlabeled_category = self.run_file[model_training_spec][unlabeled_category]
        self.save_zinb_param = self.run_file[model_training_spec][save_zinb_param]
        self.use_raw_as_output = self.run_file[model_training_spec][use_raw_as_output]
        self.contrastive_margin =self.run_file[model_training_spec][contrastive_margin]
        self.same_class_pct=self.run_file[model_training_spec][same_class_pct]

        # train test split
        self.mode = self.run_file[dataset_train_split][mode]
        self.pct_split = self.run_file[dataset_train_split][pct_split]
        self.obs_key = self.run_file[dataset_train_split][obs_key]
        self.n_keep = self.run_file[dataset_train_split][n_keep]
        self.split_strategy = self.run_file[dataset_train_split][split_strategy]
        self.keep_obs = self.run_file[dataset_train_split][keep_obs]
        self.train_test_random_seed = self.run_file[dataset_train_split][train_test_random_seed]
        self.use_TEST = self.run_file[dataset_train_split][use_TEST]
        self.obs_subsample = self.run_file[dataset_train_split][obs_subsample]
        # Create fake annotations
        self.make_fake = self.run_file[dataset_fake_annotation][make_fake]
        self.true_celltype = self.run_file[dataset_fake_annotation][true_celltype]
        self.false_celltype = self.run_file[dataset_fake_annotation][false_celltype]
        self.pct_false = self.run_file[dataset_fake_annotation][pct_false]
        # predictor parameters
        self.predictor_model = self.run_file[predictor_spec][predictor_model]
        self.predict_key = self.run_file[predictor_spec][predict_key]
        self.predictor_hidden_sizes = self.run_file[predictor_spec][predictor_hidden_sizes]
        self.predictor_epochs = self.run_file[predictor_spec][predictor_epochs]
        self.predictor_batch_size = self.run_file[predictor_spec][predictor_batch_size]
        self.predictor_activation = self.run_file[predictor_spec][predictor_activation]
        
        self.latent_space = anndata.AnnData()
        self.corrected_count = anndata.AnnData()
        self.scarches_combined_emb = anndata.AnnData()
        self.DR_hist = dict()
        self.DR_model = None
        
        self.predicted_class = pd.Series()
        self.pred_hist = dict()
        
        self.data_dir = working_dir + '/data'
        self.result_dir = working_dir + '/results'
        self.result_path = self.result_dir + f'/result_ID_{self.workflow_ID}'
        self.DR_model_path = self.result_path + '/DR_model'
        self.predictor_model_path = self.result_path + '/predictor_model'
        self.DR_history_path = self.result_path + '/DR_hist.pkl'
        self.pred_history_path = self.result_path + '/pred_hist.pkl'
        self.adata_path = self.result_path + '/latent.h5ad'
        self.corrected_count_path = self.result_path + '/corrected_counts.h5ad'
        self.scarches_combined_emb_path = self.result_path + '/combined_emb.h5ad'
        self.metric_path = self.result_path + '/metrics.csv'
        
        self.run_log_dir = working_dir + '/logs/run'
        self.run_log_path = self.run_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt'
        self.predict_log_dir = working_dir + '/logs/predicts'
        self.predict_log_path = self.predict_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt'
        self.umap_log_dir = working_dir + '/logs/umap'
        self.umap_log_path = self.umap_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt'
        self.metrics_log_dir = working_dir + '/logs/metrics'
        self.metrics_log_path = self.metrics_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt'
        
        self.start_time = time.time()
        self.stop_time = time.time()
        self.runtime_path = self.result_path + '/runtime.txt'

        self.run_done = False
        self.predict_done = False
        self.umap_done = False

        self.dataset = None
        self.model = None
        self.predictor = None
        
        self.training_kwds = {}
        self.network_kwds = {}
    
    def write_metric_log(self):
        open(self.metrics_log_path, 'a').close()
    
    def check_metric_log(self):
        return os.path.isfile(self.metrics_log_path)

    def write_run_log(self):
        open(self.run_log_path, 'a').close()
        
    def write_predict_log(self):
        open(self.predict_log_path, 'a').close()
    
    def write_umap_log(self):
        open(self.umap_log_path, 'a').close()

    
    def check_run_log(self):
        return os.path.isfile(self.run_log_path)
        
    def check_predict_log(self):
        return os.path.isfile(self.predict_log_path)
    
    def check_umap_log(self):
        return os.path.isfile(self.umap_log_path)

    def load_dataset(self) -> None:
        self.dataset = Dataset(dataset_dir = self.data_dir,
                               dataset_name = self.dataset_name,
                               class_key = self.class_key,
                               filter_min_counts = self.filter_min_counts,
                               normalize_size_factors = self.normalize_size_factors,
                               scale_input = self.scale_input,
                               logtrans_input = self.logtrans_input,
                               use_hvg = self.use_hvg,
                               n_perm = self.n_perm,
                               semi_sup = self.semi_sup,
                               unlabeled_category = self.unlabeled_category)
    
    def make_experiment(self):
        self.dataset = Dataset(dataset_dir = self.data_dir,
                               dataset_name = self.dataset_name,
                               class_key = self.class_key,
                               filter_min_counts = self.filter_min_counts,
                               normalize_size_factors = self.normalize_size_factors,
                               scale_input = self.scale_input,
                               logtrans_input = self.logtrans_input,
                               use_hvg = self.use_hvg,
                               n_perm = self.n_perm,
                               semi_sup = self.semi_sup,
                               unlabeled_category = self.unlabeled_category)
        if self.model_name == 'dca_permuted':
            print(self.model_name)
            self.model = DCA_Permuted(ae_type = self.ae_type,
                                     hidden_size = self.hidden_size,
                                     hidden_dropout = self.hidden_dropout, 
                                     batchnorm = self.batchnorm, 
                                     activation = self.activation, 
                                     init = self.init,
                                     batch_removal_weight = self.batch_removal_weight,
                                     epochs = self.epochs,
                                     reduce_lr = self.reduce_lr, 
                                     early_stop = self.early_stop, 
                                     batch_size = self.batch_size, 
                                     optimizer = self.optimizer,
                                     verbose = self.verbose,
                                     threads = self.threads, 
                                     learning_rate = self.learning_rate,
                                     training_kwds = self.training_kwds,
                                     network_kwds = self.network_kwds, 
                                     n_perm = self.n_perm, 
                                     permute = self.permute,
                                     change_perm = self.change_perm,
                                     class_key = self.class_key,
                                     unlabeled_category = self.unlabeled_category,
                                     use_raw_as_output=self.use_raw_as_output,
                                     contrastive_margin = self.contrastive_margin,
                                     same_class_pct = self.same_class_pct)
            
        if self.model_name == 'dca_into_perm':    
            self.model = DCA_into_Perm(ae_type = self.ae_type,
                                     hidden_size = self.hidden_size,
                                     hidden_dropout = self.hidden_dropout, 
                                     batchnorm = self.batchnorm, 
                                     activation = self.activation, 
                                     init = self.init,
                                     epochs = self.epochs,
                                     reduce_lr = self.reduce_lr, 
                                     early_stop = self.early_stop, 
                                     batch_size = self.batch_size, 
                                     optimizer = self.optimizer,
                                     verbose = self.verbose,
                                     threads = self.threads, 
                                     learning_rate = self.learning_rate,
                                     training_kwds = self.training_kwds,
                                     network_kwds = self.network_kwds,
                                     n_perm = self.n_perm, 
                                     change_perm = self.change_perm,
                                     class_key = self.class_key,
                                     unlabeled_category = self.unlabeled_category,
                                     use_raw_as_output=self.use_raw_as_output)
            
        if self.model_name == 'scanvi':
            self.model = Scanvi(class_key = self.class_key, unlabeled_category=self.unlabeled_category)
            # with method make_net(Dataset), train_net(Dataset)-> training hist, predict_net(Dataset) -> latent_space, corrected_count,        save_net(DR_model_path)

        if self.model_name == 'scarches_scanvi_LCA':
            self.model = ScarchesScanvi_LCA(class_key = self.class_key, unlabeled_category=self.unlabeled_category)
        self.dataset.load_dataset()
        self.dataset.normalize()
        self.dataset.train_split(mode = self.mode,
                            pct_split = self.pct_split,
                            obs_key = self.obs_key,
                            n_keep = self.n_keep,
                            keep_obs = self.keep_obs,
                            split_strategy = self.split_strategy,
                            obs_subsample = self.obs_subsample,
                            train_test_random_seed = self.train_test_random_seed)
        if self.make_fake:
            self.dataset.fake_annotation(true_celltype=self.true_celltype,
                                    false_celltype=self.false_celltype,
                                    pct_false=self.pct_false,
                                    train_test_random_seed = self.train_test_random_seed)
        if self.n_perm > 1:
            self.dataset.small_clusters_totest()
        print('dataset has been preprocessed')
        self.model.make_net(self.dataset)
        self.DR_hist = self.model.train_net(self.dataset)
        if 'scarches' in self.model_name:
            self.latent_space, self.scarches_combined_emb = self.model.predict_net(self.dataset)
        else: 
            self.latent_space, self.corrected_count = self.model.predict_net(self.dataset)
#         if self.predictor_model == 'MLP':
#             self.predictor = MLP_Predictor(latent_space = self.latent_space,
#                                            predict_key = self.predict_key,
#                                            predictor_hidden_sizes = self.predictor_hidden_sizes,
#                                            predictor_epochs = self.predictor_epochs,
#                                            predictor_batch_size = self.predictor_batch_size,
#                                            predictor_activation = self.predictor_activation,
#                                            unlabeled_category = self.unlabeled_category)
#             self.predictor.preprocess_one_hot()
#             self.predictor.build_predictor()
#             self.predictor.train_model()
#             self.predictor.predict_on_test()
#             self.pred_hist = self.predictor.train_history
#             self.predicted_class = self.predictor.y_pred
#             self.latent_space.obs[f'{self.class_key}_pred'] = self.predicted_class
        if type(self.run_file[model_spec][hidden_size]) == tuple:
            stored_rf = str(self.run_file[model_spec][hidden_size])
        self.latent_space.uns['runfile_dict'] = stored_rf
        self.run_done = True
        self.stop_time = time.time()

    
    def compute_prediction_only(self):
        self.latent_space = sc.read_h5ad(self.adata_path)
        if self.predictor_model == 'MLP':
            self.predictor = MLP_Predictor(latent_space = self.latent_space,
                                           predict_key = self.predict_key,
                                           predictor_hidden_sizes = self.predictor_hidden_sizes,
                                           predictor_epochs = self.predictor_epochs,
                                           predictor_batch_size = self.predictor_batch_size,
                                           predictor_activation = self.predictor_activation,
                                           unlabeled_category = self.unlabeled_category)
            self.predictor.preprocess_one_hot()
            self.predictor.build_predictor()
            self.predictor.train_model()
            self.predictor.predict_on_test()
            self.pred_hist = self.predictor.train_history
            self.predicted_class = self.predictor.y_pred
            self.latent_space.obs[f'{self.class_key}_pred'] = self.predicted_class
            self.predict_done = True
        
    
    def compute_umap(self):
        sc.tl.pca(self.latent_space)
        sc.pp.neighbors(self.latent_space, use_rep = 'X', key_added = f'neighbors_{self.model_name}')
        sc.pp.neighbors(self.latent_space, use_rep = 'X_pca', key_added = 'neighbors_pca')
        sc.tl.umap(self.latent_space, neighbors_key = f'neighbors_{self.model_name}')
        print(self.latent_space)
        print(self.adata_path)
        self.latent_space.write(self.adata_path)
        self.write_umap_log()
        self.umap_done = True

    def predict_knn_classifier(self, n_neighbors = 50, embedding_key=None, return_clustering = False):
        adata_train = self.latent_space[self.latent_space.obs['TRAIN_TEST_split'] == 'train']
        adata_train = adata_train[adata_train.obs[self.class_key] != self.unlabeled_category]
       
        knn_class = KNeighborsClassifier(n_neighbors = n_neighbors)
        
        if embedding_key:
            knn_class.fit(adata_train.obsm[embedding_key], adata_train.obs[self.class_key])        
            pred_clusters = knn_class.predict(self.latent_space.X)
            if return_clustering:
                return pred_clusters
            else:
                self.latent_space.obs[f'{self.class_key}_{embedding_key}_knn_classifier{n_neighbors}_pred'] = pred_clusters
        else :
            knn_class.fit(adata_train.X, adata_train.obs[self.class_key])
            pred_clusters = knn_class.predict(self.latent_space.X)
            if return_clustering:
                return pred_clusters
            else:
                self.latent_space.obs[f'{self.class_key}_knn_classifier{n_neighbors}_pred'] = pred_clusters


    def predict_kmeans(self, embedding_key=None):
        n_clusters = len(np.unique(self.latent_space.obs[self.class_key]))

        kmeans = KMeans(n_clusters = n_clusters)
        
        if embedding_key:
            kmeans.fit_predict(self.latent_space.obsm[embedding_key])
            self.latent_space.obs[f'{embedding_key}_kmeans_pred'] = kmeans.predict(self.latent_space.obsm[embedding_key])
        else :
            kmeans.fit_predict(self.latent_space.X)
            self.latent_space.obs[f'kmeans_pred'] = kmeans.predict(self.latent_space.X)

    def compute_leiden(self, **kwargs):
        sc.tl.leiden(self.latent_space, key_added = 'leiden_latent', neighbors_key = f'neighbors_{self.model_name}', **kwargs)
    
    def save_results(self):
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)
        try:
            self.latent_space.write(self.adata_path)
        except NotImplementedError:
            self.latent_space.uns['runfile_dict'] = dict() # Quick workaround
            self.latent_space.write(self.adata_path)
#         self.model.save_net(self.DR_model_path)
#         if self.predictor_model:
#             self.predictor.save_model(self.predictor_model_path)
        if self.scarches_combined_emb:
            self.scarches_combined_emb.write(self.scarches_combined_emb_path)
        if self.save_zinb_param:
            if self.corrected_count :
                self.corrected_count.write(self.corrected_count_path)
        if self.model_name != 'scanvi':
            if self.DR_hist:
                with open(self.DR_history_path, 'wb') as file_pi:
                    pickle.dump(self.DR_hist.history, file_pi)
        if self.predict_done and self.pred_hist:
            with open(self.pred_history_path, 'wb') as file_pi:
                pickle.dump(self.pred_hist.history, file_pi)
        if self.run_done:
            self.write_run_log()
        if self.predict_done:
            self.write_predict_log()
        if self.umap_done:
            self.write_umap_log()
        if not os.path.exists(self.result_path):
            metric_series = pd.DataFrame(index = [self.workflow_ID], data={'workflow_ID':pd.Series([self.workflow_ID], index = [self.workflow_ID])})
            metric_series.to_csv(self.metric_path)
        if not os.path.exists(self.runtime_path):
            with open(self.runtime_path, 'w') as f:
                f.write(str(self.stop_time - self.start_time))


    def load_results(self):
        if os.path.isdir(self.result_path):
            try:
                self.latent_space = sc.read_h5ad(self.adata_path)
            except OSError as e:
                print(e)
                print(f'failed to load {self.workflow_ID}')
                return self.workflow_ID # Returns the failed id
        if self.check_run_log():
            self.run_done = True
        if self.check_predict_log():
            self.predict_done = True
        if self.check_umap_log():
            self.umap_done = True
            
    def load_corrected(self):
        if os.path.isdir(self.result_path):
            self.corrected_count = sc.read_h5ad(self.corrected_count_path)
            self.corrected_count.obs = self.latent_space.obs
            self.corrected_count.layers['X_dca_dropout'] = self.corrected_count.obsm['X_dca_dropout']
            self.corrected_count.layers['X_dca_dispersion'] = self.corrected_count.obsm['X_dca_dispersion']
            self.corrected_count.obsm['X_umap'] = self.latent_space.obsm['X_umap']

    def __str__(self):
        return str(self.run_file)