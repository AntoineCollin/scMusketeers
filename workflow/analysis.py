try :
    from .load import load_runfile
    from .dataset import Dataset
    from .predictor import MLP_Predictor
    from .model import DCA_Permuted
    from .workflow import Workflow
    from .runfile_handler import RunFile
    from .clust_compute import *
except ImportError:
    from load import load_runfile
    from dataset import Dataset
    from predictor import MLP_Predictor
    from model import DCA_Permuted
    from workflow import Workflow
    from runfile_handler import RunFile
    from clust_compute import *
    
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, plot_confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc


class AnalysisWorkflow:
    def __init__(self, working_dir, id_list):
        '''
        At the moment, most of the functions might not support workflows coming from different datasets (especially, it's better if every latent spaces has the same number of obs)
        '''
        self.working_dir = working_dir
        self.runfile_paths = {wf_id: RunFile(working_dir=working_dir, workflow_ID=wf_id).run_file_path for wf_id in id_list}
        self.workflow_list = {wf_id: Workflow(working_dir=working_dir, yaml_name=rf_path) for wf_id, rf_path in self.runfile_paths.items()}
        self.true_class = dict()
        self.pred_class = dict()
        self.corrected_counts = dict()
        self.balanced_accuracy_scores = dict()
        self.accuracy_scores = dict()
        
        self.runfile_csv_path = self.working_dir + '/runfile_dir/runfile_list.csv'
        self.runfile_df = pd.read_csv(self.runfile_csv_path, index_col = 'index').loc[id_list,:]
        
        self.metrics_table = self.runfile_df.copy()               
        
    def load_corrected_counts(self):
        for workflow in self.workflow_list.values():
            workflow.load_corrected()
        self.corrected_counts = {wf_id: workflow.corrected_count for wf_id, workflow in self.workflow_list.items()}

    
    def load_latent_spaces(self):
        for workflow in self.workflow_list.values():
            workflow.load_results()
        self.latent_spaces = {wf_id: workflow.latent_space for wf_id, workflow in self.workflow_list.items()}
        self.true_class = {wf_id: workflow.latent_space.obs[f'true_{workflow.class_key}'] for wf_id, workflow in self.workflow_list.items()}
        self.pred_class = {wf_id: workflow.latent_space.obs[f'{workflow.class_key}_pred'] for wf_id, workflow in self.workflow_list.items()}
        self.pred_tables = {wf_id: workflow.latent_space.obs.loc[:,[f'true_{workflow.class_key}',f'{workflow.class_key}_pred', 'train_split']] for wf_id, workflow in self.workflow_list.items()}
#         for wf_id,workflow in self.workflow_list.items():
#             class_key = self.workflow_list[wf_id].class_key
#             true_celltype = self.workflow_list[wf_id].true_celltype
#             false_celltype = self.workflow_list[wf_id].false_celltype
#             false_true_only = pd.Series(['Other'] * workflow.latent_space.n_obs, index = workflow.latent_space.obs.index)
#             false_true_only[workflow.latent_space.obs[f'true_{class_key}'] == true_celltype] = true_celltype
#             false_true_only[workflow.latent_space.obs[f'true_{class_key}'] == false_celltype] = false_celltype
#             false_true_only[workflow.latent_space.obs['faked']] = f'fake {false_celltype} - true {true_celltype}'        
#             workflow.latent_space.obs['false_true_only'] = false_true_only
#             false_only = pd.Series(['Other'] * workflow.latent_space.n_obs, index = workflow.latent_space.obs.index)
#             false_only[workflow.latent_space.obs['faked']] = f'fake {false_celltype} - true {true_celltype}'        
#             workflow.latent_space.obs['false_only'] = false_only
#         self.latent_spaces = {wf_id: workflow.latent_space for wf_id, workflow in self.workflow_list.items()}
    
    def subsample_true_false(self, keep='true and false'):
        '''
        Use only when celltypes have been faked in the workflows. Keeps only a few cells hein 
        keep is either 'true', 'false' or 'true and false'.
        '''
        for wf_id, latent_space in self.latent_spaces.items():
            class_key = self.workflow_list[wf_id].class_key
            if keep == 'true':
                keep_celltype = [self.workflow_list[wf_id].true_celltype]
            elif keep == 'false':
                keep_celltype = [self.workflow_list[wf_id].false_celltype]
            elif keep == 'true and false':
                keep_celltype = [self.workflow_list[wf_id].false_celltype, self.workflow_list[wf_id].true_celltype]
            self.latent_spaces[wf_id] = latent_space[latent_space.obs[f'true_{class_key}'].isin(keep_celltype),:].copy()
            self.workflow_list[wf_id].latent_space =  latent_space[latent_space.obs[f'true_{class_key}'].isin(keep_celltype),:].copy()
            self.true_class = {wf_id: self.latent_spaces[wf_id].obs[f'true_{workflow.class_key}'] for wf_id, workflow in self.workflow_list.items()}
            self.pred_class = {wf_id: self.latent_spaces[wf_id].obs[f'{workflow.class_key}_pred'] for wf_id, workflow in self.workflow_list.items()}
            self.pred_tables = {wf_id: self.latent_spaces[wf_id].obs.loc[:,[f'true_{workflow.class_key}',f'{workflow.class_key}_pred', 'train_split']] for wf_id, workflow in self.workflow_list.items()}
                
    
    def subsample_on_obs(self, obs_filter, condition=True):
        '''
        Reduces every workflow on a common obs condition. If obs_filter is a boolean Series, no need to specify condition.
        The True and False value might be different between workflows therefore, workflows won't share the same cells after this operation
        '''
        if (type(condition) == str) or (type(condition) == bool):
                condition = [condition]
        for wf_id, latent_space in self.latent_spaces.items():
            self.latent_spaces[wf_id] = latent_space[latent_space.obs[obs_filter].isin(condition),:].copy()
            self.workflow_list[wf_id].latent_space =  latent_space[latent_space.obs[obs_filter].isin(condition),:].copy()
            self.true_class = {wf_id: self.latent_spaces[wf_id].obs[f'true_{workflow.class_key}'] for wf_id, workflow in self.workflow_list.items()}
            self.pred_class = {wf_id: self.latent_spaces[wf_id].obs[f'{workflow.class_key}_pred'] for wf_id, workflow in self.workflow_list.items()}
            self.pred_tables = {wf_id: self.latent_spaces[wf_id].obs.loc[:,[f'true_{workflow.class_key}',f'{workflow.class_key}_pred', 'train_split']] for wf_id, workflow in self.workflow_list.items()}
    
    
    def subsample_on_index(self, cell_index):
        '''
        Reduces every workflow to cells specified in cell_index
        '''
        for wf_id, latent_space in self.latent_spaces.items():
            self.latent_spaces[wf_id] = latent_space[cell_index,:].copy()
            self.workflow_list[wf_id].latent_space =  latent_space[cell_index,:].copy()
            self.true_class = {wf_id: self.latent_spaces[wf_id].obs[f'true_{workflow.class_key}'] for wf_id, workflow in self.workflow_list.items()}
            self.pred_class = {wf_id: self.latent_spaces[wf_id].obs[f'{workflow.class_key}_pred'] for wf_id, workflow in self.workflow_list.items()}
            self.pred_tables = {wf_id: self.latent_spaces[wf_id].obs.loc[:,[f'true_{workflow.class_key}',f'{workflow.class_key}_pred', 'train_split']] for wf_id, workflow in self.workflow_list.items()}

            
    def add_obs_metadata(self, metadata):
        '''
        Add metadata to every latent_space. metadata should be a pandas DataFrame with correct number of observations
        '''
        for wf_id, latent_space in self.latent_spaces.items():
            assert latent_space.obs.shape[0] == metadata.shape[0], "metadata doesn't have the same number of cell as the latent spaces"
            latent_space.obs = pd.concat([latent_space.obs, metadata], axis=1)
        
        
    def compute_metrics(self, verbose=False): #         y_iterate = lambda:zip(self.workflow_list.keys(),self.true_class.values(),self.pred_class.values())
        def verbose_print(to_print):
            if verbose:
                print(to_print)
        
        verbose_print('computing balanced_accuracy_scores')
        self.balanced_accuracy_scores = {wf_id: balanced_accuracy(adata=workflow.latent_space, 
                                                                  partition_key=f'{workflow.class_key}_pred',
                                                                  reference=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.balanced_accuracy_scores = pd.Series(self.balanced_accuracy_scores, name = 'balanced_accuracy_scores')
        
        self.balanced_accuracy_scores_test = {wf_id: balanced_accuracy(adata=workflow.latent_space[workflow.latent_space.obs['train_split'] == 'test'], 
                                                                  partition_key=f'{workflow.class_key}_pred',
                                                                  reference=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.balanced_accuracy_scores_test = pd.Series(self.balanced_accuracy_scores_test, name = 'balanced_accuracy_scores_test')
        
        verbose_print('computing accuracy_scores')
        self.accuracy_scores = {wf_id: accuracy(adata=workflow.latent_space, 
                                                         partition_key=f'{workflow.class_key}_pred',
                                                         reference=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.accuracy_scores = pd.Series(self.accuracy_scores,  name = 'accuracy_scores')

        self.accuracy_scores_test = {wf_id: accuracy(adata=workflow.latent_space[workflow.latent_space.obs['train_split'] == 'test'], 
                                                         partition_key=f'{workflow.class_key}_pred',
                                                         reference=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.accuracy_scores_test = pd.Series(self.accuracy_scores_test,  name = 'accuracy_scores_test')
        
#         verbose_print('computing silhouette_true')
#         self.silhouette_true = {wf_id: silhouette(adata=workflow.latent_space, 
#                                                   partition_key=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
#         self.silhouette_true = pd.Series(self.silhouette_true,  name = 'silhouette_true')
        
#         verbose_print('computing silhouette_pred')
#         self.silhouette_pred = {wf_id: silhouette(adata=workflow.latent_space, 
#                                                   partition_key=f'{workflow.class_key}_pred') for wf_id, workflow in self.workflow_list.items()}
#         self.silhouette_pred = pd.Series(self.silhouette_pred,  name = 'silhouette_pred')
        
        verbose_print('computing davies_bouldin_true')
        self.davies_bouldin_true = {wf_id: davies_bouldin(adata=workflow.latent_space, 
                                                  partition_key=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.davies_bouldin_true = pd.Series(self.davies_bouldin_true,  name = 'davies_bouldin_true')
        
        verbose_print('computing davies_bouldin_pred')
        self.davies_bouldin_pred = {wf_id: davies_bouldin(adata=workflow.latent_space, 
                                                  partition_key=f'{workflow.class_key}_pred') for wf_id, workflow in self.workflow_list.items()}
        self.davies_bouldin_pred = pd.Series(self.davies_bouldin_pred,  name = 'davies_bouldin_pred')
        
        
        
        verbose_print('computing nmi')
        self.nmi = {wf_id: nmi(adata=workflow.latent_space, 
                                     partition_key=f'{workflow.class_key}_pred',
                                     reference=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.nmi = pd.Series(self.nmi,  name = 'nmi')
        
        metrics_table_add = pd.concat([self.balanced_accuracy_scores, 
                                        self.accuracy_scores,
                                        self.balanced_accuracy_scores_test, 
                                        self.accuracy_scores_test,
#                                             self.silhouette_true, 
#                                             self.silhouette_pred,
                                        self.davies_bouldin_true,
                                        self.davies_bouldin_pred,
                                        self.nmi], axis = 1)
        
        self.metrics_table = pd.concat([metrics_table_add, self.metrics_table], axis=1)

        
    def compute_clustering_metrics(self, cluster_key, verbose=False):
        '''
        Compute the clustering metrics for a selected cluster for example batches
        '''
        self.davies_bouldin_cluster = {wf_id: davies_bouldin(adata=workflow.latent_space, 
                                                  partition_key=cluster_key) for wf_id, workflow in self.workflow_list.items()}
        self.davies_bouldin_cluster = pd.Series(self.davies_bouldin_cluster,  name = f'davies_bouldin_{cluster_key}')

        self.silhouette_cluster = {wf_id: silhouette(adata=workflow.latent_space, 
                                                  partition_key=f'true_{workflow.class_key}') for wf_id, workflow in self.workflow_list.items()}
        self.silhouette_cluster = pd.Series(self.silhouette_cluster,  name = f'silhouette_{cluster_key}')
        
        metrics_table_add = pd.concat([self.davies_bouldin_cluster, 
                                            self.silhouette_cluster], axis = 1)
        
        self.metrics_table = pd.concat([metrics_table_add, self.metrics_table], axis=1)

        
    
    def plot_confusion_matrices_single(self, workflow_ID, test_only,normalize = 'true', ax=None, **kwargs):
        if not test_only:
            y_true = self.true_class[workflow_ID]
            y_pred = self.pred_class[workflow_ID]
        else:
            y_true = self.true_class[workflow_ID][self.pred_tables[workflow_ID]['train_split'] == 'test']
            y_pred = self.pred_class[workflow_ID][self.pred_tables[workflow_ID]['train_split'] == 'test']
        labels = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        conf_mat = confusion_matrix(y_true, y_pred, labels = labels, normalize=normalize)
        confusion_to_plot=pd.DataFrame(conf_mat, index = labels, columns=labels)
        plt.figure(figsize = (15,15))
        ax = sns.heatmap(confusion_to_plot, annot=True, ax=ax, **kwargs)
        return confusion_matrix, ax
    
    
    def plot_confusion_matrices_multiple(self, IDs_to_plot = 'all', params = ['dataset_name']):
        '''
        This plots confusion matrices 
        IDs_to_plot should be either a list of the IDs to plot or 'all' in which case all the indices of the object are plotted
        '''
        if IDs_to_plot == 'all':
            IDs_to_plot = list(self.workflow_list.keys())
        n_plots = len(IDs_to_plot)
        f, axes = plt.subplots(n_plots,2,figsize = (30,12.5*n_plots))
        i=0
        for wf_id in IDs_to_plot:
            self.plot_prediction_results_single(workflow_ID = wf_id, test_only = False, ax=axes[i,0], normalize = 'true')
            axes[i,0].set_title([f'{param} = {getattr(self.workflow_list[wf_id], param)}' for param in params])
            axes[i,0].set_xlabel(f'worflow_{wf_id}, acc={round(self.accuracy_scores[wf_id],2)}, weighted_acc={round(self.balanced_accuracy_scores[wf_id],2)}')
            
            self.plot_prediction_results_single(workflow_ID = wf_id, test_only = True, ax=axes[i,1], normalize = 'true')
            axes[i,1].set_xlabel(f'worflow_{wf_id}_test_only, acc={round(self.accuracy_scores[wf_id],2)}, weighted_acc={round(self.balanced_accuracy_scores[wf_id],2)}')
            i+=1
        return f
    
    
    def plot_umaps(self, IDs_to_plot='all', params=['dataset'], **kwargs):
        '''
        params are the params to specify in the umap title
        '''
        if IDs_to_plot == 'all':
            IDs_to_plot = self.workflow_list.keys()
        for wf_id in IDs_to_plot:
            sc.pl.umap(self.latent_spaces[wf_id], title = f'workflow_{wf_id}' + str([f'{param} = {getattr(self.workflow_list[wf_id], param)}' for param in params]), **kwargs)

    
    
    def plot_comparative_boxplot(self, x, y, hue, title = None, IDs_to_plot = 'all', **kwargs):     
        metrics_to_plot = self.metrics_table.copy()
        if IDs_to_plot != 'all':
            metrics_to_plot = metrics_to_plot.loc[metrics_to_plot['workflow_ID'].isin(IDs_to_plot)]
            print(metrics_to_plot)
        if type(hue) == list:
            meta_hue = '_'.join(hue)
            col_to_plot = pd.Series(metrics_to_plot[hue[0]].astype(str),index = metrics_to_plot.index)
            for h in hue[1:] :
                col_to_plot =  col_to_plot + '_' + metrics_to_plot[h].astype(str)
            metrics_to_plot[meta_hue] = col_to_plot
        else :
            meta_hue = hue
        sns.set(rc={'figure.figsize':(11.7,8.27)})
        
        sns.boxplot(x=x, y = y, hue = meta_hue,
                    data = metrics_to_plot, **kwargs)

        plt.xticks(rotation =90)
        plt.legend(title = x)
        if title:
            plt.title(title)
        else: 
            plt.title(f'{y} split by {x}')
    
    def plot_class_confidence(self, ID, mode='box', layout = True, **kwargs):
        '''
        mode is either bar (average) or box (boxplot)
        '''
        adata = self.latent_spaces[ID]
        workflow = self.workflow_list[ID]
        true_key = f'true_{workflow.class_key}'
        pred_key = f'{workflow.class_key}_pred'
        y_pred_raw = pd.DataFrame(adata.obsm['y_pred_raw'], index = adata.obs_names, columns = adata.uns['prediction_decoder'])# n_obs x n_class matrix
        y_pred_raw = y_pred_raw[adata.obs[true_key].cat.categories]
        class_df_dict = {ct : y_pred_raw.loc[adata.obs[true_key] == ct, :] for ct in adata.obs[true_key].cat.categories}
        mean_acc_dict = {ct : df.mean(axis = 0) for ct, df in class_df_dict.items()}

        f, axes = plt.subplots(3,3, constrained_layout=layout)
    #     plt.constrained_layout()
        i = 0
        for ct, df in class_df_dict.items():
            r = i // 3
            c = i % 3
            ax = axes[r,c]
            if mode == 'box':
                df.plot.box(ax = ax, figsize = (20,15), ylim = (-0.01,1.01), xlabel = adata.obs[true_key].cat.categories, **kwargs)
            if mode == 'bar':
                df = mean_acc_dict[ct]
                df.plot.bar(ax = ax, figsize = (20,15), ylim = (0,1), **kwargs)
            ax.tick_params(axis='x', labelrotation=90 )
            ax.set_title(ct + f'- {class_df_dict[ct].shape[0]} cells')
            i+=1
        
        
    def plot_size_conf_correlation(self,ID):
        adata = self.latent_spaces[ID]
        workflow = self.workflow_list[ID]
        true_key = f'true_{workflow.class_key}'
        pred_key = f'{workflow.class_key}_pred'
        y_pred_raw = pd.DataFrame(adata.obsm['y_pred_raw'], index = adata.obs_names, columns = adata.uns['prediction_decoder'])# n_obs x n_class matrix
        class_df_dict = {ct : y_pred_raw.loc[adata.obs[true_key] == ct, :] for ct in adata.obs[true_key].cat.categories} # The order of the plot is defined here (adata.obs['true_louvain'].cat.categories)
        mean_acc_dict = {ct : df.mean(axis = 0) for ct, df in class_df_dict.items()}

        f, axes = plt.subplots(1,2, figsize = (10,5))
        f.suptitle('correlation between confidence and class size')
        pd.Series({ct : class_df_dict[ct].shape[0] for ct in mean_acc_dict.keys()}).plot.bar(ax = axes[0])
        pd.Series({ct : mean_acc_dict[ct][ct] for ct in mean_acc_dict.keys()}).plot.bar(ax =axes[1])
        
        
    def plot_class_accuracy(self,ID , layout = True, **kwargs):
        '''
        mode is either bar (average) or box (boxplot)
        '''
        adata = self.latent_spaces[ID]
        workflow = self.workflow_list[ID]
        true_key = f'true_{workflow.class_key}'
        pred_key = f'{workflow.class_key}_pred'
        labels = adata.obs[true_key].cat.categories
        conf_mat = pd.DataFrame(confusion_matrix(adata.obs[true_key], adata.obs[pred_key], labels=labels),index = labels, columns = labels)

        f, axes = plt.subplots(3,3, constrained_layout=layout)
        f.suptitle("Accuracy & confusion by celltype")
    #     plt.constrained_layout()
        i = 0
        for ct in labels:
            r = i // 3
            c = i % 3
            ax = axes[r,c]
            df = conf_mat.loc[ct,:]/conf_mat.loc[ct,:].sum()  
            df.plot.bar(ax = ax, figsize = (20,15), ylim = (0,1), **kwargs)
            ax.tick_params(axis='x', labelrotation=90 )
            ax.set_title(ct + f'- {conf_mat.loc[ct,:].sum()} cells')
            i+=1