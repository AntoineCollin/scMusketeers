import keras
import sys
import os
try :
    from ...scmusketeers.workflow.dataset import Dataset, load_dataset
    from ...scmusketeers.tools.utils import scanpy_to_input, default_value, str2bool,densify
    from ...scmusketeers.tools.clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg

except ImportError:
    from dataset import Dataset, load_dataset
    from obsolete.old_workflow.predictor import MLP_Predictor
    from obsolete.old_workflow.model import DCA_Permuted,Scanvi
    from utils import scanpy_to_input, default_value, str2bool,densify
    from clust_compute import nn_overlap, batch_entropy_mixing_score,lisi_avg
# from dca.utils import str2bool,tuple_to_scalar
import argparse
import functools
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from tools.models import DANN_AE
from tools.permutationÒ import batch_generator_training_permuted

from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef, f1_score,cohen_kappa_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,davies_bouldin_score,adjusted_rand_score,confusion_matrix

f1_score = functools.partial(f1_score, average = 'weighted')
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import os
import sys
import gc
import tensorflow as tf
import neptune
# from numba import cuda
from neptune.utils import stringify_unsupported

from ax.service.managed_loop import optimize
# from ax import RangeParameter, SearchSpace, ParameterType, FixedParameter, ChoiceParameter

physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)
    
# Reset Keras Session
def reset_keras():
    sess = tf.compat.v1.keras.backend.get_session()
    tf.compat.v1.keras.backend.clear_session()
    sess.close()
    sess = tf.compat.v1.keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect())

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

import subprocess as sp
import os

def get_gpu_memory(txt) :
    # command = "nvidia-smi --query-gpu=memory.used --format=csv"
    # memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    memory_free_values = tf.config.experimental.get_memory_info('GPU:0')['current']
    print(f'GPU memory usage in {txt} HERE: {memory_free_values/1e6} MB ')


class Workflow:
    def __init__(self, run_file, working_dir):
        '''
        run_file : a dictionary outputed by the function load_runfile
        '''
        self.run_file = run_file
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
        self.batch_size = self.run_file.batch_size
        # self.optimizer = self.run_file.optimizer
        # self.verbose = self.run_file[model_training_spec][verbose] # TODO : not implemented yet for DANN_AE
        # self.threads = self.run_file[model_training_spec][threads] # TODO : not implemented yet for DANN_AE
        self.learning_rate = self.run_file.learning_rate
        self.n_perm = 1
        self.semi_sup = False # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
        self.unlabeled_category = 'UNK' # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
      

        # train test split # TODO : Simplify this, or at first only use the case where data is split according to batch
        self.test_split_key = self.run_file.test_split_key
        self.mode = self.run_file.mode
        self.pct_split = self.run_file.pct_split
        self.obs_key = self.run_file.obs_key
        self.n_keep = self.run_file.n_keep
        self.split_strategy = self.run_file.split_strategy
        self.keep_obs = self.run_file.keep_obs
        self.train_test_random_seed = self.run_file.train_test_random_seed
        # self.use_TEST = self.run_file[dataset_train_split][use_TEST] # TODO : remove, obsolete in the case of DANN_AE
        self.obs_subsample = self.run_file.obs_subsample
        # Create fake annotations
        self.make_fake = self.run_file.make_fake
        self.true_celltype = self.run_file.true_celltype
        self.false_celltype = self.run_file.false_celltype
        self.pct_false = self.run_file.pct_false
        
        self.working_dir = working_dir
        self.data_dir = working_dir + '/data'
        
        self.start_time = time.time()
        self.stop_time = time.time()
        # self.runtime_path = self.result_path + '/runtime.txt'

        self.run_done = False
        self.predict_done = False
        self.umap_done = False

        self.dataset = None
        self.model = None
        self.predictor = None

        self.training_kwds = {}
        self.network_kwds = {}

        ##### TODO : Add to runfile



        self.clas_loss_name = self.run_file.clas_loss_name
        self.clas_loss_name = default_value(self.clas_loss_name, 'MSE')
        self.dann_loss_name = self.run_file.dann_loss_name
        self.dann_loss_name = default_value(self.dann_loss_name ,'categorical_crossentropy')
        self.rec_loss_name = self.run_file.rec_loss_name
        self.rec_loss_name = default_value(self.rec_loss_name , 'categorical_crossentropy')

        self.clas_loss_fn = None
        self.dann_loss_fn = None
        self.rec_loss_fn = None

        # self.weight_decay = self.run_file.weight_decay
        self.weight_decay = None
        self.optimizer_type = self.run_file.optimizer_type
        self.optimizer_type = default_value(self.optimizer_type , 'adam')

        self.clas_w = self.run_file.clas_w
        self.dann_w = self.run_file.dann_w
        self.rec_w = self.run_file.rec_w
        self.warmup_epoch = self.run_file.warmup_epoch

        self.num_classes = None
        self.num_batches = None

        self.ae_hidden_size = self.run_file.ae_hidden_size
        self.ae_hidden_size = default_value(self.ae_hidden_size , (128,64,128))
        self.ae_hidden_dropout = self.run_file.ae_hidden_dropout
        # self.ae_hidden_dropout = default_value(self.ae_hidden_dropout , None)
        self.ae_activation = self.run_file.ae_activation
        self.ae_activation = default_value(self.ae_activation , "relu")
        self.ae_output_activation = self.run_file.ae_output_activation
        self.ae_output_activation = default_value(self.ae_output_activation , "linear")
        self.ae_init = self.run_file.ae_init
        self.ae_init = default_value(self.ae_init , 'glorot_uniform')
        self.ae_batchnorm = self.run_file.ae_batchnorm
        self.ae_batchnorm = default_value(self.ae_batchnorm , True)
        self.ae_l1_enc_coef = self.run_file.ae_l1_enc_coef
        self.ae_l1_enc_coef = default_value(self.ae_l1_enc_coef , 0)
        self.ae_l2_enc_coef = self.run_file.ae_l2_enc_coef
        self.ae_l2_enc_coef = default_value(self.ae_l2_enc_coef , 0)

        self.class_hidden_size = self.run_file.class_hidden_size
        self.class_hidden_size = default_value(self.class_hidden_size , None) # default value will be initialize as [(bottleneck_size + num_classes)/2] once we'll know num_classes
        self.class_hidden_dropout = self.run_file.class_hidden_dropout
        self.class_batchnorm = self.run_file.class_batchnorm
        self.class_batchnorm = default_value(self.class_batchnorm , True)
        self.class_activation = self.run_file.class_activation
        self.class_activation = default_value(self.class_activation , 'relu')
        self.class_output_activation = self.run_file.class_output_activation
        self.class_output_activation = default_value(self.class_output_activation , 'softmax')

        self.dann_hidden_size = self.run_file.dann_hidden_size
        self.dann_hidden_size = default_value(self.dann_hidden_size , None) # default value will be initialize as [(bottleneck_size + num_batches)/2] once we'll know num_classes
        self.dann_hidden_dropout = self.run_file.dann_hidden_dropout
        self.dann_batchnorm = self.run_file.dann_batchnorm
        self.dann_batchnorm = default_value(self.dann_batchnorm , True)
        self.dann_activation = self.run_file.dann_activation
        self.dann_activation = default_value(self.dann_activation , 'relu')
        self.dann_output_activation =  self.run_file.dann_output_activation
        self.dann_output_activation = default_value(self.dann_output_activation , 'softmax')

        self.dann_ae = None

        self.pred_metrics_list = {'balanced_acc' : balanced_accuracy_score, 
                            'mcc' : matthews_corrcoef,
                            'f1_score': f1_score,
                            'KPA' : cohen_kappa_score,
                            'ARI': adjusted_rand_score,
                            'NMI': normalized_mutual_info_score,
                            'AMI':adjusted_mutual_info_score}

        self.clustering_metrics_list = {#'clisi' : lisi_avg, 
                                    'db_score' : davies_bouldin_score
                                    }

        self.batch_metrics_list = {'batch_entropy': batch_entropy_mixing_score,
                            #'ilisi': lisi_avg
                            }
        self.metrics = []

        self.mean_loss_fn = keras.metrics.Mean(name='total loss') # This is a running average : it keeps the previous values in memory when it's called ie computes the previous and current values
        self.mean_clas_loss_fn = keras.metrics.Mean(name='classification loss')
        self.mean_dann_loss_fn = keras.metrics.Mean(name='dann loss')
        self.mean_rec_loss_fn = keras.metrics.Mean(name='reconstruction loss')

        self.training_scheme = self.run_file.training_scheme

        self.log_neptune = self.run_file.log_neptune
        self.run = None

        self.hparam_path = self.run_file.hparam_path

    def make_experiment(self):# -> Any:
        # print(params)
        # self.use_hvg = params['use_hvg']
        # self.batch_size = params['batch_size']
        # self.clas_w =  params['clas_w']
        # self.dann_w = params['dann_w']
        # self.rec_w =  1
        # self.weight_decay =  params['weight_decay']
        # self.learning_rate = params['learning_rate']
        # self.warmup_epoch =  params['warmup_epoch']
        # self.dropout =  params['dropout']
        # self.layer1 = params['layer1']
        # self.layer2 =  params['layer2']
        # self.bottleneck = params['bottleneck']

        # self.ae_hidden_size = [self.layer1, self.layer2, self.bottleneck, self.layer2, self.layer1]

        # self.dann_hidden_dropout, self.class_hidden_dropout, self.ae_hidden_dropout = self.dropout, self.dropout, self.dropout

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
                               test_split_key= self.test_split_key,
                               unlabeled_category = self.unlabeled_category)
        
        # Processing dataset. Splitting train/test. 
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
        print('dataset has been preprocessed')
        self.dataset.create_inputs()

        get_gpu_memory('data loaded')
        adata_list = {'full': self.dataset.adata,
                      'train': self.dataset.adata_train,
                      'val': self.dataset.adata_val,
                      'test': self.dataset.adata_test}

        X_list = {'full': self.dataset.X,
                'train': self.dataset.X_train,
                'val': self.dataset.X_val,
                'test': self.dataset.X_test}

        y_list = {'full': self.dataset.y_one_hot,
                      'train': self.dataset.y_train_one_hot,
                      'val': self.dataset.y_val_one_hot,
                      'test': self.dataset.y_test_one_hot}

        batch_list = {'full': self.dataset.batch_one_hot,
                      'train': self.dataset.batch_train_one_hot,
                      'val': self.dataset.batch_val_one_hot,
                      'test': self.dataset.batch_test_one_hot}

        self.num_classes = len(np.unique(self.dataset.y_train))
        self.num_batches = len(np.unique(self.dataset.batch))

        get_gpu_memory('created data lists')

        bottleneck_size = int(self.ae_hidden_size[int(len(self.ae_hidden_size)/2)])

        self.class_hidden_size = default_value(self.class_hidden_size , (bottleneck_size + self.num_classes)/2) # default value [(bottleneck_size + num_classes)/2]
        self.dann_hidden_size = default_value(self.dann_hidden_size , (bottleneck_size + self.num_batches)/2) # default value [(bottleneck_size + num_batches)/2]

        if self.log_neptune :
            self.run = neptune.init_run(
                    project="becavin-lab/sc-permut",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
)
            self.run[f"parameters/model"] = "scPermut"
            for par,val in self.run_file.__dict__.items():
                self.run[f"parameters/{par}"] = stringify_unsupported(getattr(self, par))

            # for par,val in params.items():
            #     self.run[f"parameters/{par}"] = stringify_unsupported(val)
            self.run[f'parameters/ae_hidden_size'] = stringify_unsupported(self.ae_hidden_size)
        # Creation of model
        self.dann_ae = DANN_AE(ae_hidden_size=self.ae_hidden_size, 
                        ae_hidden_dropout=self.ae_hidden_dropout,
                        ae_activation=self.ae_activation,
                        ae_output_activation=self.ae_output_activation,
                        ae_init=self.ae_init,
                        ae_batchnorm=self.ae_batchnorm,
                        ae_l1_enc_coef=self.ae_l1_enc_coef,
                        ae_l2_enc_coef=self.ae_l2_enc_coef,
                        num_classes=self.num_classes,
                        class_hidden_size=self.class_hidden_size,
                        class_hidden_dropout=self.class_hidden_dropout,
                        class_batchnorm=self.class_batchnorm,
                        class_activation=self.class_activation,
                        class_output_activation=self.class_output_activation,
                        num_batches=self.num_batches,
                        dann_hidden_size=self.dann_hidden_size,
                        dann_hidden_dropout=self.dann_hidden_dropout,
                        dann_batchnorm=self.dann_batchnorm,
                        dann_activation=self.dann_activation,
                        dann_output_activation=self.dann_output_activation)
        get_gpu_memory('model created')
        self.optimizer = self.get_optimizer(self.learning_rate, self.weight_decay, self.optimizer_type)
        self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn = self.get_losses() # redundant
        self.training_scheme = self.get_scheme()
        start_time = time.time()

        # Training
        history = self.train_scheme(training_scheme=self.training_scheme,
                                    verbose = False,
                                    ae = self.dann_ae,
                                     adata_list= adata_list,
                                     X_list= X_list,
                                     y_list= y_list,
                                     batch_list= batch_list,
                                    #  optimizer= self.optimizer, # not an **loop_param since it resets between strategies
                                     clas_loss_fn = self.clas_loss_fn,
                                     dann_loss_fn = self.dann_loss_fn,
                                     rec_loss_fn = self.rec_loss_fn)
        stop_time = time.time()
        if self.log_neptune :
            self.run['evaluation/training_time'] = stop_time - start_time
        # TODO also make it on gpu with smaller batch size
        if self.log_neptune:
            neptune_run_id = self.run['sys/id'].fetch()
            for group in ['full', 'train', 'val', 'test']:
                with tf.device('CPU'):
                    input_tensor = {k:tf.convert_to_tensor(v) for k,v in scanpy_to_input(adata_list[group],['size_factors']).items()}
                    enc, clas, dann, rec = self.dann_ae(input_tensor, training=False).values() # Model predict       
                    clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
                    if group in ['train', 'val', 'test']:
                        for metric in self.pred_metrics_list: # only classification metrics ATM
                            self.run[f"evaluation/{group}/{metric}"] = self.pred_metrics_list[metric](adata_list[group].obs[f'true_{self.class_key}'], self.dataset.ohe_celltype.inverse_transform(clas))
                        for metric in self.clustering_metrics_list: # only classification metrics ATM
                            self.run[f"evaluation/{group}/{metric}"] = self.clustering_metrics_list[metric](enc, self.dataset.ohe_celltype.inverse_transform(clas))
                        if len(np.unique(np.asarray(batch_list[group].argmax(axis=1)))) >= 2: # If there are more than 2 batches in this group
                            for metric in self.batch_metrics_list:
                                self.run[f'evaluation/{group}/{metric}'] = self.batch_metrics_list[metric](enc, np.asarray(batch_list[group].argmax(axis=1).reshape(-1,)))
                        
                        labels = list(set(np.unique(adata_list[group].obs[f'true_{self.class_key}'])).union(set(np.unique(self.dataset.ohe_celltype.inverse_transform(clas)))))
                        cm_no_label = confusion_matrix(adata_list[group].obs[f'true_{self.class_key}'], self.dataset.ohe_celltype.inverse_transform(clas))
                        print(f'no label : {cm_no_label.shape}')
                        cm = confusion_matrix(adata_list[group].obs[f'true_{self.class_key}'], self.dataset.ohe_celltype.inverse_transform(clas), labels = labels)
                        cm_norm = cm / cm.sum(axis = 1, keepdims=True)
                        print(f'label : {cm.shape}')
                        cm_to_plot=pd.DataFrame(cm_norm, index = labels, columns=labels)
                        cm_to_save=pd.DataFrame(cm, index = labels, columns=labels)
                        cm_to_plot = cm_to_plot.fillna(value=0)
                        cm_to_save = cm_to_save.fillna(value=0)
                        cm_to_save.to_csv(save_dir + f'confusion_matrix_{group}.csv')
                        self.run[f'evaluation/{group}/confusion_matrix_file'].track_files(save_dir + f'confusion_matrix_{group}.csv')
                        size = len(labels)
                        f, ax = plt.subplots(figsize=(size/1.5,size/1.5))
                        sns.heatmap(cm_to_plot, annot=True, ax=ax,fmt='.2f')
                        show_mask = np.asarray(cm_to_plot>0.01)
                        print(f'label df : {cm_to_plot.shape}')
                        for text, show_annot in zip(ax.texts, (element for row in show_mask for element in row)):
                            text.set_visible(show_annot)
                    
                        self.run[f'evaluation/{group}/confusion_matrix'].upload(f)

                        # self.run[f'evaluation/{group}/knn_overlap'] = nn_overlap(enc, X_list[group])
                    if group == 'full':
                        if len(np.unique(np.asarray(batch_list[group].argmax(axis=1)))) >= 2: # If there are more than 2 batches in this group
                            for metric in self.batch_metrics_list:
                                self.run[f'evaluation/{group}/{metric}'] = self.batch_metrics_list[metric](enc, np.asarray(batch_list[group].argmax(axis=1).reshape(-1,)))
                        save_dir = self.working_dir + 'experiment_script/results/' + str(neptune_run_id) + '/'
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        y_pred = pd.DataFrame(self.dataset.ohe_celltype.inverse_transform(clas), index = adata_list[group].obs_names)
                        np.save(save_dir + f'latent_space_{group}.npy', enc.numpy())
                        y_pred.to_csv(save_dir + f'predictions_{group}.csv')
                        self.run[f'evaluation/{group}/latent_space'].track_files(save_dir + f'latent_space_{group}.npy')
                        self.run[f'evaluation/{group}/predictions'].track_files(save_dir + f'predictions_{group}.csv')

                        pred_adata = sc.AnnData(X = adata_list[group].X, obs = adata_list[group].obs, var = adata_list[group].var)
                        pred_adata.obs[f'{self.class_key}_pred'] = y_pred
                        pred_adata.obsm['latent_space'] = enc.numpy()
                        sc.pp.neighbors(pred_adata, use_rep = 'latent_space')
                        sc.tl.umap(pred_adata)
                        np.save(save_dir + f'umap_{group}.npy', pred_adata.obsm['X_umap'])
                        self.run[f'evaluation/{group}/umap'].track_files(save_dir + f'umap_{group}.npy')
                        sc.set_figure_params(figsize=(15, 10), dpi = 300)
                        fig_class = sc.pl.umap(pred_adata, color = f'true_{self.class_key}', size = 10,return_fig = True)
                        fig_pred = sc.pl.umap(pred_adata, color = f'{self.class_key}_pred', size = 10,return_fig = True)
                        fig_batch = sc.pl.umap(pred_adata, color = self.batch_key, size = 10,return_fig = True)
                        fig_split = sc.pl.umap(pred_adata, color = 'train_split', size = 10,return_fig = True)
                        self.run[f'evaluation/{group}/classif_umap'].upload(fig_class)
                        self.run[f'evaluation/{group}/pred_umap'].upload(fig_pred)
                        self.run[f'evaluation/{group}/batch_umap'].upload(fig_batch)
                        self.run[f'evaluation/{group}/split_umap'].upload(fig_split)

        # Redondant, à priori c'est le mcc qu'on a déjà calculé au dessus.
        with tf.device('GPU'):
            inp = scanpy_to_input(adata_list['val'],['size_factors'])
            inp = {k:tf.convert_to_tensor(v) for k,v in inp.items()}
            _, clas, dann, rec = self.dann_ae(inp, training=False).values()
            clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
            opt_metric = self.pred_metrics_list['mcc'](np.asarray(y_list['val'].argmax(axis=1)), clas.argmax(axis=1)) # We retrieve the last metric of interest
        if self.log_neptune:
            self.run.stop()
        del enc
        del clas
        del dann
        del rec
        del _
        del input_tensor
        del inp
        del self.dann_ae
        del self.dataset
        del history
        del self.optimizer
        del self.rec_loss_fn
        del self.clas_loss_fn
        del self.dann_loss_fn
        del self.metrics_list

        gc.collect()
        tf.keras.backend.clear_session()

        return opt_metric

    def train_scheme(self,
                    training_scheme,
                    verbose = True ,
                    **loop_params):
        """
        training scheme : dictionnary explaining the succession of strategies to use as keys with the corresponding number of epochs and use_perm as values.
                        ex :  training_scheme_3 = {"warmup_dann" : (10, False), "full_model":(10, False)}
        """
        history = {'train': {}, 'val': {}} # initialize history
        for group in history.keys():
            history[group] = {'total_loss':[],
                            'clas_loss':[],
                            'dann_loss':[],
                            'rec_loss':[]}
            for m in self.pred_metrics_list:
                history[group][m] = []

        # if self.log_neptune:
        #     for group in history:
        #         for par,val in history[group].items():
        #             self.run[f"training/{group}/{par}"] = []
        i = 0

        total_epochs = np.sum([n_epochs for _, n_epochs, _ in training_scheme])
        running_epoch = 0

        for (strategy, n_epochs, use_perm) in training_scheme:
            optimizer = self.get_optimizer(self.learning_rate, self.weight_decay, self.optimizer_type) # resetting optimizer state when switching strategy
            if verbose :
                print(f"Step number {i}, running {strategy} strategy with permuation = {use_perm} for {n_epochs} epochs")
                time_in = time.time()

                # Early stopping for those strategies only
            if strategy in  ['full_model', 'classifier_branch', 'permutation_only']:
                wait = 0
                best_epoch = 0
                patience = 10
                min_delta = 0.01
                if strategy == 'permutation_only':
                    monitored = 'rec_loss'
                    es_best = np.inf # initialize early_stopping
                else:
                    monitored = 'mcc'
                    es_best = -np.inf


            for epoch in range(1, n_epochs+1):
                running_epoch +=1
                print(f"Epoch {running_epoch}/{total_epochs}, Current strat Epoch {epoch}/{n_epochs}")
                history, _, _, _ ,_ = self.training_loop(history=history,
                                                             training_strategy=strategy,
                                                             use_perm=use_perm,
                                                             optimizer=optimizer,
                                                               **loop_params)

                if self.log_neptune:
                    for group in history:
                        for par,value in history[group].items():
                            self.run[f"training/{group}/{par}"].append(value[-1])
                            self.run['training/train/tf_GPU_memory'].append(tf.config.experimental.get_memory_info('GPU:0')['current']/1e6)
                if strategy in ['full_model', 'classifier_branch', 'permutation_only']:
                    # Early stopping
                    wait += 1
                    monitored_value = history['val'][monitored][-1]

                    if 'loss' in monitored:
                        if monitored_value < es_best - min_delta:
                            best_epoch = epoch
                            es_best = monitored_value
                            wait = 0
                            best_model = self.dann_ae.get_weights()
                    else :
                        if monitored_value > es_best + min_delta:
                            best_epoch = epoch
                            es_best = monitored_value
                            wait = 0
                            best_model = self.dann_ae.get_weights()
                    if wait >= patience:
                        print(f'Early stopping at epoch {best_epoch}, restoring model parameters from this epoch')
                        self.dann_ae.set_weights(best_model)
                        break

            if verbose:
                time_out = time.time()
                print(f"Strategy duration : {time_out - time_in} s")
        if self.log_neptune:
            self.run[f"training/{group}/total_epochs"] = running_epoch
        return history

    def training_loop(self, history,
                            ae,
                            adata_list,
                            X_list,
                            y_list,
                            batch_list,
                            optimizer,
                            clas_loss_fn,
                            dann_loss_fn,
                            rec_loss_fn,
                            use_perm=None,
                            training_strategy="full_model",
                            verbose = False):
        '''
        A consolidated training loop function that covers common logic used in different training strategies.

        training_strategy : one of ["full", "warmup_dann", "warmup_dann_no_rec", "classifier_branch", "permutation_only"]
            - full_model : trains the whole model, optimizing the 3 losses (reconstruction, classification, anti batch discrimination ) at once
            - warmup_dann : trains the dann, encoder and decoder with reconstruction (no permutation because unsupervised), maximizing the dann loss and minimizing the reconstruction loss
            - warmup_dann_no_rec : trains the dann and encoder without reconstruction, maximizing the dann loss only.
            - dann_with_ae : same as warmup dann but with permutation. Separated in two strategies because this one is supervised
            - classifier_branch : trains the classifier branch only, without the encoder. Use to fine tune the classifier once training is over
            - permutation_only : trains the autoencoder with permutations, optimizing the reconstruction loss without the classifier
        use_perm : True by default except form "warmup_dann" training strategy. Note that for training strategies that don't involve the reconstruction, this parameter has no impact on training
        '''

        self.unfreeze_all(ae) # resetting freeze state
        if training_strategy == "full_model":
            group = 'train'
        elif training_strategy == "warmup_dann":
            group = 'full' # unsupervised setting
            ae.classifier.trainable = False # Freezing classifier just to be sure but should not be necessary since gradient won't be propagating in this branch
            use_perm = False # no permutation for warming up the dann. No need to specify it in the no rec version since we don't reconstruct
        elif training_strategy == "warmup_dann_no_rec":
            group = 'full'
            self.freeze_block(ae, 'all_but_dann')
        elif training_strategy == "dann_with_ae":
            group = 'train'
            ae.classifier.trainable = False
            use_perm = True
        elif training_strategy == "classifier_branch":
            group = 'train'
            self.freeze_block(ae, 'all_but_classifier_branch') # traning only classifier branch
        elif training_strategy == "permutation_only":
            group = 'train'
            self.freeze_block(ae, 'all_but_autoencoder')
            use_perm = True

        if not use_perm:
            use_perm = True

        get_gpu_memory('before generator creation')
        batch_generator = batch_generator_training_permuted(X = X_list[group],
                                                            y = y_list[group],
                                                            batch_ID = batch_list[group],
                                                            sf = adata_list[group].obs['size_factors'],                    
                                                            ret_input_only=False,
                                                            batch_size=self.batch_size,
                                                            n_perm=1, 
                                                            use_perm=use_perm)
        get_gpu_memory('after generator creation')

        n_obs = adata_list[group].n_obs
        steps = n_obs // self.batch_size + 1
        n_steps = steps
        n_samples = 0

        self.mean_loss_fn.reset_state()
        self.mean_clas_loss_fn.reset_state()
        self.mean_dann_loss_fn.reset_state()
        self.mean_rec_loss_fn.reset_state()
        get_gpu_memory('before epoch')
        for step in range(1, n_steps + 1):
            input_batch, output_batch = next(batch_generator)
            # X_batch, sf_batch = input_batch.values()
            clas_batch, dann_batch, rec_batch = output_batch.values()

            with tf.GradientTape() as tape:
                input_batch = {k:tf.convert_to_tensor(v) for k,v in input_batch.items()}
                enc, clas, dann, rec = ae(input_batch, training=True).values()
                clas_loss = tf.reduce_mean(clas_loss_fn(clas_batch, clas))
                dann_loss = tf.reduce_mean(dann_loss_fn(dann_batch, dann))
                rec_loss = tf.reduce_mean(rec_loss_fn(rec_batch, rec))
                if training_strategy == "full_model":
                    loss = tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses)
                elif training_strategy == "warmup_dann":
                    loss = tf.add_n([self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses)
                elif training_strategy == "warmup_dann_no_rec":
                    loss = tf.add_n([self.dann_w * dann_loss] + ae.losses)
                elif training_strategy == "dann_with_ae":
                    loss = tf.add_n([self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses)
                elif training_strategy == "classifier_branch":
                    loss = tf.add_n([self.clas_w * clas_loss] + ae.losses)
                elif training_strategy == "permutation_only":
                    loss = tf.add_n([self.rec_w * rec_loss] + ae.losses)

            n_samples += enc.shape[0]
            gradients = tape.gradient(loss, ae.trainable_variables)
            
            optimizer.apply_gradients(zip(gradients, ae.trainable_variables))

            self.mean_loss_fn(loss.__float__())
            self.mean_clas_loss_fn(clas_loss.__float__())
            self.mean_dann_loss_fn(dann_loss.__float__())
            self.mean_rec_loss_fn(rec_loss.__float__())

            if verbose :
                self.print_status_bar(n_samples, n_obs, [self.mean_loss_fn, self.mean_clas_loss_fn, self.mean_dann_loss_fn, self.mean_rec_loss_fn], self.metrics)
        get_gpu_memory('after epoch')
        self.print_status_bar(n_samples, n_obs, [self.mean_loss_fn, self.mean_clas_loss_fn, self.mean_dann_loss_fn, self.mean_rec_loss_fn], self.metrics)
        history, _, clas, dann, rec = self.evaluation_pass(history, ae, adata_list, X_list, y_list, batch_list, clas_loss_fn, dann_loss_fn, rec_loss_fn)
        del input_batch
        return history, _, clas, dann, rec

    def evaluation_pass(self,history, ae, adata_list, X_list, y_list, batch_list, clas_loss_fn, dann_loss_fn, rec_loss_fn):
        '''
        evaluate model and logs metrics. Depending on "on parameter, computes it on train and val or train,val and test.

        on : "epoch_end" to evaluate on train and val, "training_end" to evaluate on train, val and "test".
        '''
        for group in ['train', 'val']: # evaluation round
            inp = {'counts':  densify(X_list[group]),'size_factors' : adata_list[group]['size_factors']}
            # inp = scanpy_to_input(adata_list[group],['size_factors'])
            get_gpu_memory('before eval')
            with tf.device('CPU'):
                inp = {k:tf.convert_to_tensor(v) for k,v in inp.items()}
                _, clas, dann, rec = ae(inp, training=False).values()
                get_gpu_memory('after eval')

        #         return _, clas, dann, rec
                clas_loss = tf.reduce_mean(clas_loss_fn(y_list[group], clas)).numpy()
                history[group]['clas_loss'] += [clas_loss]
                dann_loss = tf.reduce_mean(dann_loss_fn(batch_list[group], dann)).numpy()
                history[group]['dann_loss'] += [dann_loss]
                rec_loss = tf.reduce_mean(rec_loss_fn(X_list[group], rec)).numpy()
                history[group]['rec_loss'] += [rec_loss]
                history[group]['total_loss'] += [self.clas_w * clas_loss + self.dann_w * dann_loss + self.rec_w * rec_loss + np.sum(ae.losses)] # using numpy to prevent memory leaks
                # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

                clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
                for metric in self.pred_metrics_list: # only classification metrics ATM
                    history[group][metric] += [self.pred_metrics_list[metric](np.asarray(y_list[group].argmax(axis=1)).reshape(-1,), clas.argmax(axis=1))] # y_list are onehot encoded
        del inp
        return history, _, clas, dann, rec

    # def evaluation_pass_gpu(self,history, ae, adata_list, X_list, y_list, batch_list, clas_loss_fn, dann_loss_fn, rec_loss_fn):
    #     '''
    #     evaluate model and logs metrics. Depending on "on parameter, computes it on train and val or train,val and test.

    #     on : "epoch_end" to evaluate on train and val, "training_end" to evaluate on train, val and "test".
    #     '''
    #     for group in ['train', 'val']: # evaluation round
    #         # inp = scanpy_to_input(adata_list[group],['size_factors'])
    #         batch_generator = batch_generator_training_permuted(X = X_list[group],
    #                                                 y = y_list[group],
    #                                                 batch_ID = batch_list[group],
    #                                                 sf = adata_list[group].obs['size_factors'],                    
    #                                                 ret_input_only=False,
    #                                                 batch_size=self.batch_size,
    #                                                 n_perm=1, 
    #                                                 use_perm=use_perm)
    #         n_obs = adata_list[group].n_obs
    #         steps = n_obs // self.batch_size + 1
    #         n_steps = steps
    #         n_samples = 0

    #         clas_batch, dann_batch, rec_batch = output_batch.values()

    #         with tf.GradientTape() as tape:
    #             input_batch = {k:tf.convert_to_tensor(v) for k,v in input_batch.items()}
    #             enc, clas, dann, rec = ae(input_batch, training=True).values()
    #             clas_loss = tf.reduce_mean(clas_loss_fn(clas_batch, clas)).numpy()
    #             dann_loss = tf.reduce_mean(dann_loss_fn(dann_batch, dann)).numpy()
    #             rec_loss = tf.reduce_mean(rec_loss_fn(rec_batch, rec)).numpy()
    #     #         return _, clas, dann, rec
    #             history[group]['clas_loss'] += [clas_loss]
    #             history[group]['dann_loss'] += [dann_loss]
    #             history[group]['rec_loss'] += [rec_loss]
    #             history[group]['total_loss'] += [self.clas_w * clas_loss + self.dann_w * dann_loss + self.rec_w * rec_loss + np.sum(ae.losses)] # using numpy to prevent memory leaks
    #             # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

    #             clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
    #             for metric in self.metrics_list: # only classification metrics ATM
    #                 history[group][metric] += [self.metrics_list[metric](np.asarray(y_list[group].argmax(axis=1)), clas.argmax(axis=1))] # y_list are onehot encoded
    #     del inp
    #     return history, _, clas, dann, rec

    def freeze_layers(self, ae, layers_to_freeze):
        '''
        Freezes specified layers in the model.

        ae: Model to freeze layers in.
        layers_to_freeze: List of layers to freeze.
        '''
        for layer in layers_to_freeze:
            layer.trainable = False


    def freeze_block(self, ae, strategy):
        if strategy == "all_but_classifier_branch":
            layers_to_freeze = [ae.dann_discriminator, ae.enc, ae.dec, ae.ae_output_layer]
        elif strategy == "all_but_classifier":
            layers_to_freeze = [ae.dann_discriminator, ae.dec, ae.ae_output_layer]
        elif strategy == "all_but_dann_branch":
            layers_to_freeze = [ae.classifier, ae.dec, ae.ae_output_layer, ae.enc]
        elif strategy == "all_but_dann":
            layers_to_freeze = [ae.classifier, ae.dec, ae.ae_output_layer]
        elif strategy == "all_but_autoencoder":
            layers_to_freeze = [ae.classifier, ae.dann_discriminator]
        else:
            raise ValueError("Unknown freeze strategy: " + strategy)

        self.freeze_layers(ae, layers_to_freeze)


    def freeze_all(self, ae):
        for l in ae.layers:
            l.trainable = False

    def unfreeze_all(self, ae):
        for l in ae.layers:
            l.trainable = True


    def get_scheme(self):
        if self.training_scheme == 'training_scheme_1':
            self.training_scheme = [("warmup_dann", self.warmup_epoch, False),
                                  ("full_model", 100, False)] # This will end with a callback
        if self.training_scheme == 'training_scheme_2':
            self.training_scheme = [("warmup_dann", self.warmup_epoch, False),
                                ("permutation_only", 100, True),  # This will end with a callback
                                ("classifier_branch", 50, False)] # This will end with a callback
        if self.training_scheme == 'training_scheme_3':
            self.training_scheme = [("permutation_only", 1, True),  # This will end with a callback
                                ("classifier_branch", 1, False)]
        return self.training_scheme



    def get_losses(self):
        if self.rec_loss_name == 'MSE':
            self.rec_loss_fn = tf.keras.losses.mean_squared_error
        else:
            print(self.rec_loss_name + ' loss not supported for rec')

        if self.clas_loss_name == 'categorical_crossentropy':
            self.clas_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            print(self.clas_loss_name + ' loss not supported for classif')
        
        if self.dann_loss_name == 'categorical_crossentropy':
            self.dann_loss_fn = tf.keras.losses.categorical_crossentropy
        else:
            print(self.dann_loss_name + ' loss not supported for dann')
        return self.rec_loss_fn,self.clas_loss_fn,self.dann_loss_fn


    def print_status_bar(self, iteration, total, loss, metrics=None):
        metrics = ' - '.join(['{}: {:.4f}'.format(m.name, m.result())
                            for m in loss + (metrics or [])])

        end = "" if int(iteration) < int(total) else "\n"
    #     print(f"{iteration}/{total} - "+metrics ,end="\r")
    #     print(f"\r{iteration}/{total} - " + metrics, end=end)
        print("\r{}/{} - ".format(iteration,total) + metrics, end =end)

    def get_optimizer(self,learning_rate, weight_decay, optimizer_type, momentum=0.9):
        """
        This function takes a  learning rate, weight decay and optionally momentum and returns an optimizer object
        Args:
            learning_rate: The optimizer's learning rate
            weight_decay: The optimizer's weight decay
            optimizer_type: The optimizer's type [adam or sgd]
            momentum: The optimizer's momentum  
        Returns:
            an optimizer object
        """
        # TODO Add more optimizers
        print(optimizer_type)
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                        #  decay=weight_decay
                                        )
        elif optimizer_type == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate,
                                        weight_decay=weight_decay
                                        )
        elif optimizer_type == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    momentum=momentum 
                                                    )
        elif optimizer_type == 'adafactor':
            optimizer = tf.keras.optimizers.Adafactor(learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                            )
        else:
            optimizer = tf.keras.optimizers(learning_rate=learning_rate,
                                            weight_decay=weight_decay,
                                            momentum=momentum
                                            )
        return optimizer




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
    # parser.add_argument('--reduce_lr', type = , default = , help ='')
    # parser.add_argument('--early_stop', type = , default = , help ='')
    parser.add_argument('--batch_size', type = int, nargs='?', default = 128, help ='Training batch size') # Default identified with hp optimization
    # parser.add_argument('--verbose', type = , default = , help ='')
    # parser.add_argument('--threads', type = , default = , help ='')
    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = None, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--make_fake', type=str2bool, nargs='?',const=False, default=False, help ='')
    parser.add_argument('--true_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--false_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--pct_false', type = float,nargs='?', default = None, help ='')
    parser.add_argument('--clas_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default = 'categorical_crossentropy' , help ='Loss of the classification branch')
    parser.add_argument('--dann_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default ='categorical_crossentropy', help ='Loss of the DANN branch')
    parser.add_argument('--rec_loss_name', type = str,nargs='?', choices = ['MSE'], default ='MSE', help ='Reconstruction loss of the autoencoder')
    parser.add_argument('--weight_decay', type = float,nargs='?', default = 2e-6, help ='Weight decay applied by th optimizer') # Default identified with hp optimization
    parser.add_argument('--learning_rate', type = float,nargs='?', default = 0.001, help ='Starting learning rate for training')# Default identified with hp optimization
    parser.add_argument('--optimizer_type', type = str, nargs='?',choices = ['adam','adamw','rmsprop'], default = 'adam' , help ='Name of the optimizer to use')
    parser.add_argument('--clas_w', type = float,nargs='?', default = 0.1, help ='Weight of the classification loss')
    parser.add_argument('--dann_w', type = float,nargs='?', default = 0.1, help ='Weight of the DANN loss')
    parser.add_argument('--rec_w', type = float,nargs='?', default = 0.8, help ='Weight of the reconstruction loss')
    parser.add_argument('--warmup_epoch', type = float,nargs='?', default = 50, help ='Number of epoch to warmup DANN')
    parser.add_argument('--ae_hidden_size', type = int,nargs='+', default = [128,64,128], help ='Hidden sizes of the successive ae layers')
    parser.add_argument('--ae_hidden_dropout', type =float, nargs='?', default = None, help ='')
    parser.add_argument('--ae_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--ae_output_activation', type = str,nargs='?', default = 'linear', help ='')
    parser.add_argument('--ae_init', type = str,nargs='?', default = 'glorot_uniform', help ='')
    parser.add_argument('--ae_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--ae_l1_enc_coef', type = float,nargs='?', default = None, help ='')
    parser.add_argument('--ae_l2_enc_coef', type = float,nargs='?', default = None, help ='')
    parser.add_argument('--class_hidden_size', type = int,nargs='+', default = [64], help ='Hidden sizes of the successive classification layers')
    parser.add_argument('--class_hidden_dropout', type =float, nargs='?', default = None, help ='')
    parser.add_argument('--class_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--class_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--class_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--dann_hidden_size', type = int,nargs='?', default = [64], help ='')
    parser.add_argument('--dann_hidden_dropout', type =float, nargs='?', default = None, help ='')
    parser.add_argument('--dann_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--dann_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--dann_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--training_scheme', type = str,nargs='?', default = 'training_scheme_1', help ='')
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--hparam_path', type=str, nargs='?', default=None, help ='')

    run_file = parser.parse_args()
    working_dir = '/home/acollin/dca_permuted_workflow/'
    workflow = Workflow(run_file=run_file, working_dir=working_dir)
    print("Workflow loaded")
    opt_metric = workflow.make_experiment()

    print('yahou')
