import sys
import os
try :
    from .load import load_runfile
    from .dataset import Dataset, load_dataset
    from .predictor import MLP_Predictor
    from .model import DCA_Permuted,Scanvi,DCA_into_Perm, ScarchesScanvi_LCA
    from .utils import get_optimizer, scanpy_to_input, default_value, str2bool, densify


except ImportError:
    from load import load_runfile
    from dataset import Dataset, load_dataset
    from predictor import MLP_Predictor
    from model import DCA_Permuted,Scanvi
    from utils import get_optimizer, scanpy_to_input, default_value, str2bool, densify
# from dca.utils import str2bool,tuple_to_scalar
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dca.scPermut_subclassing import DANN_AE
from dca.permutation import batch_generator_training_permuted

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score,matthews_corrcoef
import time
import pickle
import anndata
import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import os
import sys
import keras
import gc
import tensorflow as tf
import neptune
from numba import cuda
from neptune.utils import stringify_unsupported
import subprocess

from ax.service.managed_loop import optimize
# from ax import RangeParameter, SearchSpace, ParameterType, FixedParameter, ChoiceParameter

import multiprocessing

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


workflow_ID = 'workflow_ID'

dataset = 'dataset'
dataset_name = 'dataset_name'
class_key = 'class_key'
batch_key = 'batch_key'

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

clas_loss_fn = 'clas_loss_fn'
dann_loss_fn = 'dann_loss_fn'
rec_loss_fn = 'rec_loss_fn'

weight_decay = 'weight_decay'
optimizer_type = 'optimizer_type'

clas_w = 'clas_w'
dann_w = 'dann_w'
rec_w = 'rec_w'
warmup_epoch = 'warmup_epoch'

ae_hidden_size = 'ae_hidden_size'
ae_hidden_dropout = 'ae_hidden_dropout'
ae_activation = 'ae_activation'
ae_output_activation = 'ae_output_activation'
ae_init = 'ae_init'
ae_batchnorm = 'ae_batchnorm'
ae_l1_enc_coef = 'ae_l1_enc_coef'
ae_l2_enc_coef = 'ae_l2_enc_coef'
num_classes = 'num_classes'
class_hidden_size = 'class_hidden_size'
class_hidden_dropout = 'class_hidden_dropout'
class_batchnorm = 'class_batchnorm'
class_activation = 'class_activation'
class_output_activation = 'class_output_activation'
num_batches = 'num_batches'
dann_hidden_size = 'dann_hidden_size'
dann_hidden_dropout = 'dann_hidden_dropout'
dann_batchnorm = 'dann_batchnorm'
dann_activation = 'dann_activation'
dann_output_activation = 'dann_output_activation'


class Workflow:
    def __init__(self, run_file, working_dir):
        '''
        run_file : a dictionary outputed by the function load_runfile
        '''
        self.run_file = run_file
        self.workflow_ID = self.run_file.workflow_id
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
        # model parameters
        # self.model_name = self.run_file[model_spec][model_name] # TODO : remove, obsolete in the case of DANN_AE
        # self.ae_type = self.run_file[model_spec][ae_type] # TODO : remove, obsolete in the case of DANN_AE
        # self.hidden_size = self.run_file[model_spec][hidden_size] # TODO : remove, obsolete in the case of DANN_AE
        # if type(self.hidden_size) == int: # TODO : remove, obsolete in the case of DANN_AE
        #     self.hidden_size = [2*self.hidden_size, self.hidden_size, 2*self.hidden_size] # TODO : remove, obsolete in the case of DANN_AE
        # self.hidden_dropout = self.run_file[model_spec][hidden_dropout] # TODO : remove, obsolete in the case of DANN_AE
        # if not self.hidden_dropout: # TODO : remove, obsolete in the case of DANN_AE
        #     self.hidden_dropout = 0
        # self.hidden_dropout = len(self.hidden_size) * [self.hidden_dropout] # TODO : remove, obsolete in the case of DANN_AE
        # self.batchnorm = self.run_file[model_spec][batchnorm] # TODO : remove, obsolete in the case of DANN_AE
        # self.activation = self.run_file[model_spec][activation] # TODO : remove, obsolete in the case of DANN_AE
        # self.init = self.run_file[model_spec][init] # TODO : remove, obsolete in the case of DANN_AE
        # self.batch_removal_weight = self.run_file[model_spec][batch_removal_weight] # TODO : remove, obsolete in the case of DANN_AE
        # model training parameters
        # self.epochs = self.run_file.epochs # TODO : remove, obsolete in the case of DANN_AE, we use training scheme
        # self.reduce_lr = self.run_file.reduce_lr # TODO : not implemented yet for DANN_AE
        # self.early_stop = self.run_file.early_stop # TODO : not implemented yet for DANN_AE
        self.batch_size = self.run_file.batch_size
        # self.optimizer = self.run_file.optimizer
        # self.verbose = self.run_file[model_training_spec][verbose] # TODO : not implemented yet for DANN_AE
        # self.threads = self.run_file[model_training_spec][threads] # TODO : not implemented yet for DANN_AE
        self.learning_rate = self.run_file.learning_rate
        self.n_perm = 1
        self.semi_sup = False # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
        self.unlabeled_category = 'UNK' # TODO : Not yet handled by DANN_AE, the case wwhere unlabeled cells are reconstructed as themselves
        # self.n_perm = self.run_file[model_training_spec][n_perm] # TODO : remove, n_perm is always 1
        # self.permute = self.run_file[model_training_spec][permute] # TODO : remove, obsolete in the case of DANN_AE, handled during training
        # self.change_perm = self.run_file[model_training_spec][change_perm] # TODO : remove, change_perm is always True
        # self.save_zinb_param = self.run_file[model_training_spec][save_zinb_param] # TODO : remove, obsolete in the case of DANN_AE
        # self.use_raw_as_output = self.run_file[model_training_spec][use_raw_as_output]
        # self.contrastive_margin =self.run_file[model_training_spec][contrastive_margin] # TODO : Not yet handled by DANN_AE, the case where we use constrastive loss
        # self.same_class_pct=self.run_file[model_training_spec][same_class_pct] # TODO : Not yet handled by DANN_AE, the case where we use constrastive loss

        # train test split # TODO : Simplify this, or at first only use the case where data is split according to batch
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
        # predictor parameters # TODO : remove, obsolete in the case of DANN_AE, predictor is directly on the model
        # self.predictor_model = self.run_file[predictor_spec][predictor_model]  # TODO : remove, obsolete in the case of DANN_AE
        # self.predict_key = self.run_file[predictor_spec][predict_key] # TODO : remove, obsolete in the case of DANN_AE
        # self.predictor_hidden_sizes = self.run_file[predictor_spec][predictor_hidden_sizes] # TODO : remove, obsolete in the case of DANN_AE
        # self.predictor_epochs = self.run_file[predictor_spec][predictor_epochs] # TODO : remove, obsolete in the case of DANN_AE
        # self.predictor_batch_size = self.run_file[predictor_spec][predictor_batch_size] # TODO : remove, obsolete in the case of DANN_AE
        # self.predictor_activation = self.run_file[predictor_spec][predictor_activation] # TODO : remove, obsolete in the case of DANN_AE

        # self.latent_space = anndata.AnnData() # TODO : probably unnecessary, should be handled by logging
        # self.corrected_count = anndata.AnnData() # TODO : probably unnecessary, should be handled by logging
        # self.scarches_combined_emb = anndata.AnnData() # TODO : probably unnecessary, should be handled by logging
        # self.DR_hist = dict() # TODO : probably unnecessary, should be handled by logging
        # self.DR_model = None # TODO : probably unnecessary, should be handled by logging

        # self.predicted_class = pd.Series() # TODO : probably unnecessary, should be handled by logging
        # self.pred_hist = dict() # TODO : probably unnecessary, should be handled by logging

        # Paths used for the analysis workflow, probably unnecessary
        self.working_dir = working_dir
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

        # self.run_log_dir = working_dir + '/logs/run' # TODO : probably unnecessary, should be handled by logging
        # self.run_log_path = self.run_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt' # TODO : probably unnecessary, should be handled by logging
        # self.predict_log_dir = working_dir + '/logs/predicts' # TODO : probably unnecessary, should be handled by logging
        # self.predict_log_path = self.predict_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt' # TODO : probably unnecessary, should be handled by logging
        # self.umap_log_dir = working_dir + '/logs/umap' # TODO : probably unnecessary, should be handled by logging
        # self.umap_log_path = self.umap_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt' # TODO : probably unnecessary, should be handled by logging
        # self.metrics_log_dir = working_dir + '/logs/metrics' # TODO : probably unnecessary, should be handled by logging
        # self.metrics_log_path = self.metrics_log_dir + f'/workflow_ID_{self.workflow_ID}_DONE.txt' # TODO : probably unnecessary, should be handled by logging

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

        self.metrics_list = {'balanced_acc' : balanced_accuracy_score, 'mcc' : matthews_corrcoef}

        self.metrics = []

        self.mean_loss_fn = keras.metrics.Mean(name='total loss') # This is a running average : it keeps the previous values in memory when it's called ie computes the previous and current values
        self.mean_clas_loss_fn = keras.metrics.Mean(name='classification loss')
        self.mean_dann_loss_fn = keras.metrics.Mean(name='dann loss')
        self.mean_rec_loss_fn = keras.metrics.Mean(name='reconstruction loss')

        self.training_scheme = self.run_file.training_scheme

        self.log_neptune = self.run_file.log_neptune
        self.run = None

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

    def make_experiment(self, params):
        print(params)
#        self.use_hvg = params['use_hvg']
        self.clas_w =  params['clas_w']
        self.dann_w = params['dann_w']
        self.rec_w =  1
        self.weight_decay =  params['weight_decay']
        self.warmup_epoch =  params['warmup_epoch']
        self.dropout =  params['dropout']
        self.layer1 = params['layer1']
        self.layer2 =  params['layer2']
        self.bottleneck = params['bottleneck']

        self.ae_hidden_size = [self.layer1, self.layer2, self.bottleneck, self.layer2, self.layer1]

        self.dann_hidden_dropout, self.class_hidden_dropout, self.ae_hidden_dropout = self.dropout, self.dropout, self.dropout

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
                               n_perm = self.n_perm,
                               semi_sup = self.semi_sup,
                               unlabeled_category = self.unlabeled_category)
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
        
        sf_list = {'full': self.dataset.sf,
                      'train': self.dataset.sf_train,
                      'val': self.dataset.sf_val,
                      'test': self.dataset.sf_test}

        self.num_classes = len(np.unique(self.dataset.y_train))
        self.num_batches = len(np.unique(self.dataset.batch))

        bottleneck_size = int(self.ae_hidden_size[int(len(self.ae_hidden_size)/2)])

        self.class_hidden_size = default_value(self.class_hidden_size , (bottleneck_size + self.num_classes)/2) # default value [(bottleneck_size + num_classes)/2]
        self.dann_hidden_size = default_value(self.dann_hidden_size , (bottleneck_size + self.num_batches)/2) # default value [(bottleneck_size + num_batches)/2]

        if self.log_neptune :
            self.run = neptune.init_run(
                    project="blaireaufurtif/scPermut",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
)
            for par,val in self.run_file.__dict__.items():
                self.run[f"parameters/{par}"] = stringify_unsupported(val)

            for par,val in params.items():
                self.run[f"parameters/{par}"] = stringify_unsupported(val)
            self.run[f'parameters/ae_hidden_size'] = stringify_unsupported(self.ae_hidden_size)

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

        self.optimizer = get_optimizer(self.learning_rate, self.weight_decay, self.optimizer_type)
        self.rec_loss_fn, self.clas_loss_fn, self.dann_loss_fn = self.get_losses() # redundant
        self.training_scheme = self.get_scheme()

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

        if self.log_neptune:
            neptune_run_id = self.run['sys/id'].fetch()
            for group in ['full', 'train', 'val', 'test']:
                input_tensor = {k:tf.convert_to_tensor(v) for k,v in scanpy_to_input(adata_list[group],['size_factors']).items()}
                enc, clas, dann, rec = self.dann_ae(input_tensor, training=False).values()                
                clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
                for metric in self.metrics_list: # only classification metrics ATM
                    self.run[f"evaluation/{group}/{metric}"] = self.metrics_list[metric](np.asarray(y_list[group].argmax(axis=1)), clas.argmax(axis=1))
                if group == 'full':
                    save_dir = self.working_dir + 'experiment_script/results/' + str(neptune_run_id) + '/'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    y_pred = pd.DataFrame(self.dataset.ohe_celltype.inverse_transform(clas), index = adata_list[group].obs_names)
                    np.save(save_dir + f'latent_space_{group}.npy', enc.numpy())
                    y_pred.to_csv(save_dir + f'predictions_{group}.csv')
                    self.run[f'evaluation/{group}/latent_space'].track_files(save_dir + f'latent_space_{group}.npy')
                    self.run[f'evaluation/{group}/predictions'].track_files(save_dir + f'predictions_{group}.csv')
                    
                    pred_adata = sc.AnnData(X = adata_list[group].X, obs = adata_list[group].obs, var = adata_list[group].var)
                    pred_adata.obs[f'{class_key}_pred'] = y_pred
                    pred_adata.obsm['latent_space'] = enc.numpy()
                    sc.pp.neighbors(pred_adata, use_rep = 'latent_space')
                    sc.tl.umap(pred_adata)
                    np.save(save_dir + f'umap_{group}.npy', pred_adata.obsm['X_umap'])
                    self.run[f'evaluation/{group}/umap'].track_files(save_dir + f'umap_{group}.npy')
                    sc.set_figure_params(figsize=(15, 10), dpi = 300)
                    fig_class = sc.pl.umap(pred_adata, color = f'true_{self.class_key}', size = 5,return_fig = True)
                    fig_batch = sc.pl.umap(pred_adata, color = self.batch_key, size = 5,return_fig = True)
                    fig_split = sc.pl.umap(pred_adata, color = 'train_split', size = 5,return_fig = True)
                    self.run[f'evaluation/{group}/classif_umap'].upload(fig_class)
                    self.run[f'evaluation/{group}/batch_umap'].upload(fig_batch)
                    self.run[f'evaluation/{group}/split_umap'].upload(fig_split)

        inp = scanpy_to_input(adata_list['val'],['size_factors'])
        inp = {k:tf.convert_to_tensor(v) for k,v in inp.items()}
        _, clas, dann, rec = self.dann_ae(inp, training=False).values()
        clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
        opt_metric = self.metrics_list['mcc'](np.asarray(y_list['val'].argmax(axis=1)), clas.argmax(axis=1)) # We retrieve the last metric of interest
        if self.log_neptune:
            self.run.stop()
        gc.collect()
        tf.keras.backend.clear_session()
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
            for m in self.metrics_list:
                history[group][m] = []

        # if self.log_neptune:
        #     for group in history:
        #         for par,val in history[group].items():
        #             self.run[f"training/{group}/{par}"] = []
        i = 0

        total_epochs = np.sum([n_epochs for _, n_epochs, _ in training_scheme])
        running_epoch = 0

        for (strategy, n_epochs, use_perm) in training_scheme:
            optimizer = get_optimizer(self.learning_rate, self.weight_decay, self.optimizer_type) # resetting optimizer state when switching strategy
            if verbose :
                print(f"Step number {i}, running {strategy} strategy with permuation = {use_perm} for {n_epochs} epochs")
                time_in = time.time()

                # Early stopping for those strategies only
            if strategy in  ['full_model', 'classifier_branch', 'permutation_only']:
                wait = 0
                best_epoch = 0
                es_best = np.inf # initialize early_stopping
                patience = 20
                if strategy == 'permutation_only':
                    monitored = 'rec_loss'
                else:
                    monitored = 'mcc'


            for epoch in range(1, n_epochs+1):
                running_epoch +=1
                print(f"Epoch {running_epoch}/{total_epochs}, Current strat Epoch {epoch}/{n_epochs}")
                history, bot, cl, d ,re = self.training_loop(history=history,
                                                             training_strategy=strategy,
                                                             use_perm=use_perm,
                                                             optimizer=optimizer,
                                                               **loop_params)

                if self.log_neptune:
                    for group in history:
                        for par,value in history[group].items():
                            self.run[f"training/{group}/{par}"].append(value[-1])

                if strategy in ['full_model', 'classifier_branch', 'permutation_only']:
                    # Early stopping
                    wait += 1
                    monitored_value = history['val'][monitored][-1]

                    if monitored_value < es_best:
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

        # dann_ae.set_weights(best_weights)
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

        batch_generator = batch_generator_training_permuted(X = X_list[group],
                                                            y = y_list[group],
                                                            batch_ID = batch_list[group],
                                                            sf = adata_list[group].obs['size_factors'],                    
                                                            ret_input_only=False,
                                                            batch_size=self.batch_size,
                                                            n_perm=1, 
                                                            use_perm=use_perm)
        n_obs = adata_list[group].n_obs
        steps = n_obs // self.batch_size + 1
        n_steps = steps
        n_samples = 0

        self.mean_loss_fn.reset_state()
        self.mean_clas_loss_fn.reset_state()
        self.mean_dann_loss_fn.reset_state()
        self.mean_rec_loss_fn.reset_state()

        for step in range(1, n_steps + 1):
            input_batch, output_batch = next(batch_generator)
            X_batch, sf_batch = input_batch.values()
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

            self.mean_loss_fn(loss)
            self.mean_clas_loss_fn(clas_loss)
            self.mean_dann_loss_fn(dann_loss)
            self.mean_rec_loss_fn(rec_loss)

            if verbose :
                self.print_status_bar(n_samples, n_obs, [self.mean_loss_fn, self.mean_clas_loss_fn, self.mean_dann_loss_fn, self.mean_rec_loss_fn], self.metrics)
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
            inp = {'counts':X_list[group], 'size_factors':sf_list[group]}
            # inp = {k:tf.convert_to_tensor(v) for k,v in inp.items()}
            try :
                _, clas, dann, rec = ae(inp, training=False).values()
            except:
                with tf.device('CPU'):
                    _, clas, dann, rec = ae(inp, training=False).values()

    #         return _, clas, dann, rec
            clas_loss = tf.reduce_mean(clas_loss_fn(y_list[group], clas)).numpy()
            history[group]['clas_loss'] += [clas_loss]
            dann_loss = tf.reduce_mean(dann_loss_fn(batch_list[group], dann)).numpy()
            history[group]['dann_loss'] += [dann_loss]
            with tf.device('CPU'): # Otherwise, risks of memory allocation errors
                rec_loss = tf.reduce_mean(rec_loss_fn(densify(X_list[group]), rec)).numpy()
            history[group]['rec_loss'] += [rec_loss]
            history[group]['total_loss'] += [self.clas_w * clas_loss + self.dann_w * dann_loss + self.rec_w * rec_loss + np.sum(ae.losses)] # using numpy to prevent memory leaks
            # history[group]['total_loss'] += [tf.add_n([self.clas_w * clas_loss] + [self.dann_w * dann_loss] + [self.rec_w * rec_loss] + ae.losses).numpy()]

            clas = np.eye(clas.shape[1])[np.argmax(clas, axis=1)]
            for metric in self.metrics_list: # only classification metrics ATM
                history[group][metric] += [self.metrics_list[metric](np.asarray(y_list[group].argmax(axis=1)), clas.argmax(axis=1))] # y_list are onehot encoded
        del inp
        return history, _, clas, dann, rec

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
            self.training_scheme = [("permutation_only", 100, True),  # This will end with a callback
                                ("classifier_branch", 50, False)]
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


class MakeExperiment:
    def __init__(self, run_file, working_dir):
        # super()._init_()
        self.run_file = run_file
        self.working_dir = working_dir
        self.workflow = None

    def train(self, params):
        reset_keras()
        # cuda.select_device(0)
        self.workflow = Workflow(run_file=self.run_file, working_dir=self.working_dir)
        mcc = self.workflow.make_experiment(params)
        del self.workflow  # Should not be necessary
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        return mcc
 
    # def train_process(self, params, q):
    #     q.put(self.train(params))

    # def train_run(self,params):
    #     q = multiprocessing.Queue()
    #     p = multiprocessing.Process(target=self.train_process, args=(params, q))
    #     p.start()
    #     mcc = q.get()
    #     return mcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument('--dataset_name', type = str, default = 'disco_ajrccm_downsampled', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--class_key', type = str, default = 'celltype_lv2_V3', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'manip', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')# TODO :remove, we always want to do that
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')
    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=10000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")
    # parser.add_argument('--reduce_lr', type = , default = , help ='')
    # parser.add_argument('--early_stop', type = , default = , help ='')
    parser.add_argument('--batch_size', type = int, nargs='?', default = 256, help = 'Training batch size')
    # parser.add_argument('--verbose', type = , default = , help ='')
    # parser.add_argument('--threads', type = , default = , help ='')
    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = 0, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--make_fake', type=str2bool, nargs='?',const=False, default=False, help ='')
    parser.add_argument('--true_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--false_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--pct_false', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--clas_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default = 'categorical_crossentropy' , help ='Loss of the classification branch')
    parser.add_argument('--dann_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default ='categorical_crossentropy', help ='Loss of the DANN branch')
    parser.add_argument('--rec_loss_name', type = str,nargs='?', choices = ['MSE'], default ='MSE', help ='Reconstruction loss of the autoencoder')
    parser.add_argument('--weight_decay', type = float,nargs='?', default = 1e-4, help ='Weight decay applied by th optimizer')
    parser.add_argument('--learning_rate', type = float,nargs='?', default = 0.001, help ='Starting learning rate for training')
    parser.add_argument('--optimizer_type', type = str, nargs='?',choices = ['adam','adamw','rmsprop'], default = 'adam' , help ='Name of the optimizer to use')
    parser.add_argument('--clas_w', type = float,nargs='?', default = 0.1, help ='Wight of the classification loss')
    parser.add_argument('--dann_w', type = float,nargs='?', default = 0.1, help ='Wight of the DANN loss')
    parser.add_argument('--rec_w', type = float,nargs='?', default = 0.8, help ='Wight of the reconstruction loss')
    parser.add_argument('--warmup_epoch', type = float,nargs='?', default = 0.8, help ='Wight of the reconstruction loss')
    parser.add_argument('--ae_hidden_size', type = int,nargs='+', default = [128,64,128], help ='Hidden sizes of the successive ae layers')
    parser.add_argument('--ae_hidden_dropout', type =float, nargs='?', default = 0, help ='')
    parser.add_argument('--ae_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--ae_output_activation', type = str,nargs='?', default = 'linear', help ='')
    parser.add_argument('--ae_init', type = str,nargs='?', default = 'glorot_uniform', help ='')
    parser.add_argument('--ae_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--ae_l1_enc_coef', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--ae_l2_enc_coef', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--class_hidden_size', type = int,nargs='+', default = [64], help ='Hidden sizes of the successive classification layers')
    parser.add_argument('--class_hidden_dropout', type =float, nargs='?', default = 0, help ='')
    parser.add_argument('--class_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--class_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--class_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--dann_hidden_size', type = int,nargs='?', default = [64], help ='')
    parser.add_argument('--dann_hidden_dropout', type =float, nargs='?', default = 0, help ='')
    parser.add_argument('--dann_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--dann_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--dann_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--training_scheme', type = str,nargs='?', default = 'training_scheme_1', help ='')
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--workflow_id', type=str, nargs='?', default='default', help ='')
    # parser.add_argument('--epochs', type=int, nargs='?', default=100, help ='')

    run_file = parser.parse_args()
    working_dir = '/home/acollin/dca_permuted_workflow/'
    experiment = MakeExperiment(run_file=run_file, working_dir=working_dir)
    # workflow = Workflow(run_file=run_file, working_dir=working_dir)
    print("Workflow loaded")

    hparams = [
        #{"name": "use_hvg", "type": "range", "bounds": [5000, 10000], "log_scale": False},
        {"name": "clas_w", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
        {"name": "dann_w", "type": "range", "bounds": [1e-4, 1e2], "log_scale": False},
        {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-4], "log_scale": True},
        {"name": "warmup_epoch", "type": "range", "bounds": [1, 50]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "bottleneck", "type": "range", "bounds": [32, 64]},
        {"name": "layer2", "type": "range", "bounds": [64, 512]},
        {"name": "layer1", "type": "range", "bounds": [512, 2048]},

    ]

    # workflow.make_experiment(hparams)

    # def train(params):
    #     print(params)
    #     run_file.clas_w =  params['clas_w']
    #     run_file.dann_w = params['dann_w']
    #     run_file.rec_w =  1
    #     run_file.learning_rate = params['learning_rate']
    #     run_file.weight_decay =  params['weight_decay']
    #     run_file.warmup_epoch =  params['warmup_epoch']
    #     dropout =  params['dropout']
    #     layer1 = params['layer1']
    #     layer2 =  params['layer2']
    #     bottleneck = params['bottleneck']

    #     run_file.ae_hidden_size = [layer1, layer2, bottleneck, layer2, layer1]

    #     run_file.dann_hidden_dropout, run_file.class_hidden_dropout, run_file.ae_hidden_dropout = dropout, dropout, dropout
        
    #     cmd = ['sbatch', '--wait', '/home/acollin/dca_permuted_workflow/workflow/run_workflow_cmd.sh']
    #     for k, v in run_file.__dict__.items():
    #         cmd += ([f'--{k}'])
    #         if type(v) == list:
    #             cmd += ([str(i) for i in v])
    #         else :
    #             cmd += ([str(v)])
    #     print(cmd)
    #     subprocess.Popen(cmd).wait()
    #     working_dir = '/home/acollin/dca_permuted_workflow/'
    #     with open(working_dir + 'mcc_res.txt', 'r') as my_file:
    #         mcc = float(my_file.read())
    #     os.remove(working_dir + 'mcc_res.txt')
    #     return mcc

    best_parameters, values, experiment, model = optimize(
        parameters=hparams,
        evaluation_function=experiment.train,
        objective_name='mcc',
        minimize=False,
        total_trials=30,
        random_seed=40,
    )

    