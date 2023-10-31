import scanpy as sc
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy
from sklearn.model_selection import train_test_split


def scanpy_to_input(adata,keys, use_raw = False):
    '''
    Converts a scanpy object to a csv count matrix + an array for each metadata specified in *args 
    '''
    adata_to_dict = {}
    if use_raw :
        adata_to_dict['counts'] = adata.raw.X.copy()
    else :
        adata_to_dict['counts'] = adata.X.copy()
    if (type(adata_to_dict['counts']) == scipy.sparse.csr_matrix) or (type(adata_to_dict['counts']) == scipy.sparse.csc_matrix):
        adata_to_dict['counts'] = adata_to_dict['counts'].todense()
    for key in keys:
        adata_to_dict[key] = adata.obs[key]
    return adata_to_dict

def input_to_scanpy(count_matrix, obs_df, obs_names=None):
    '''
    Converts count matrix and metadata to a 

    count_matrix : matrix type data (viable input to sc.AnnData(X=...))
    obs_df : either a pd.DataFrame or a list of pd.Series in which case they will be concatenated
    obs_name : optional, row/index/obs name for the AnnData object
    '''
    if type(obs_df) != pd.DataFrame:
        obs_df = pd.concat(obs_df)
    ad = sc.AnnData(X = count_matrix, obs = obs_df)
    if obs_names:
        ad.obs_names = obs_names
    return ad

def get_optimizer(learning_rate, weight_decay, optimizer_type, momentum=0.9):
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
    if optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                     weight_decay=weight_decay
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
    else:
        optimizer = tf.keras.optimizers(learning_rate=learning_rate,
                                        weight_decay=weight_decay,
                                        momentum=momentum
                                        )
    return optimizer

def create_meta_stratify(adata, cats):
    obs = adata.obs
    meta_cats = '_'.join(cats)
    meta_col = pd.Series(obs[cats[0]].astype(str),index = obs.index)
    for h in cats[1:] :
        meta_col =  meta_col + '_' + obs[h].astype(str)
    obs[meta_cats] = meta_col
    adata.obs[meta_cats] = meta_col
    
    
def split_adata(adata, cats):
    '''
    Split adata in train and test in a stratified manner according to cats
    '''
    if type(cats) == str:
        meta_cats = cats
    else:
        create_meta_stratify(adata, cats)
        meta_cats = '_'.join(cats)
    train_idx, test_idx = train_test_split(np.arange(adata.n_obs),
                                       train_size=0.8,
                                       stratify =adata.obs[meta_cats],
                                       random_state=50)
    spl = pd.Series(['train'] * adata.n_obs, index = adata.obs.index)
    spl.iloc[test_idx] = 'test'
    adata.obs['TRAIN_TEST_split'] = spl.values
    adata_train = adata[adata.obs['TRAIN_TEST_split'] == 'train'].copy()
    adata_test = adata[adata.obs['TRAIN_TEST_split'] == 'test'].copy()
    return adata_train, adata_test