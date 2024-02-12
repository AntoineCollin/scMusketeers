import scanpy as sc
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split
import argparse

def densify(X):
    if (type(X) == scipy.sparse.csr_matrix) or (type(X) == scipy.sparse.csc_matrix):
        return np.asarray(X.todense())
    else :
        return np.asarray(X)

def check_raw(X,n):
    return X[:n,:n].todense()

def ann_subset(adata, obs_key, conditions):
    """
    Return a subset of the adata for cells with obs_key verifying conditions
    """
    if type(conditions) == str:
        conditions = [conditions]
    return adata[adata.obs[obs_key].isin(conditions),:].copy()

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

def default_value(var, val):
            """
            Returns var when val is None
            """
            if not var:
                return val
            else:
                return var
            
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def tuple_to_scalar(v):
    if isinstance(v, int) or isinstance(v, float):
        return v
    elif len(v) == 1: 
        return float(v[0])
    else :
        return [float(i) for i in v]


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