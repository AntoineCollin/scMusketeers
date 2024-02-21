# Copyright 2017 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle, os, numbers

import numpy as np
import scipy as sp
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


#TODO: Fix this
class AnnSequence:
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            self.size_factors = np.ones((self.matrix.shape[0], 1),
                                        dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        batch = self.matrix[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_sf = self.size_factors[idx*self.batch_size:(idx+1)*self.batch_size]

        # return an (X, Y) pair
        return {'count': batch, 'size_factors': batch_sf}, batch


def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata, first_column_names=True)
    else:
        raise NotImplementedError

    # check if observations are unnormalized using first 10
    X_subset = adata.X[:10]
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    if sp.sparse.issparse(X_subset):
        assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error
    else:
        assert np.all(X_subset.astype(int) == X_subset), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'

    adata.obs['dca_split'] = adata.obs['dca_split'].astype('category')
    print('dca: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('dca: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))

def train_split(adata, task=None, pct_split=None, obs_key=None, n_keep=None, keep_obs = None, random_seed=None, obs_subsample=None):
    """
    Splits train and test datasets according to several modalities.
    task 1 : Classic train test split
        pct_split : proportion (between 0 and 1) of the dataset to use as train
    task 2 : Splits by keeping certain batches of obs_key in train and others in test 
        obs_key : in task2, one of adata.obs. Key to use to split the batch
        keep_obs : list of observations of obs_key to keep in train
    task 3 : Symmetrical subsampling. We keep n_keep cells of each class of obs_key condition. Number of cells in training set will be equal to n_keep * adata.obs[obs_key].unique()
        obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
        n_keep : number of cells to keep for each class.
    task 4 : Asymetrical subsampling. We subsample one class while keeping the other ones intact. We keep n_keep cells for the obs_subsample class of obs_key.
        obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
        obs_subsample : class of obs_key to subsample
        n_keep : number of cells to keep
    """
    if task == 1:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), train_size=pct_split, random_state=random_seed)
        spl = pd.Series(['train'] * adata.n_obs, index = adata.obs.index)
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    if task == 2:
        keep_idx = adata.obs[obs_key].isin(keep_obs)
        to_keep = pd.Series(['test'] * adata.n_obs, index = adata.obs.index)
        to_keep[keep_idx] = 'train'
        adata.obs['dca_split'] = to_keep
    if task == 3:
        keep_idx=[]
        for obs_class in adata.obs[obs_key].unique():
            n_keep = min(adata_train.obs[class_key].value_counts()[obs_class], n_keep) # For celltypes with nb of cells < n_keep, we keep every cells
            keep_idx += list(adata[adata.obs[obs_key]==obs_class].obs.sample(n_keep, random_state=random_seed).index)
        to_keep = pd.Series(['test'] * adata.n_obs, index = adata.obs.index)
        to_keep[keep_idx] = 'train'
        adata.obs['dca_split'] = to_keep
    if task == 4:
        n_remove = adata[adata.obs[obs_key]==obs_subsample].n_obs - n_keep
        remove_idx = adata[adata.obs[obs_key]==obs_subsample].obs.sample(n_remove, random_state=random_seed).index
        to_keep = pd.Series(['train'] * adata.n_obs, index = adata.obs.index)
        to_keep[remove_idx] = 'test'
        adata.obs['dca_split'] = to_keep
    return adata

def make_fake_obs(series, pct_true, pct_false):
    """
    Partitioning function for creating wrong index to test inacurate annotation. If pct_true and pct_false (between 0 & 1) 
    don't add up to 1, the rest of the cells are marked as UNK and are treated by the unsupervised pipeline.
    """
    n = len(series)
    np.random.shuffle(series)
    true_idx = series[:round(n*pct_true)].index
    false_idx= series[round(n*pct_true): round(n*pct_true) + round(n*pct_false)].index
    UNK_idx = series[round(n*pct_true) + round(n*pct_false):].index
    return true_idx, false_idx, UNK_idx
    

def make_unk(adata_train, class_key, task=None, pct_split=None, obs_key=None, n_keep=None, keep_obs = None, random_seed=None, obs_subsample=None, true_celltype=None, false_celltype=None, pct_true=None, pct_false=None):
    """
    Removes labels of some observations (replacing them with the 'UNK' tag) following patterns similar to train_split.
    task 1 : Classic train test split
        pct_split : proportion (between 0 and 1) of the dataset to use as train
    task 2 : Splits by keeping certain batches of obs_key in train and others in test
        obs_key : in task2, one of adata.obs. Key to use to split the batch
        keep_obs : list of observations of obs_key to keep in train
    task 3 : Symmetrical subsampling. We keep n_keep cells of each class of obs_key condition. Number of cells in training set will be equal to n_keep * adata.obs[obs_key].unique()
        obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
        n_keep : number of cells to keep for each class.
    task 5 : Creates fake annotation by modifying true_celltype to false_celltype. Keeps pct_true true_celltype, pct_false false_celltype and transforms the 
    rest to UNK if pct_false + pct_true !=1
        class_key : default to 'celltype' in this case
        true_celltype : celltype to fake/rig from
        false_celltype : celltype to fake/rig to
        pct_true : percentage of true true_celltype to keep
        pct_false : percentage of true false_celltype to keep
    """
    if task == 1:
        keep_idx, UNK_idx = train_test_split(np.arange(adata_train.n_obs), train_size=pct_split, random_state=random_seed)
        obs_series = adata_train.obs[class_key].astype('str')
        obs_series[UNK_idx] = 'UNK'
        adata_train.obs[class_key] = obs_series
    if task == 2:
        UNK_idx = ~adata_train.obs[obs_key].isin(keep_obs)
        obs_series = adata_train.obs[class_key].astype('str')
        obs_series[UNK_idx] = 'UNK'
        adata_train.obs[class_key] = obs_series
    if task == 3:
        keep_idx=[]
        for obs_class in adata_train.obs[class_key].unique():
            n_keep = min(adata_train.obs[class_key].value_counts()[obs_class], n_keep) # For celltypes with nb of cells < n_keep, we keep every cells
            keep_idx += list(adata_train[adata_train.obs[class_key]==obs_class].obs.sample(n_keep, random_state=random_seed).index)
        to_keep = pd.Series(['UNK'] * adata_train.n_obs, index = adata_train.obs.index)
        to_keep[keep_idx] = adata_train.obs[class_key][keep_idx].astype('str')
        adata_train.obs[class_key] = to_keep
    if task == 5:
        true_series = adata_train.obs[class_key][adata_train.obs[class_key] == true_celltype]
        true_idx, false_idx, UNK_idx = make_fake_obs(true_series, pct_true, pct_false)
        obs_series = adata_train.obs[class_key].astype('str')
        obs_series[UNK_idx] = 'UNK'
        obs_series[false_idx] = false_celltype
        adata_train.obs[class_key] = obs_series
    return adata_train