import anndata
import scanpy as sc
import pandas as pd
from sklearn.utils import shuffle

try :
    from sklearn.model_selection import train_test_split
except ImportError:
    pass
import numpy as np


class Dataset:
    def __init__(self, dataset_dir,dataset_name,class_key,filter_min_counts,normalize_size_factors,scale_input,logtrans_input, n_perm, semi_sup,unlabeled_category):
        self.dataset_name = dataset_name
        self.adata = anndata.AnnData()
        self.adata_TRAIN = anndata.AnnData()
        self.adata_train = anndata.AnnData()
        self.adata_val = anndata.AnnData()
        self.adata_test = anndata.AnnData()
        self.class_key = class_key
        self.filter_min_counts = filter_min_counts
        self.normalize_size_factors = normalize_size_factors
        self.scale_input = scale_input
        self.logtrans_input = logtrans_input
        self.n_perm = n_perm
        self.semi_sup = semi_sup
        self.unlabeled_category = unlabeled_category 
        self.mode=str()
        self.pct_split=float()
        self.obs_key=str()
        self.n_keep=int()
        self.keep_obs = str()
        self.train_test_random_seed=float()
        self.obs_subsample=[]
        self.true_celltype=str()
        self.false_celltype=str()
        self.pct_false=float()
        
        self.dataset_dir = dataset_dir
        self.dataset_names = {'htap':'htap_annotated',
                                'lca' : 'LCA_log1p',
                                'discovair':'discovair_V6',
                              'discovair_V7':'discovair_V7',
                              'discovair_V7_filtered':'discovair_V7_filtered_raw', # Filtered version with doublets, made lighter to pass through the model
                              'discovair_V7_filtered_no_D53':'discovair_V7_filtered_raw_no_D53',
                                'ajrccm':'HCA_Barbry_Grch38_Raw',
                                "disco_htap_ajrccm":'disco_htap_ajrccm_raw',
                                'htap_ajrccm': 'htap_ajrccm_raw',
                             'pbmc3k_processed':'pbmc_3k',
                             'htap_final':'htap_final',
                             'htap_final_ajrccm': 'htap_final_ajrccm'}
        self.dataset_path = self.dataset_dir + '/' + self.dataset_names[self.dataset_name] + '.h5ad'
        
        
    def load_dataset(self):
        self.adata = sc.read_h5ad(self.dataset_path)
        self.adata.obs[f'true_{self.class_key}'] = self.adata.obs[self.class_key]
#         if type(self.adata.X) != np.ndarray :
#             self.adata.X = self.adata.X.toarray()
        if not self.adata.raw:
            self.adata.raw = self.adata
        
        
    def normalize(self):
        if self.filter_min_counts:
            sc.pp.filter_genes(self.adata, min_counts=1)
            sc.pp.filter_cells(self.adata, min_counts=1)
        nonzero_genes, _ = sc.pp.filter_genes(self.adata.X, min_counts=1)
        assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA.'

        if self.normalize_size_factors or self.scale_input or self.logtrans_input:
            self.adata.raw = self.adata.copy()
        else:
            self.adata.raw = self.adata

        if self.normalize_size_factors:
            sc.pp.normalize_per_cell(self.adata)
            self.adata.obs['size_factors'] = self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
        else:
            self.adata.obs['size_factors'] = 1.0

        if self.logtrans_input:
            sc.pp.log1p(self.adata)

        if self.scale_input:
            sc.pp.scale(self.adata)
        
        self.adata_test = self.adata[self.adata.obs['TRAIN_TEST_split'] == 'test']
        self.adata_TRAIN = self.adata[self.adata.obs['TRAIN_TEST_split'] == 'train']
        print('right after loading')
        print(self.adata)
        print(self.adata_test)
        print(self.adata_TRAIN)
        print(self.adata_TRAIN.obs[self.class_key].value_counts())
        self.adata_train = self.adata_TRAIN.copy()

        
    
    def train_test_split(self):
        train_idx, val_idx = train_test_split(np.arange(self.adata_TRAIN.n_obs), train_size=self.train_size, random_state=self.train_test_random_seed)
        spl = pd.Series(['train'] * self.adata.n_obs, index = self.adata.obs.index)
        spl.iloc[val_idx] = 'val'
        self.adata.obs['train_split'] = spl.values
        self.adata_train = self.adata_TRAIN[self.adata_TRAIN.obs['train_split'] == 'train'].copy()
        self.adata_val = self.adata_TRAIN[self.adata_TRAIN.obs['train_split'] == 'val'].copy()
        
    
    def train_split(self,mode=None, pct_split=None, obs_key=None, n_keep=None, keep_obs = None,obs_subsample=None,train_test_random_seed=None):
        """
        Splits train and val datasets according to several modalities.
        percentage : Classic train test split
            pct_split : proportion (between 0 and 1) of the dataset to use as train
        entire_condition : Splits by keeping certain batches of obs_key in train and others in val 
            obs_key : in task2, one of adata.obs. Key to use to split the batch
            keep_obs : list of observations of obs_key to keep in train
        fixed_number : Symmetrical subsampling. We keep n_keep cells of each class of obs_key condition. Number of cells in training set will be equal to n_keep * adata.obs[obs_key].unique()
            obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
            n_keep : number of cells to keep for each class.
        Asymetrical_subsampling : We subsample one class while keeping the other ones intact. We keep n_keep cells for the obs_subsample class of obs_key.
            obs_key : in task3, one of adata.obs. Key to use to select subsample of cells
            obs_subsample : class of obs_key to subsample
            n_keep : number of cells to keep
        """
        self.mode = mode
        self.train_test_random_seed = train_test_random_seed
        if mode == 'percentage':
            self.pct_split = pct_split
            print(self.adata_TRAIN.obs[self.class_key].value_counts())
            train_idx, val_idx = train_test_split(np.arange(self.adata_TRAIN.n_obs), train_size=self.pct_split, stratify =self.adata_TRAIN.obs[self.class_key], random_state=self.train_test_random_seed)
            spl = pd.Series(['train'] * self.adata_TRAIN.n_obs, index = self.adata_TRAIN.obs.index)
            spl.iloc[val_idx] = 'val'
            print(len(spl))
            print(self.adata_TRAIN)
            print(self.adata_TRAIN.obs[self.class_key].value_counts())
            self.adata_TRAIN.obs['train_split'] = spl.values
        if mode == 'entire_condition':
            self.obs_key = obs_key
            self.keep_obs = keep_obs
            keep_idx = self.adata_TRAIN.obs[obs_key].isin(self.keep_obs)
            to_keep = pd.Series(['val'] * self.adata_TRAIN.n_obs, index = self.adata_TRAIN.obs.index)
            to_keep[keep_idx] = 'train'
            self.adata_TRAIN.obs['train_split'] = to_keep
        if mode == 'fixed_number':
            self.obs_key=obs_key
            self.n_keep=n_keep
            keep_idx=[]
            for obs_class in self.adata_TRAIN.obs[self.obs_key].unique():
                n_keep = min(self.adata_TRAIN.obs[self.class_key].value_counts()[obs_class], self.n_keep) # For celltypes with nb of cells < n_keep, we keep every cells
                keep_idx += list(self.adata_TRAIN[self.adata_TRAIN.obs[self.obs_key]==obs_class].obs.sample(n_keep, random_state=self.train_test_random_seed).index)
            to_keep = pd.Series(['val'] * self.adata_TRAIN.n_obs, index = self.adata_TRAIN.obs.index)
            to_keep[keep_idx] = 'train'
            self.adata_TRAIN.obs['train_split'] = to_keep
        if mode == 'Asymetrical_subsampling':
            self.obs_key=obs_key
            self.obs_subsample=obs_subsample
            self.n_keep=n_keep
            n_remove = self.adata_TRAIN[self.adata_TRAIN.obs[self.obs_key]==self.obs_subsample].n_obs - self.n_keep
            remove_idx = self.adata_TRAIN[self.adata_TRAIN.obs[self.obs_key]==self.obs_subsample].obs.sample(n_remove, random_state=self.train_test_random_seed).index
            to_keep = pd.Series(['train'] * self.adata_TRAIN.n_obs, index = self.adata_TRAIN.obs.index)
            to_keep[remove_idx] = 'val'
            self.adata_TRAIN.obs['train_split'] = to_keep
        # removing unnanotated cells
        if not self.semi_sup:
            to_keep = self.adata_TRAIN.obs['train_split']
            UNK_cells = self.adata_TRAIN.obs[self.class_key] == self.unlabeled_category
            to_keep[UNK_cells] = 'val'
            self.adata_TRAIN.obs['train_split'] = to_keep
            self.adata_train = self.adata_TRAIN[self.adata_TRAIN.obs['train_split'] == 'train'].copy()
            self.adata_val = self.adata_TRAIN[self.adata_TRAIN.obs['train_split'] == 'val'].copy()
            train_split = self.adata.obs['TRAIN_TEST_split'].astype('str')
            train_split[self.adata_TRAIN.obs_names] = self.adata_TRAIN.obs['train_split']
            self.adata.obs['train_split'] = train_split

            print(f'train split, train : {self.adata_train}')
            print(self.adata_train[self.class_key].value_counts())
        elif self.semi_sup:
            obs = self.adata_TRAIN.obs[self.class_key].astype('str')
            obs[self.adata_TRAIN.obs['train_split'] == 'val'] = self.unlabeled_category
            self.adata_TRAIN.obs[self.class_key] = obs
            self.adata_train = self.adata_TRAIN[self.adata_TRAIN.obs['train_split'] == 'train'].copy()
            self.adata_val = self.adata_TRAIN[self.adata_TRAIN.obs['train_split'] == 'val'].copy()
            print(f'train split, train : {self.adata_train}')
            print(self.adata_train[self.class_key].value_counts())
        

        
    def fake_annotation(self,true_celltype,false_celltype,pct_false, train_test_random_seed=None):
        '''
        Creates fake annotation by modifying true_celltype to false_celltype only in adata_train. Changes pct_false cells to the wrong label
            true_celltype : celltype to fake/rig from
            false_celltype : celltype to fake/rig to
            pct_false : percentage of true false_celltype to keep
        '''
        self.true_celltype = true_celltype
        self.false_celltype = false_celltype
        self.pct_false = pct_false
        true_series = self.adata_train.obs[self.class_key][self.adata_train.obs[self.class_key] == self.true_celltype]
        false_series = self.adata_train.obs[self.class_key][self.adata_train.obs[self.class_key] == self.false_celltype]
        n = len(true_series)
        true_series = shuffle(true_series, random_state = train_test_random_seed)
        false_idx = true_series[:round(n*self.pct_false)].index
        true_idx= true_series[round(n*self.pct_false):].index
        obs_series = self.adata_train.obs[self.class_key].astype('str')
        print(f'1 {obs_series.value_counts()}')
        print(f'2 {obs_series[false_idx]}')
        print(f'2.5 {false_idx}')
        print(f'2.75 {len(false_idx)}')
        obs_series[false_idx] = self.false_celltype
        print(f'3 {obs_series.value_counts()}')
        self.adata_train.obs[self.class_key] = obs_series
        adata_obs = self.adata.obs[self.class_key].astype('str')
        print(f'4 {adata_obs.value_counts()}')
        print(f'adata_obs.index : {adata_obs.index}')
        print(f'obs_series.index : {obs_series.index}')
        adata_obs.loc[obs_series.index] = obs_series # Les valeurs trafiquées
        print(f"il y a {len(true_series)} True ({self.true_celltype}) et {len(false_series)} False ({self.false_celltype})")
        print(f"on fake donc {len(false_idx)} cellules avec un pct_false de {self.pct_false}")
        print(f"adata_obs  (faked) : {adata_obs.value_counts()}") 
        print(f"true celltype : {self.adata.obs[f'true_{self.class_key}']}")
        self.adata.obs[f'fake_{self.class_key}'] = adata_obs
        self.adata.obs['faked'] = (self.adata.obs[f'fake_{self.class_key}'] != self.adata.obs[f'true_{self.class_key}'])
        self.adata.obs['faked_color'] = self.adata.obs['faked'].replace({True : 'faked', False : 'not faked'})
        print(self.adata.obs['faked'].value_counts())
        
        
    def small_clusters_totest(self):
        inf_n_cells = ~self.adata_train.obs[self.class_key].isin(self.adata_train.obs[self.class_key].value_counts()[self.adata_train.obs[self.class_key].value_counts()>self.n_perm].index) # Celltypes with too few cells in train are transferred to test
        self.adata_train = self.adata_train[~inf_n_cells].copy()
        inf_n_cells = inf_n_cells[inf_n_cells].index
        self.adata.obs.loc[inf_n_cells, 'train_split'] = 'test'
        self.adata_test = self.adata[self.adata.obs['train_split'] == 'test'].copy()

