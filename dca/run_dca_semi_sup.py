import os, tempfile, shutil, random
import anndata
import numpy as np
import scanpy as sc
import yaml
import pickle

try:
    import tensorflow as tf
except ImportError:
    raise ImportError('DCA requires tensorflow. Please follow instructions'
                      ' at https://www.tensorflow.org/install/ to install'
                      ' it.')
    
from .io import read_dataset, normalize, train_split, make_unk
from .train import train
from .network import AE_types
from .paths import results_save_path,dataset_path

WD_PATH = "/home/acollin/dca_permuted"

def dca_permuted(adata,
                 dataset,
        mode='denoise',
        ae_type='zinb-conddisp',
        normalize_per_cell=True,
        scale=True,
        log1p=True,
        hidden_size=(64, 32, 64), # network args
        hidden_dropout=0,
        batchnorm=True,
        activation='relu',
        init='glorot_uniform',
        network_kwds={},
        epochs=10,               # training args
        reduce_lr=10,
        early_stop=5,
        batch_size=32,
        optimizer='RMSprop',
        learning_rate=None,
        random_state=0,
        threads=None,
        verbose=True,
        training_kwds={},
        return_model=False,
        return_info=False,
        copy=False,
        # Here starts dca permuted
        permute=True,
        experiment=1,
        save_all = True,
        class_key='celltype',
        split_kwds = None,
        train_spliting = True,
        semi_sup = False
        ):
    """Deep count autoencoder(DCA) API.

    Fits a count autoencoder to the count data given in the anndata object
    in order to denoise the data and capture hidden representation of
    cells in low dimensions. Type of the autoencoder and return values are
    determined by the parameters.

    Parameters
    ----------
    adata : :class:`~scanpy.api.AnnData`
        An anndata file with `.raw` attribute representing raw counts.
    mode : `str`, optional. `denoise`(default), or `latent`.
        `denoise` overwrites `adata.X` with denoised expression values.
        In `latent` mode DCA adds `adata.obsm['X_dca']` to given adata
        object. This matrix represent latent representation of cells via DCA.
    ae_type : `str`, optional. `zinb-conddisp`(default), `zinb`, `nb-conddisp` or `nb`.
        Type of the autoencoder. Return values and the architecture is
        determined by the type e.g. `nb` does not provide dropout
        probabilities.
    normalize_per_cell : `bool`, optional. Default: `True`.
        If true, library size normalization is performed using
        the `sc.pp.normalize_per_cell` function in Scanpy and saved into adata
        object. Mean layer is re-introduces library size differences by
        scaling the mean value of each cell in the output layer. See the
        manuscript for more details.
    scale : `bool`, optional. Default: `True`.
        If true, the input of the autoencoder is centered using
        `sc.pp.scale` function of Scanpy. Note that the output is kept as raw
        counts as loss functions are designed for the count data.
    log1p : `bool`, optional. Default: `True`.
        If true, the input of the autoencoder is log transformed with a
        pseudocount of one using `sc.pp.log1p` function of Scanpy.
    hidden_size : `tuple` or `list`, optional. Default: (64, 32, 64).
        Width of hidden layers.
    hidden_dropout : `float`, `tuple` or `list`, optional. Default: 0.0.
        Probability of weight dropout in the autoencoder (per layer if list
        or tuple).
    batchnorm : `bool`, optional. Default: `True`.
        If true, batch normalization is performed.
    activation : `str`, optional. Default: `relu`.
        Activation function of hidden layers.
    init : `str`, optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.
    network_kwds : `dict`, optional.
        Additional keyword arguments for the autoencoder.
    epochs : `int`, optional. Default: 300.
        Number of total epochs in training.
    reduce_lr : `int`, optional. Default: 10.
        Reduces learning rate if validation loss does not improve in given number of epochs.
    early_stop : `int`, optional. Default: 15.
        Stops training if validation loss does not improve in given number of epochs.
    batch_size : `int`, optional. Default: 32.
        Number of samples in the batch used for SGD.
    learning_rate : `float`, optional. Default: None.
        Learning rate to use in the training.
    optimizer : `str`, optional. Default: "RMSprop".
        Type of optimization method used for training.
    random_state : `int`, optional. Default: 0.
        Seed for python, numpy and tensorflow.
    threads : `int` or None, optional. Default: None
        Number of threads to use in training. All cores are used by default.
    verbose : `bool`, optional. Default: `False`.
        If true, prints additional information about training and architecture.
    training_kwds : `dict`, optional.
        Additional keyword arguments for the training process.
    return_model : `bool`, optional. Default: `False`.
        If true, trained autoencoder object is returned. See "Returns".
    return_info : `bool`, optional. Default: `False`.
        If true, all additional parameters of DCA are stored in `adata.obsm` such as dropout
        probabilities (obsm['X_dca_dropout']) and estimated dispersion values
        (obsm['X_dca_dispersion']), in case that autoencoder is of type
        zinb or zinb-conddisp.
    copy : `bool`, optional. Default: `False`.
        If true, a copy of anndata is returned.

    Returns
    -------
    If `copy` is true and `return_model` is false, AnnData object is returned.

    In "denoise" mode, `adata.X` is overwritten with the denoised values. In "latent" mode, latent
    low dimensional representation of cells are stored in `adata.obsm['X_dca']` and `adata.X`
    is not modified. Note that these values are not corrected for library size effects.

    If `return_info` is true, all estimated distribution parameters are stored in AnnData such as:

    - `.obsm["X_dca_dropout"]` which is the mixture coefficient (pi) of the zero component
    in ZINB, i.e. dropout probability. (Only if ae_type is zinb or zinb-conddisp)

    - `.obsm["X_dca_dispersion"]` which is the dispersion parameter of NB.

    - `.uns["dca_loss_history"]` which stores the loss history of the training.

    Finally, the raw counts are stored as `.raw`.

    If `return_model` is given, trained model is returned. When both `copy` and `return_model`
    are true, a tuple of anndata and model is returned in that order.
    """
    print("entering_dca")
    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('denoise', 'latent'), '%s is not a valid mode.' % mode

    # set seed for reproducibility
    random.seed(random_state)
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    os.environ['PYTHONHASHSEED'] = '0'
    
    # this creates adata.raw with raw counts and copies adata if copy==True
    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=copy)

    # check for zero genes
    nonzero_genes, _ = sc.pp.filter_genes(adata.X, min_counts=1)
    assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA.'

    adata = normalize(adata,
                      filter_min_counts=False, # no filtering, keep cell and gene idxs same
                      size_factors=normalize_per_cell,
                      normalize_input=scale,
                      logtrans_input=log1p)
    
    network_kwds = {**network_kwds,
        'hidden_size': hidden_size,
        'hidden_dropout': hidden_dropout,
        'batchnorm': batchnorm,
        'activation': activation,
        'init': init
    }

    input_size = output_size = adata.n_vars
    net = AE_types[ae_type](input_size=input_size,
                            output_size=output_size,
                            **network_kwds)
    #net.save()
    net.build()
    
    if not split_kwds:
        split_kwds = {'task':0,
                      'pct_split':None,
                      'obs_key':None,
                      'n_keep':None,
                      #keep_obs=None, 
                      'random_seed':None,
                      'obs_subsample':None,
                      'true_celltype':None, 
                      'false_celltype':None, 
                      'pct_true':None,
                      'pct_false':None}
        
    training_kwds = {**training_kwds,
        'epochs': epochs,
        'reduce_lr': reduce_lr,
        'early_stop': early_stop,
        'batch_size': batch_size,
        'optimizer': optimizer,
        'verbose': verbose,
        'threads': threads,
        'learning_rate': learning_rate,
        'permute': permute,
        'class_key': class_key,
        'n_perm': 5
    }
    
    # train_test_split
    if train_split:
        if semi_sup: # If running semi-supervised mode, the dataset is split in 0.8 train and 0.2 test
            adata = train_split(adata, **{'task':1,
                                          'pct_split':0.8,
                                          'obs_key':None,
                                          'n_keep':None,
                                          #keep_obs=None,
                                          'random_seed':42,
                                          'obs_subsample':None})
        else :
            adata = train_split(adata, **split_kwds)
        adata_train = adata[adata.obs.dca_split == 'train'].copy()
        inf_5_cells = ~adata_train.obs[class_key].isin(adata_train.obs[class_key].value_counts()[adata_train.obs[class_key].value_counts()>training_kwds['n_perm']].index) # Celltypes with too few cells in train are transferred to test
        inf_5_cells = inf_5_cells[inf_5_cells].index
        print(inf_5_cells)
        print(adata.obs.dca_split)
        adata.obs.loc[inf_5_cells, 'dca_split'] = 'test'
        adata_train = adata[adata.obs.dca_split == 'train'].copy()
        if semi_sup:
            adata_train = make_unk(adata_train,class_key=class_key, **split_kwds) # Adding artificial unknown cells
            celltype_unk = adata.obs[class_key].astype('str')
            unk_cells = adata_train.obs[class_key]
            celltype_unk[unk_cells.index] = unk_cells
            adata.obs[f'{class_key}_UNK'] = celltype_unk
            inf_5_cells = ~adata_train.obs[class_key].isin(adata_train.obs[class_key].value_counts()[adata_train.obs[class_key].value_counts()>training_kwds['n_perm']].index) # Rerun this step because the creation of UNK might have removed too many cells from certain class
            print(inf_5_cells)
            adata_train = adata_train[~inf_5_cells].copy()
            inf_5_cells = inf_5_cells[inf_5_cells].index
            adata.obs.loc[inf_5_cells, 'dca_split'] = 'test'
            
    print(adata_train)
    print(adata_train.obs[class_key].value_counts())
    hist = train(adata_train, net, **training_kwds)
    corrected_count = anndata.AnnData(net.model.predict({'count': adata.X, 'size_factors': adata.obs.size_factors}))
    latent_space = anndata.AnnData(net.encoder.predict({'count': adata.X, 'size_factors': adata.obs.size_factors}), obs=adata.obs)
    
#     sc.tl.pca(latent_space)
#     sc.pp.neighbors(latent_space)
#     sc.tl.umap(latent_space)
       
    if save_all :
        latent_path, model_path, net_kwd_path, train_kwd_path, train_hist_path, umap_path = results_save_path(dataset=dataset,
                                                                                                              class_key=class_key, 
                                                                                                              semi_sup=semi_sup,
                                                                                                              experiment=experiment,
                                                                                                              latent_size = hidden_size[1],
                                                                                                              **split_kwds).values()
        latent_space.write(latent_path)
        net.model.save(model_path)
        with open(net_kwd_path,'w') as net_file:
            yaml.dump(network_kwds, net_file, default_flow_style=False)
        with open(train_kwd_path,'w') as train_file:
            yaml.dump(training_kwds, train_file, default_flow_style=False) 
        sc.set_figure_params(dpi=100, figsize=(14,7))
        # sc.pl.umap(latent_space, color = class_key, save = umap_path , size = 20) # figure save path handling is a bit tedious, we don't save the figure, it's easy to plot anyway
        import pickle
        with open(train_hist_path, 'wb') as file_pi:
            pickle.dump(hist.history, file_pi)
    
    if return_info:
        adata.uns['dca_loss_history'] = hist.history

    if return_model:
        return (adata, net) if copy else net
    else:
        return adata if copy else None


def main():
    for class_key in ['random_sym','random_asym']:
        adata = sc.read_h5ad(WD_PATH + '/data/htap_annotated.h5ad')
        sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
        adata = adata[sup_5_cells,:].copy()
        adata.X = adata.raw.X.copy()
        sc.pp.filter_genes(adata,min_cells = 1)
        import pickle
        dca_permuted(adata, class_key=class_key)

        
def task_1(hidden_size=(64,32,64), experiment=None):
    class_key = 'celltype'
    train_spliting = True
    split_kwds = {'task':1,
                  'pct_split':0,
                  'obs_key':'celltype',
                  'n_keep':None,
                  #keep_obs=None,
                  'random_seed':42,
                  'obs_subsample':None}
    for pct_split in np.arange(0.1,1,0.1):
        adata = sc.read_h5ad(WD_PATH + '/data/htap_annotated.h5ad')
        sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
        adata = adata[sup_5_cells,:].copy()
        adata.X = adata.raw.X.copy()
        sc.pp.filter_genes(adata,min_cells = 1)
        split_kwds['pct_split'] = pct_split
        import pickle
        dca_permuted(adata,
            class_key=class_key,
            train_spliting = train_spliting,
            split_kwds=split_kwds,
            hidden_size=hidden_size,
            experiment=experiment,
            semi_sup = True)

            
def run_task1(dataset,
              class_key,
              latent_size,
              pct_split,
              experiment=None):
    split_kwds = {'task':1,
                  'pct_split':pct_split,
                  'obs_key':'celltype',
                  'n_keep':None,
                  #keep_obs=None,
                  'random_seed':42,
                  'obs_subsample':None}
    hidden_size = (latent_size * 2, latent_size, latent_size * 2)
    adata = sc.read_h5ad(dataset_path(dataset))
    sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
    adata = adata[sup_5_cells,:].copy()
    adata.X = adata.raw.X.copy()
    sc.pp.filter_genes(adata,min_cells = 1)
    import pickle
    dca_permuted(adata,
        class_key=class_key,
        train_spliting = True,
        split_kwds=split_kwds,
        hidden_size=hidden_size,
        experiment=experiment,
        semi_sup = True)


            
def task_2(hidden_size=(64,32,64), experiment=None):
    obs_key = 'dataset' # use celltype if want to drop celltypes, use dataset if want to train on various datasets
    train_spliting = True
    split_kwds = {'task':2,
                  'pct_split':0,
                  'obs_key':'celltype',
                  'n_keep':None,
                  'keep_obs':None,
                  'random_seed':42,
                  'obs_subsample':None}
    adata = sc.read_h5ad(WD_PATH + '/data/disco_htap_ajrccm_raw.h5ad')
    conditions = np.array(adata.obs[obs_key].cat.categories)
# For removing celltypes one at a time
#         for drop_obs in conditions:
#             adata = sc.read_h5ad('/data/dca_permuted/data/htap_annotated.h5ad')
#             sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
#             adata = adata[sup_5_cells,:].copy()
#             adata.X = adata.raw.X.copy()
#             sc.pp.filter_genes(adata,min_cells = 1)
#             split_kwds['keep_obs'] = celltypes[~(celltypes == drop_obs)] # keep every celltype but this one
#             import pickle
#             dca(adata, class_key=condition, train_spliting = train_spliting, split_kwds=split_kwds,hidden_size=hidden_size, experiment=experiment)
    for train_dataset in conditions:
        adata = sc.read_h5ad('/data/dca_permuted/data/disco_htap_ajrccm_raw.h5ad')
        class_key = 'conc_celltypes'
        sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
        adata = adata[sup_5_cells,:].copy()
        #adata.X = adata.raw.X.copy()
        sc.pp.filter_genes(adata,min_cells = 1)
        split_kwds['keep_obs'] = conditions[conditions == train_dataset] # Keep this dataset
        print(split_kwds['keep_obs'])
        split_kwds['obs_key'] = obs_key
        dataset = 'disco_htap_ajrccm'
        import pickle
        dca_permuted(adata, dataset=dataset, class_key=class_key, train_spliting = train_spliting, split_kwds=split_kwds,hidden_size=hidden_size, experiment=experiment)

            
def task_3(hidden_size=(64,32,64), experiment=None):
    class_key = 'celltype'
    train_spliting = True
    split_kwds = {'task':3,
                  'pct_split':None,
                  'obs_key':'celltype',
                  'n_keep':5,
                  #keep_obs=None,
                  'random_seed':42,
                  'obs_subsample':None}
    for n_keep in np.arange(10,75,5):
        adata = sc.read_h5ad(WD_PATH + '/data/htap_annotated.h5ad')
        sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
        adata = adata[sup_5_cells,:].copy()
        adata.X = adata.raw.X.copy()
        sc.pp.filter_genes(adata,min_cells = 1)
        split_kwds['n_keep'] = n_keep
        import pickle
        dca_permuted(adata, class_key=class_key, train_spliting=train_spliting, split_kwds=split_kwds,hidden_size=hidden_size, experiment=experiment)
        

def task_4(hidden_size=(64,32,64), experiment=None):
    n_keep = 10
    class_key = 'celltype'
    train_spliting = True
    split_kwds = {'task':4,
                  'pct_split':0,
                  'obs_key':'celltype',
                  'n_keep':n_keep, 
                  #keep_obs=None,
                  'random_seed':42,
                  'obs_subsample':None}
    adata = sc.read_h5ad(WD_PATH + '/data/htap_annotated.h5ad')
    classes = adata.obs[class_key].unique()
    for obs_subsample in classes:
        adata = sc.read_h5ad('/data/dca_permuted/data/htap_annotated.h5ad')
        sup_5_cells = adata.obs[class_key].isin(adata.obs[class_key].value_counts()[adata.obs[class_key].value_counts()>5].index)
        adata = adata[sup_5_cells,:].copy()
        adata.X = adata.raw.X.copy()
        sc.pp.filter_genes(adata,min_cells = 1)
        split_kwds['obs_subsample'] = obs_subsample
        import pickle
        dca_permuted(adata, class_key=class_key, train_spliting=train_spliting, split_kwds=split_kwds,hidden_size=hidden_size, experiment=experiment)

        
# task_1(hidden_size=(128,64,128))


# # for dataset in ['htap','ajrccm','discovair']:
# #     for pct_split in range(0.1,1,0.1):
# #         for latent_size in [32,64,128]:
# #             if dataset == 'ajrccm':
# #                 class_key = 'celltype'
# #                 run_task1(dataset,
# #                           class_key,
# #                           latent_size,
# #                           pct_split,
# #                           experiment=None):
# #             else ;
# #                 for class_key in ['celltype', 'phenotype', 'celltype_phenotype']:
# #                 run_task1(dataset,
# #                           class_key,
# #                           latent_size,
# #                           pct_split,
# #                           experiment=None)

# for dataset in ['htap','ajrccm','discovair']:
#     for pct_split in range(0.1,1,0.1):
#         for latent_size in [32,64,128]:
#             if dataset == 'ajrccm':
#                 class_key = 'celltype'
#                 latent_path, model_path, net_kwd_path, train_kwd_path, train_hist_path, umap_path = results_save_path(dataset=dataset,
#                                                                                                               class_key=class_key, 
#                                                                                                               semi_sup=semi_sup,
#                                                                                                               experiment=experiment,
#                                                                                                               latent_size = hidden_size[1],
#                                                                                                               **split_kwds).values()
#                 latent = sc.read_h5ad(latent_path)
#                 sc.tl.pca(latent_space)
#                 sc.pp.neighbors(latent_space)
#                 sc.tl.umap(latent_space)
#                 latent.write(latent_path)