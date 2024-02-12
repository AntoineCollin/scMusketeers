#!/usr/bin/python
from sklearn import svm
import anndata as ad
import pandas as pd
import scanpy as sc
import scanpy.external as sce
import numpy as np 
import anndata
import scvi
print("Last run with scvi-tools version:", scvi.__version__)

def pca_svm(X_list, y_list, batch_list, assign,
            pred_full = True): 
    """ Perform PCA reduction and then predict cell type's 
    annotation with a SVM algorithm 
    return :
        - latent = X_pca
        - y_pred = prediction for all cells"""
    adata = sc.AnnData(X = X_list['full'],
                       obs = pd.DataFrame({
                           'celltype': y_list['full'],
                           'batch': batch_list['full'],
                           'split': assign}))
    sc.tl.pca(adata)
    X_pca = adata.obsm['X_pca']
    y_pred = svm_label(X_pca, y_list, assign, pred_full = pred_full)
    
    X_pca_list = {group : X_pca[assign == group, :] for group in np.unique(assign)}
    X_pca_list['full'] = X_pca
    y_pred_list =  {group : y_pred[assign == group] for group in np.unique(assign)}
    y_pred_list['full'] = y_pred
    return X_pca_list, y_pred_list


def harmony_svm(X_list, y_list, batch_list, assign,
                pred_full = True): 
    """ Perform an integration from different dataset
    and then predict cell type's 
    annotation with a SVM algorithm 
    return :
        - latent = X_pca_harmony
        - y_pred = prediction for all cells"""
    adata = sc.AnnData(X = X_list['full'],
                       obs = pd.DataFrame({
                           'celltype': y_list['full'],
                           'batch': batch_list['full'],
                           'split': assign}))
    sc.tl.pca(adata)
    sce.pp.harmony_integrate(adata, 'batch')
    X_pca_harmony = adata.obsm['X_pca_harmony']
    y_pred = svm_label(X_pca_harmony, y_list, assign, pred_full = pred_full)

    X_pca_harmony_list = {group : X_pca_harmony[assign == group, :] for group in np.unique(assign)}
    X_pca_harmony_list['full'] = X_pca_harmony
    y_pred_list =  {group : y_pred[assign == group] for group in np.unique(assign)}
    y_pred_list['full'] = y_pred
    return X_pca_harmony_list, y_pred_list
    


def svm_label(X_full, y_list, assign, pred_full = True):
    X_train = X_full[assign != 'test', :]
    X_test = X_full[assign == 'test', :]
 
    y_train = y_list['full'][assign != 'test']
    y_test = y_list['test']
    
    clf = svm.SVC() # default rbf ok ? or Linear Kernel ?
    clf.fit(X_train, y_train)
    
    if pred_full:
        y_pred = clf.predict(X_full)
    else:
        y_pred = clf.predict(X_test)
        
    return y_pred


def scanvi(X_list, y_list, batch_list, assign):
    """ Perform scvi and scanvi integration and 
    do the predict on the unknow cells with scanvi
    input : 
        - anndata : .X = raw count ++ (not scale)
    return :
        - latent = latent_space from scanvi
        - y_pred = prediction for all cells"""
    unlabeled_category = "Unknown"
    SCANVI_LATENT_KEY = "X_scANVI"
    SCANVI_PREDICTION_KEY = "pred_scANVI" # "C_scANVI"

    adata = sc.AnnData(X = X_list['full'],
                       obs = pd.DataFrame({
                           'celltype': y_list['full'],
                           'celltype_full': y_list['full'],
                           'batch': batch_list['full'],
                           'split': assign}))
    adata.obs['celltype'] = adata.obs['celltype'].astype(str)
    adata.obs['celltype'][adata.obs['split'].isin(['val', "test"])] = unlabeled_category
    sorting_order = {'train': 1, 'val': 2, 'test': 3}
    sorted_df = adata.obs.sort_values(by='split', 
                                      key=lambda x:x.map(sorting_order))
    adata = adata[sorted_df.index]
    adata.obs = adata.obs.reset_index(drop=True)
    # adata.layers['count'] = adata.X
    
    # Run scvi
    scvi.model.SCVI.setup_anndata(adata, 
                                  # layer = "count", 
                                  batch_key = 'batch',
                                  labels_key = 'celltype')
    scvi_model = scvi.model.SCVI(adata, 
                                 n_layers = 50, # default = 10  or 50 ?
                                 n_latent = 2) # default = 1 
    print("start train scvi")
    scvi_model.train(train_size = 1,
                     validation_size = None#,
                     # shuffle_set_split = False
                    )
    
    # Run scanvi
    scanvi_model = scvi.model.SCANVI.from_scvi_model(scvi_model,
                                                     adata = adata,
                                                     unlabeled_category = unlabeled_category,
                                                     labels_key = 'celltype')
    print("start train scanvi")
    scanvi_model.train(max_epochs = 20, 
                       n_samples_per_label = 100,
                       train_size = 1,
                       validation_size = None#,
                       # shuffle_set_split = False
                      )

    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(adata)
    adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata)

    latent_list = {adata.obsm[SCANVI_LATENT_KEY][assign == group, :] for group in np.unique(assign)}
    latent_list['full'] = adata.obsm[SCANVI_LATENT_KEY]
    y_pred_list =  {adata.obs[SCANVI_PREDICTION_KEY][assign == group, :] for group in np.unique(assign)}
    y_pred_list['full'] = adata.obs[SCANVI_PREDICTION_KEY]
    return latent_list, y_pred_list

