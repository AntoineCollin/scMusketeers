import nbformat as nbf
from jinja2 import Environment, FileSystemLoader
import pandas as pd
import numpy as np
import os
from workflow.clust_compute import *
import scanpy as sc
from workflow.runfile_handler import run_file_handler
from workflow.analysis import AnalysisWorkflow
from tqdm.notebook import trange, tqdm


work_dir = '/home/acollin/dca_permuted_workflow/'

hca = sc.read_h5ad('/home/acollin/Discovair/data/HLCA_v1.h5ad')
raw_ajrccm = sc.read_h5ad(work_dir + 'data/HCA_Barbry_Grch38_Raw.h5ad')

ajrccm_hca = hca[hca.obs.study_long == 'CNRS_Barbry_Leroy_2020Deprez',:].copy()
del(hca)

ann = np.array(ajrccm_hca.obs['original_ann_level_5'])
isnone = ann == 'None'
ann[isnone] = list(ajrccm_hca.obs['original_ann_level_4'][isnone])
isnone = ann == 'None'
ann[isnone] = list(ajrccm_hca.obs['original_ann_level_3'][isnone])
isnone = ann == 'None'
ann[isnone] = list(ajrccm_hca.obs['original_ann_level_2'][isnone])
isnone = ann == 'None'
ann[isnone] = list(ajrccm_hca.obs['original_ann_level_1'][isnone])

ajrccm_hca.obs['finest_original_ann_level'] = ann

match_ct_original = pd.read_csv('/home/acollin/dca_permuted_workflow/celltype_matching_HCA_AJRCCM_original.csv', index_col = 0)
match_ct_original = {key:value for key,value in zip(match_ct_original['HCA'],match_ct_original['AJRCCM'])}
match_ct_reann = pd.read_csv('/home/acollin/dca_permuted_workflow/celltype_matching_HCA_AJRCCM_reann.csv', index_col = 0)
match_ct_reann = {key:value for key,value in zip(match_ct_reann['HCA'],match_ct_reann['AJRCCM'])}
celltypes_hca = ajrccm_hca.obs.loc[:, ['finest_original_ann_level','ann_finest_level']]
celltypes_hca['finest_original_ann_level_harmonized'] = celltypes_hca['finest_original_ann_level'].replace(match_ct_original)
celltypes_hca['ann_finest_level_harmonized'] = celltypes_hca['ann_finest_level'].replace(match_ct_reann)
aw_ajrccm = AnalysisWorkflow(working_dir = '/home/acollin/dca_permuted_workflow', id_list=np.arange(6,16))
aw_ajrccm.load_latent_spaces()
aw_ajrccm.subsample_on_index(celltypes_hca.index)
aw_ajrccm.add_obs_metadata(celltypes_hca)
aw_ajrccm.compute_metrics(verbose = True)

for wf_id, latent_space in aw_ajrccm.latent_spaces.items():
    celltypes_dca = latent_space.obs.loc[:, ['celltype','celltype_pred']]
    celltypes_hca = latent_space.obs.loc[:, ['finest_original_ann_level_harmonized','ann_finest_level_harmonized']]
    reann_dca = celltypes_dca['celltype'] != celltypes_dca['celltype_pred']
    reann_hca = celltypes_hca['finest_original_ann_level_harmonized'] != celltypes_hca['ann_finest_level_harmonized']
    latent_space.obs['reann_dca_color'] = reann_dca.replace({True:'corrected', False:'same'})
    latent_space.obs['reann_hca_color'] = reann_hca.replace({True:'corrected', False:'same'})
    latent_space.obs['common_hca_dca_color'] = (reann_hca & reann_dca).replace({True:'corrected_both', False:'other'})
    latent_space.obs['reann_dca'] = reann_dca
    latent_space.obs['reann_hca'] = reann_hca
    latent_space.obs['common_hca_dca'] = (reann_hca & reann_dca)
    
adata = aw_ajrccm.latent_spaces[8]


ajrccm_hca = ajrccm_hca[adata.obs.index,:].copy()
raw_ajrccm = raw_ajrccm[adata.obs.index,:].copy()
ajrccm_hca.obsm['original_umap'] = raw_ajrccm.obsm['umap_hg19']
ajrccm_hca.obsm['dca_permuted_embedding'] = adata.X
ajrccm_hca.obsm['dca_permuted_umap'] = adata.obsm['X_umap']
ajrccm_hca.obsm['HCA_scanvi_embedding'] = ajrccm_hca.obsm['X_scanvi_emb']
ajrccm_hca.obsm['HCA_umap'] = ajrccm_hca.obsm['X_umap']
del ajrccm_hca.obsm['X_umap']
del ajrccm_hca.obsm['X_scanvi_emb']
ajrccm_hca.obs = adata.obs
score_table = pd.DataFrame(columns = ['method', 'embedding', 'davies-bouldin', 'silhouette'])
match_obs_method = {
    'original' : 'celltype',
    'hca' : 'ann_finest_level_harmonized', 
    'dca_permuted' : 'celltype_pred'
}


match_obs_method = {
    'original' : 'celltype',
    'hca_harmonized' : 'ann_finest_level_harmonized',
    'hca' : 'ann_finest_level',
    'dca_permuted' : 'celltype_pred',
    'donor' : 'subject_ID'
}
embeddings = list(ajrccm_hca.obsm.keys())+[None]

for embedding in tqdm(embeddings):
    if not embedding :
        ajrccm_hca.X = ajrccm_hca.X.toarray()
    for method in ['original', 'hca', 'hca_harmonized', 'dca_permuted']:
        embedding_name = embedding
        silhouette_score = silhouette(ajrccm_hca, partition_key=match_obs_method[method], obsm_representation=embedding)
        print('silhouette ok')
        davies_bouldin_score = davies_bouldin(ajrccm_hca, partition_key=match_obs_method[method], obsm_representation=embedding)
        if not embedding:
            embedding_name = 'normalized_counts'
        score_table = score_table.append({'method' : method,
                            'embedding' : embedding_name,
                            'davies-bouldin' : davies_bouldin_score,
                            'silhouette' : silhouette_score},ignore_index=True)
        
score_table.to_csv(work_dir + 'metrics/dca_vs_hca_metrics.csv')