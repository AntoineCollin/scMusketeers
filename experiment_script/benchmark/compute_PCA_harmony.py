import scanpy as sc
import scanpy.external as sce
import sys
import scipy

dataset_dir = '/home/acollin/dca_permuted_workflow/data'

dataset_names = {'htap':'htap_annotated',
                'lca' : 'LCA_log1p',
                'discovair':'discovair_V6',
                'discovair_V7':'discovair_V7',
                'discovair_V7_filtered':'discovair_V7_filtered_raw', # Filtered version with doublets, made lighter to pass through the model
                'discovair_V7_filtered_no_D53':'discovair_V7_filtered_raw_no_D53',
                'ajrccm':'HCA_Barbry_Grch38_Raw',
                "ajrccm_by_batch":"ajrccm_by_batch",
                "disco_htap_ajrccm":'discovair_htap_ajrccm',
                "disco_htap": 'discovair_htap',
                "disco_ajrccm": 'discovair_ajrccm',
                "disco_ajrccm_downsampled":'discovair_ajrccm_downsampled',
                "discovair_ajrccm_small" : "discovair_ajrccm_small",
                'htap_ajrccm': 'htap_ajrccm_raw',
                'pbmc3k_processed':'pbmc_3k',
                'htap_final':'htap_final',
                'htap_final_by_batch':'htap_final_by_batch',
                'htap_final_C1_C5':'htap_final_C1_C5',
                'pbmc8k':'pbmc8k',
                'pbmc68k':'pbmc68k',
                'pbmc8k_68k':'pbmc8k_68k',
                'pbmc8k_68k_augmented':'pbmc8k_68k_augmented',
                'pbmc8k_68k_downsampled':'pbmc8k_68k_downsampled',
                'htap_final_ajrccm': 'htap_final_ajrccm',
                'hlca_par_sample_harmonized':'hlca_par_sample_harmonized',
                'hlca_par_dataset_harmonized':'hlca_par_dataset_harmonized',
                'hlca_trac_sample_harmonized':'hlca_trac_sample_harmonized',
                'hlca_trac_dataset_harmonized':'hlca_trac_dataset_harmonized' ,
                'koenig_2022' : 'celltypist_dataset/koenig_2022/koenig_2022',
                'tosti_2021' : 'celltypist_dataset/tosti_2021/tosti_2021',
                'yoshida_2021' : 'celltypist_dataset/yoshida_2021/yoshida_2021',
                'yoshida_2021_debug' : 'celltypist_dataset/yoshida_2021/yoshida_2021_debug',
                'tran_2021' : 'celltypist_dataset/tran_2021/tran_2021',
                'dominguez_2022_lymph' : 'celltypist_dataset/dominguez_2022/dominguez_2022_lymph',
                'dominguez_2022_spleen' : 'celltypist_dataset/dominguez_2022/dominguez_2022_spleen',
                'tabula_2022_spleen' : 'celltypist_dataset/tabula_2022/tabula_2022_spleen',
                'litvinukova_2020' : 'celltypist_dataset/litvinukova_2020/litvinukova_2020',
                 'lake_2021': 'celltypist_dataset/lake_2021/lake_2021'
                }

dataset_dir = '/home/acollin/dca_permuted_workflow/data'

data_name = sys.argv[1]

def load_dataset(dataset_name):
    dataset_path = dataset_dir + '/' + dataset_names[dataset_name] + '.h5ad'
    adata = sc.read_h5ad(dataset_path)
    if type(adata.X) == scipy.sparse.csr.csr_matrix:
        print(adata.X[:14,:14].todense())
    else : 
        print(adata.X[:14,:14])
    print(adata)
    return adata

def write_dataset(adata, dataset_name):
    dataset_path = dataset_dir + '/' + dataset_names[dataset_name] + '.h5ad'
    adata.write(dataset_path)

batch_list = {"ajrccm_by_batch":"manip",
                "htap_final_by_batch":"donor",
                "hlca_par_dataset_harmonized":"dataset",
                "hlca_trac_dataset_harmonized":"dataset",
                "koenig_2022": "batch",
                "tosti_2021": "batch",
                "yoshida_2021": "batch",
                "tran_2021": "batch",
                "dominguez_2022_lymph": "batch",
                "dominguez_2022_spleen": "batch",
                "tabula_2022_spleen": "batch",
                "litvinukova_2020": "batch",
                "lake_2021": "batch"}

adata = load_dataset(data_name)
# sc.tl.pca(adata)
# sce.pp.harmony_integrate(adata, batch_list[data_name])
del adata.obsm['X_pca']
del adata.obsm['X_pca_harmony']
write_dataset(adata, data_name)

print(data_name + ' done')