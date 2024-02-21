import os
import pickle
import scanpy as sc

WD_PATH = "/home/acollin/dca_permuted"
DATA_PATH = WD_PATH + "/data"
RESULTS_PATH = WD_PATH + "/results"
FIGURE_PATH = WD_PATH + "/figures"

def dataset_path(dataset_name):
    DATASETS = {'htap':'htap_annotated',
                'lca' : 'LCA_log1p',
                'discovair':'discovair_V6',
                'ajrccm':'HCA_Barbry_Grch38_Raw',
                "disco_htap_ajrccm":'disco_htap_ajrccm_raw',
                'htap_ajrccm': 'htap_ajrccm_raw'}
    return DATA_PATH + '/' + DATASETS[dataset_name] + '.h5ad'

# def kept_genes_path(ref_dataset, query_dataset, n_genes):
#     return DATA_PATH + '/kept_genes/kept_genes' + f'{query_dataset}_on_{ref_dataset}' + str(n_genes) + '.csv'

# def data_path(dataset_name):
#     return DATA_PATH + '/' + dataset_name + '.h5ad'

def experiment_handling(result_dir, umap_save_path, experiment):
    i=1
    if experiment :
        i=experiment
        result_dir = result_dir + 'experiment_' + str(i)
        umap_save_dir = FIGURE_PATH + '/' + 'umap' + '/' + umap_save_path + 'experiment_' + str(i) 
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
#         if not os.path.isdir(umap_save_dir):
#             os.makedirs(umap_save_dir)
    else:
        while os.path.isdir(result_dir + 'experiment_' + str(i)):
            i+=1
        result_dir = result_dir + 'experiment_' + str(i)
        umap_save_dir = FIGURE_PATH + '/' + 'umap' + '/' + umap_save_path + 'experiment_' + str(i) 
        os.makedirs(result_dir)
#         os.makedirs(umap_save_dir)
    return result_dir, umap_save_dir, i

def results_save_path(dataset,
                      class_key,
                      semi_sup,
                      latent_size,
                      experiment=None,
                      task=None,
                      pct_split=None,
                      obs_key=None,
                      n_keep=None,
                      keep_obs=None, 
                      random_seed=None,
                      obs_subsample=None,
                      true_celltype=None,
                      false_celltype=None, 
                      pct_true=None, 
                      pct_false=None):
    if semi_sup:
        result_dir = RESULTS_PATH + '_semi_sup_n_perm_5' + '/' + 'task_' + str(task) + '/' + dataset + '/' + class_key + '/' + 'latent' + '_' + str(latent_size) + '/'
        umap_save_path = '_semi_sup_n_perm_5' + '/' + 'task_' + str(task) + '/' + dataset + '/' + class_key + '/' + 'latent' + '_' + str(latent_size) + '/'
    else :
        result_dir = RESULTS_PATH + '/' + 'task_' + str(task) + '/' + dataset + '/' + class_key + '/' + 'latent' + '_' + str(latent_size) + '/'
        umap_save_path = '/' + 'task_' + str(task) + '/' + dataset + '/' + class_key + '/' + 'latent' + '_' + str(latent_size) + '/'
    if task == 0:
        result_dir = RESULTS_PATH + '/' 'task_' + str(task) + '/' + dataset + '/' + class_key + '/'
        umap_save_path = '/' + str(task) + '/' + dataset + '/' + class_key + '/'
    if task == 1:
        result_dir += str(pct_split) + '_pct' + '/'
        umap_save_path += str(pct_split) + '_pct' + '/'
    if task == 2:
        result_dir += obs_key + '_split' + '/' + '_'.join([obs[:3].replace('/','_') for obs in keep_obs]) + '/' # TODO : find a way to pass the keep_obs value, probably in the experiment
        umap_save_path += obs_key + '_split' + '/' + '_'.join([obs[:3].replace('/','_') for obs in keep_obs]) + '/' 
    if task == 3:
        result_dir += obs_key + '_split' + '/' + 'keep_' + str(n_keep) + '/'
        umap_save_path += obs_key + '_split' + '/' + 'keep_' + str(n_keep) + '/'
    if task == 4:
        result_dir += obs_key + '_split' + '/' + obs_subsample + '/' + 'keep_' + str(n_keep) + '/'
        umap_save_path += obs_key + '_split' + '/' + obs_subsample + '/' + 'keep_' + str(n_keep) + '/'
    if task == 5:
        result_dir += 'true_' + true_celltype.replace('/','_') + '/' + 'false_' + false_celltype.replace('/','_') + '/' + str(pct_true) + '_true_' + str(pct_false) + '_false' + '/'
        umap_save_path += 'true_' + true_celltype.replace('/','_') + '/' + 'false_' + false_celltype.replace('/','_') + '/' + str(pct_true) + '_true_' + str(pct_false) + '_false' + '/'
    result_dir, umap_save_dir, i = experiment_handling(result_dir, umap_save_path, experiment)
    latent_save_path = result_dir + '/' + 'latent.h5ad'
    model_save_path = result_dir + '/' + 'trained_model'
    net_kwd_path = result_dir + '/' + 'net_kwds.yml'
    train_kwd_path = result_dir + '/' + 'train_kwds.yml'
    umap_save_path = umap_save_path + 'experiment_' + str(i) + '/' + 'umap_latent.png'
    train_hist_path = result_dir + '/' + 'train_hist.pkl'
    return {'latent':latent_save_path, 'model': model_save_path, 'net': net_kwd_path, 'train_kwds': train_kwd_path, 'train_hist':train_hist_path, 'umap':umap_save_path}

def predict_save_path(dataset,
                      class_key,
                      semi_sup,
                      latent_size,
                      experiment=None,
                      task=None,
                      pct_split=None,
                      obs_key=None,
                      n_keep=None,
                      keep_obs=None, 
                      random_seed=None,
                      obs_subsample=None,
                      true_celltype=None,
                      false_celltype=None, 
                      pct_true=None, 
                      pct_false=None):
    result_dir = results_save_path(dataset,
                                  class_key,
                                  semi_sup,
                                  latent_size,
                                  experiment=experiment,
                                  task=task,
                                  pct_split=pct_split,
                                  obs_key=obs_key,
                                  n_keep=n_keep,
                                  keep_obs=keep_obs, 
                                  random_seed=random_seed,
                                  obs_subsample=obs_subsample,
                                  true_celltype=true_celltype,
                                  false_celltype=false_celltype, 
                                  pct_true=pct_true, 
                                  pct_false=pct_false)['latent'].strip('latent.h5ad')
    predict_dir = result_dir + 'predict'
    model_predict_save_path = predict_dir + '/' + 'trained_model'
    predict_save_path = predict_dir + '/' + 'prediction.csv'
    return {'predictions' : predict_save_path, 'model' : model_predict_save_path}

def umap_save_path(model_type, ref_dataset, query_dataset, n_genes, dataset_type):
    """
    dataset_type corresponds to 'ref', 'query', 'full' and can be extended when splitting query in multiple trainings
    """
    save_dir = FIGURE_PATH + '/umap' + '/' +  f'{query_dataset}_on_{ref_dataset}' + '/' + model_type + '/' + f'{str(n_genes)}_genes'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = '/' +  f'{query_dataset}_on_{ref_dataset}' + '/' + model_type + '/' + f'{str(n_genes)}_genes' + '/' + f'umap_latent_{dataset_type}.png'
    return save_path

def load_experiment(dataset,
                      class_key,
                      semi_sup,
                      latent_size,
                      experiment=None,
                      task=None,
                      pct_split=None,
                      obs_key=None,
                      n_keep=None,
                      #keep_obs=None, 
                      random_seed=None,
                      obs_subsample=None):
    paths = results_save_path(dataset,
                              class_key,
                              semi_sup,
                              latent_size,
                              experiment,
                              task,
                              pct_split,
                              obs_key,
                              n_keep,
                              #keep_obs=None, 
                              random_seed,
                              obs_subsample)
    adata = sc.read_h5ad(dataset_path(dataset))
    latent_adata = sc.read_h5ad(paths['latent'])
    with open(paths['train_hist'], 'rb') as hist_file:
        train_hist = pickle.load(hist_file)
    # model = keras.models.load_model(paths['model'])
    return adata, latent_adata, train_hist #, model
