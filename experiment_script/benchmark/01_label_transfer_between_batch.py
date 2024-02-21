import argparse
import sys
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd

WD_PATH = '/home/acollin/dca_permuted_workflow/'
sys.path.append(WD_PATH)

from scpermut.tools.utils import str2bool
from scpermut.workflow.benchmark import Workflow

model_list_cpu = ['scmap_cells', 'scmap_cluster', 'celltypist', 'pca_svm','harmony_svm','uce', 'scanvi']
model_list_gpu = ['scanvi']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument('--dataset_name', type = str, default = 'htap_final_by_batch', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--class_key', type = str, default = 'celltype', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'donor', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')# TODO :remove, we always want to do that
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')
    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=5000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")

    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = None, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--gpu_models', type=str2bool, nargs='?',const=False, default=False , help ='')

    run_file = parser.parse_args()
    print(run_file.class_key, run_file.batch_key)
    working_dir = '/home/acollin/dca_permuted_workflow/'

    if run_file.gpu_models :
        model_list = model_list_gpu
    else:
        model_list = model_list_cpu

    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    experiment.process_dataset()

    experiment.mode = "entire_condition"

    random_seed = 2

    n_batches = len(experiment.dataset.adata.obs[experiment.batch_key].unique())
    nfold_test = max(1,round(n_batches/5)) # if less than 8 batches, this comes to 1 batch per fold, otherwise, 20% of the number of batches for test
    kf_test = GroupShuffleSplit(n_splits=3, test_size=nfold_test, random_state=random_seed)
    test_split_key = experiment.dataset.test_split_key

    for i, (train_index, test_index) in enumerate(kf_test.split(experiment.dataset.adata.X, experiment.dataset.adata.obs[experiment.class_key], experiment.dataset.adata.obs[experiment.batch_key])):
        groups = experiment.dataset.adata.obs[experiment.batch_key]
        test_obs = list(experiment.dataset.adata.obs[experiment.batch_key].iloc[test_index].unique()) # the batches that go in the test set
        
        test_idx = experiment.dataset.adata.obs[experiment.batch_key].isin(test_obs)
        split = pd.Series(['train'] * experiment.dataset.adata.n_obs, index = experiment.dataset.adata.obs.index)
        split[test_idx] = 'test'
        experiment.dataset.adata.obs[experiment.test_split_key] = split

        experiment.dataset.test_split() # splits the train and test dataset
        nfold_val = max(1,round((n_batches-len(test_obs))/5)) # represents 20% of the remaining train set
        kf_val = GroupShuffleSplit(n_splits=5, test_size=nfold_val, random_state=random_seed)

        for j, (train_index, val_index) in enumerate(kf_val.split(experiment.dataset.adata_train_extended.X, experiment.dataset.adata_train_extended.obs[experiment.class_key], experiment.dataset.adata_train_extended.obs[experiment.batch_key])):
            groups_sub = experiment.dataset.adata_train_extended.obs[experiment.batch_key]
            experiment.keep_obs = list(experiment.dataset.adata_train_extended.obs[experiment.batch_key][train_index].unique()) # keeping only train idx
            val_obs = list(experiment.dataset.adata_train_extended.obs[experiment.batch_key][val_index].unique())
            print(f"Fold {i,j}:")
            print(f"train = {list(groups_sub.iloc[train_index].unique())}, len = {len(groups_sub.iloc[train_index].unique())}")
            print(f"val = {list(groups_sub.iloc[val_index].unique())}, len = {len(groups_sub.iloc[val_index].unique())}")
            print(f"test = {list(groups.iloc[test_index].unique())}, len = {len(groups.iloc[test_index].unique())}")

            

            print(set(groups_sub.iloc[train_index].unique()) & set(groups.iloc[test_index].unique()))
            print(set(groups_sub.iloc[train_index].unique()) & set(groups_sub.iloc[val_index].unique()))
            print(set(groups_sub.iloc[val_index].unique()) & set(groups.iloc[test_index].unique()))
            experiment.split_train_test_val()
            print(experiment.dataset.adata.obs.loc[:,[experiment.test_split_key,experiment.batch_key]].drop_duplicates())
            for model in model_list:
                print(f'Running {model}')
                experiment.start_neptune_log()
                experiment.add_custom_log('test_fold_nb',i)
                experiment.add_custom_log('val_fold_nb',j)
                experiment.add_custom_log('test_obs',test_obs)
                experiment.add_custom_log('val_obs',val_obs)
                experiment.add_custom_log('train_obs',experiment.keep_obs)
                experiment.add_custom_log('task','task_1')
                experiment.train_model(model)
                experiment.compute_metrics()
                experiment.stop_neptune_log()
