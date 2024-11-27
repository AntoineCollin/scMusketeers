import argparse
import sys
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import neptune

WD_PATH = '/home/acollin/scPermut/'
sys.path.append(WD_PATH)

from scmusketeers.tools.utils import str2bool, load_json
print(str2bool('True'))
from scmusketeers.workflow.benchmark import Workflow

model_list_cpu = ['scmap_cells', 'scmap_cluster', 'pca_svm', 'pca_knn','harmony_svm','celltypist','uce']
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
    parser.add_argument('--use_hvg', type=int, nargs='?', const=3000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")

    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--test_obs', type = str,nargs='+', default = None, help ='batches from batch_key to use as test')
    parser.add_argument('--test_index_name', type = str,nargs='+', default = None, help ='indexes to be used as test. Overwrites test_obs')

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
    parser.add_argument('--working_dir', type=str, nargs='?',const='/home/acollin/scPermut/', default='/home/acollin/scPermut/', help ='')


    run_file = parser.parse_args()
    print(run_file.class_key, run_file.batch_key)
    working_dir = run_file.working_dir
    print(f'working directory : {working_dir}')

    project = neptune.init_project(
            project="becavin-lab/benchmark",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        mode="read-only",
            )# For checkpoint

    runs_table_df = project.fetch_runs_table().to_pandas()
    project.stop()
    
    if run_file.gpu_models :
        model_list = model_list_gpu
    else:
        model_list = model_list_cpu

    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    experiment.process_dataset()

    experiment.mode = "entire_condition"

    random_seed = 2

    X = experiment.dataset.adata.X
    classes = experiment.dataset.adata.obs[experiment.class_key]
    groups = experiment.dataset.adata.obs[experiment.batch_key]


    n_batches = len(groups.unique())
    nfold_test = max(1,round(n_batches/5)) # if less than 8 batches, this comes to 1 batch per fold, otherwise, 20% of the number of batches for test
    kf_test = GroupShuffleSplit(n_splits=3, test_size=nfold_test, random_state=random_seed)
    test_split_key = experiment.dataset.test_split_key

    test_obs_json = load_json(working_dir + 'experiment_script/benchmark/hp_test_obs.json')
    test_obs = test_obs_json[run_file.dataset_name]

    experiment.test_obs = test_obs
    experiment.split_train_test()

    nfold_val = max(1,round((n_batches-len(test_obs))/5)) # represents 20% of the remaining train set
    kf_val = GroupShuffleSplit(n_splits=5, test_size=nfold_val, random_state=random_seed)
   
    X_train_val = experiment.dataset.adata_train_extended.X
    classes_train_val = experiment.dataset.adata_train_extended.obs[experiment.class_key]
    groups_train_val = experiment.dataset.adata_train_extended.obs[experiment.batch_key]

    for j, (train_index, val_index) in enumerate(kf_val.split(X_train_val, classes_train_val, groups_train_val)):
        if j == 1: # Running only on the first split at first because it takes time
            experiment.keep_obs = list(groups_train_val[train_index].unique()) # keeping only train idx
            val_obs = list(groups_train_val[val_index].unique())
            experiment.split_train_test_val()
            for model in model_list:
                checkpoint={'parameters/dataset_name': experiment.dataset_name, 'parameters/task': 'task_4', 'parameters/use_hvg': experiment.use_hvg,
                    'parameters/model': model, 'parameters/val_fold_nb':j}
                result = runs_table_df[runs_table_df[list(checkpoint.keys())].eq(list(checkpoint.values())).all(axis=1)]
                if result.empty:
                    print(checkpoint)
                    print(f'Running {model}')
                    experiment.start_neptune_log()
                    experiment.add_custom_log('val_fold_nb',j)
                    experiment.add_custom_log('test_obs',test_obs)
                    experiment.add_custom_log('val_obs',val_obs)
                    experiment.add_custom_log('train_obs',experiment.keep_obs)
                    experiment.add_custom_log('task','task_4')
                    experiment.add_custom_log('deprecated_status',False)
                    experiment.add_custom_log('debug_status', "fixed_1")
                    experiment.train_model(model)
                    experiment.compute_metrics()
                    experiment.stop_neptune_log()
