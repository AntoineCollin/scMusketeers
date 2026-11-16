import argparse
import sys
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import neptune
import logging

WD_PATH = '/workspace/cell/scMusketeers/'
sys.path.append(WD_PATH)

from scmusketeers.tools.utils import str2bool
print(str2bool('True'))
from scmusketeers.workflow.benchmark import Workflow

logger = logging.getLogger("Sc-Musketeers")
logging.basicConfig(format="|--- %(levelname)-8s    %(message)s")
logger.setLevel(getattr(logging, "DEBUG"))

# all_model_list_cpu = ['uce','celltypist'] #'scmap_cells', 'scmap_cluster', 'pca_svm', 'pca_knn','harmony_svm','celltypist','uce']

model_list_cpu = ['uce','celltypist', 'scmap_cells', 'scmap_cluster', 'pca_svm', 'pca_knn']
#harmony_svm, not working

#model_list_cpu = ['uce']
model_list_gpu = ['scanvi', ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type = str, default = 'htap_final_by_batch', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--task', type=str, nargs='?', default='', help ='The task running: task1, task2, hyperparam, etc...')
    parser.add_argument('--model', type=str, nargs='?', default='pca_svm', help ='The model to run for the benchmark: uce, celltypist, etc.. ')

    parser.add_argument('--class_key', type = str, default = 'celltype', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'donor', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')

    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=3000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")

    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--test_obs', type = str,nargs='+', default = None, help ='batches from batch_key to use as test')
    parser.add_argument('--test_index_name', type = str,nargs='+', default = None, help ='indexes to be used as test. Overwrites test_obs')

    parser.add_argument('--test_fold_nb', type=int, nargs='?', default=None, help='Test fold index to run (0, 1, 2). If None, runs all folds.')
    parser.add_argument('--pct_split_nb', type=int, nargs='?', default=None, help='pct_split index to run (0=0.05, 1=0.1, 2=0.5, 3=0.9). If None, runs all.')
    parser.add_argument('--random_seed_nb', type=int, nargs='+', default=None, help='random_seed index(es) to run (0=30 … 5=35). If None, runs all.')
    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = None, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')

    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=False , help ='')
    parser.add_argument('--gpu_models', type=str2bool, nargs='?',const=False, default=False , help ='')
    parser.add_argument('--working_dir', type=str, nargs='?',const='/workspace/cell/scMusketeers/', default='/workspace/cell/scMusketeers/', help ='')


    run_file = parser.parse_args()
    logger.debug(f"Class_key: {run_file.class_key} Batch_key: {run_file.batch_key}")
    working_dir = run_file.working_dir
    logger.debug(f'working directory : {working_dir}')
    model = run_file.model

    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    if model == 'uce':
        experiment.use_hvg = None

    experiment.process_dataset()

    experiment.mode = "percentage"

    test_random_seed = 2 # The seed for the test split

    n_batches = len(experiment.dataset.adata.obs[experiment.batch_key].unique())
    nfold_test = max(1,round(n_batches/5))
    kf_test = GroupShuffleSplit(n_splits=3, test_size=nfold_test, random_state=test_random_seed)
    test_split_key = experiment.dataset.test_split_key

    X = experiment.dataset.adata.X
    classes = experiment.dataset.adata.obs[experiment.class_key]
    groups = experiment.dataset.adata.obs[experiment.batch_key]

    for i, (train_index, test_index) in enumerate(kf_test.split(X, classes, groups)):
        if run_file.test_fold_nb is not None and i != run_file.test_fold_nb:
            continue
        test_obs = list(groups.iloc[test_index].unique()) # the batches that go in the test set
        experiment.dataset.test_split(test_obs = test_obs) # splits the train and test dataset

        pct_list = [0.05, 0.1, 0.5, 0.9]
        seed_list = [30, 31, 32, 33, 34, 35]
        for pi, pct_split in enumerate(pct_list):
            if run_file.pct_split_nb is not None and pi != run_file.pct_split_nb:
                continue
            for si, random_seed in enumerate(seed_list):
                if run_file.random_seed_nb is not None and si not in run_file.random_seed_nb:
                    continue
                experiment.pct_split = pct_split
                experiment.train_test_random_seed = random_seed # The seed for the train val split

                experiment.split_train_test_val() # splitting val and train
                split = experiment.dataset.adata.obs[experiment.test_split_key]

                train_idx = split[split == 'train']
                val_idx = split[split == 'val']
                test_idx = split[split == 'test']
                logger.debug(f"Fold {i},pct_split {pct_split},random_seed {random_seed}:")
                logger.debug(f"train len = {len(train_idx)}")
                logger.debug(f"val len = {len(val_idx)}")
                logger.debug(f"test len = {len(test_idx)}")
                logger.debug(f'{len(train_idx) + len(val_idx) +len(test_idx)}/{experiment.dataset.adata.n_obs} cells total')
                logger.debug(f'idx intersection : {set(train_idx) & set(val_idx) & set(test_idx)}')

                experiment.task = f"task_2_{model}_{i}_{pct_split}_{random_seed}"
                logger.debug(f'Running run id : {experiment.task} for {experiment.dataset_name} and {model}')
                logger.debug(f'Running {model}')
                experiment.train_model(model)
                experiment.compute_metrics()

    logger.debug(f'Task2_New finished for all models: {run_file.dataset_name}')
