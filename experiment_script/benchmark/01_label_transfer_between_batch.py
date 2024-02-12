import argparse
import sys
from sklearn.model_selection import GroupKFold

WD_PATH = '/home/acollin/dca_permuted_workflow/'
sys.path.append(WD_PATH)

from workflow.utils import str2bool
from workflow.workflow_benchmark import Workflow

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

    run_file = parser.parse_args()
    print(run_file.class_key, run_file.batch_key)
    working_dir = '/home/acollin/dca_permuted_workflow/'
    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    experiment.process_dataset()

    experiment.mode = "entire_condition"

    kf = GroupKFold(5)
    for i, (train_index, val_index) in enumerate(kf.split(experiment.dataset.adata_train_extended.X, experiment.dataset.adata_train_extended.obs[experiment.class_key], experiment.dataset.adata_train_extended.obs[experiment.batch_key])):
        for model in ['pca_svm','harmony_svm', 'scanvi']:
            print(i)
            experiment.keep_obs = list(experiment.dataset.adata_train_extended.obs[experiment.batch_key][train_index].unique()) # keeping only train idx
            experiment.split_train_test_val()
            experiment.train_model(model)
            experiment.compute_metrics()
