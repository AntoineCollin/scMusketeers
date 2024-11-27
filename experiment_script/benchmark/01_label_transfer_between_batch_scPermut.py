import argparse
from ast import Expression
import sys
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import neptune

WD_PATH = '/home/acollin/scPermut/'
sys.path.append(WD_PATH)

from scpermut.tools.utils import str2bool, load_json
print(str2bool('True'))
from scpermut.workflow.hyperparameters import Workflow

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--working_dir', type=str, nargs='?', default='', help ='The working directory')
    
    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument('--dataset_name', type = str, default = 'htap_final_by_batch', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--class_key', type = str, default = 'celltype', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'donor', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')# TODO :remove, we always want to do that
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')
    parser.add_argument('--size_factor', type=str, nargs='?',const='default', default='default', help ='Which size factor to use. "default" computes size factor on the chosen level of preprocessing. "raw" uses size factor computed on raw data as n_counts/median(n_counts). "constant" uses a size factor of 1 for every cells')
    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=5000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")
    # parser.add_argument('--reduce_lr', type = , default = , help ='')
    # parser.add_argument('--early_stop', type = , default = , help ='')
    parser.add_argument('--batch_size', type = int, nargs='?', default = 128, help ='Training batch size') # Default identified with hp optimization
    # parser.add_argument('--verbose', type = , default = , help ='')
    # parser.add_argument('--threads', type = , default = , help ='')
    parser.add_argument('--test_split_key', type = str, default = 'TRAIN_TEST_split', help ='key of obs containing the test split')
    parser.add_argument('--test_obs', type = str,nargs='+', default = None, help ='batches from batch_key to use as test')
    parser.add_argument('--test_index_name', type = str,nargs='+', default = None, help ='indexes to be used as test. Overwrites test_obs')

    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float, nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str, nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int, nargs='?', default = None, help ='batches from obs_key to use as train')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--make_fake', type=str2bool, nargs='?',const=False, default=False, help ='')
    parser.add_argument('--true_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--false_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--pct_false', type = float,nargs='?', default = None, help ='')
    parser.add_argument('--clas_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy', 'categorical_focal_crossentropy'], default = 'categorical_crossentropy' , help ='Loss of the classification branch')
    parser.add_argument('--balance_classes', type=str2bool, nargs='?',const=True, default=True , help ='Wether to weight the classification loss by inverse of classes weight')
    parser.add_argument('--dann_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default ='categorical_crossentropy', help ='Loss of the DANN branch')
    parser.add_argument('--rec_loss_name', type = str,nargs='?', choices = ['MSE'], default ='MSE', help ='Reconstruction loss of the autoencoder')
    parser.add_argument('--weight_decay', type = float,nargs='?', default = 2e-6, help ='Weight decay applied by th optimizer') # Default identified with hp optimization
    parser.add_argument('--learning_rate', type = float,nargs='?', default = 0.001, help ='Starting learning rate for training')# Default identified with hp optimization
    parser.add_argument('--optimizer_type', type = str, nargs='?',choices = ['adam','adamw','rmsprop'], default = 'adam' , help ='Name of the optimizer to use')
    parser.add_argument('--clas_w', type = float,nargs='?', default = 0.1, help ='Weight of the classification loss')
    parser.add_argument('--dann_w', type = float,nargs='?', default = 0.1, help ='Weight of the DANN loss')
    parser.add_argument('--rec_w', type = float,nargs='?', default = 0.8, help ='Weight of the reconstruction loss')
    parser.add_argument('--warmup_epoch', type = int,nargs='?', default = 50, help ='Number of epoch to warmup DANN')

    parser.add_argument('--dropout', type=float,nargs='?', default = None, help ='dropout applied to every layers of the model. If specified, overwrites other dropout arguments')
    parser.add_argument('--layer1', type=int,nargs='?', default = None, help ='size of the first layer for a 2-layers model. If specified, overwrites ae_hidden_size')
    parser.add_argument('--layer2', type=int,nargs='?', default = None, help ='size of the second layer for a 2-layers model. If specified, overwrites ae_hidden_size')
    parser.add_argument('--bottleneck', type=int,nargs='?', default = None, help ='size of the bottleneck layer. If specified, overwrites ae_hidden_size')

    parser.add_argument('--ae_hidden_size', type = int,nargs='+', default = [128,64,128], help ='Hidden sizes of the successive ae layers')
    parser.add_argument('--ae_hidden_dropout', type =float, nargs='?', default = None, help ='')
    parser.add_argument('--ae_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--ae_bottleneck_activation', type = str ,nargs='?', default = 'linear' , help ='activation of the bottleneck layer')
    parser.add_argument('--ae_output_activation', type = str,nargs='?', default = 'relu', help ='')
    parser.add_argument('--ae_init', type = str,nargs='?', default = 'glorot_uniform', help ='')
    parser.add_argument('--ae_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--ae_l1_enc_coef', type = float,nargs='?', default = None, help ='')
    parser.add_argument('--ae_l2_enc_coef', type = float,nargs='?', default = None, help ='')
    parser.add_argument('--class_hidden_size', type = int,nargs='+', default = [64], help ='Hidden sizes of the successive classification layers')
    parser.add_argument('--class_hidden_dropout', type =float, nargs='?', default = None, help ='')
    parser.add_argument('--class_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--class_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--class_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--dann_hidden_size', type = int,nargs='?', default = [64], help ='')
    parser.add_argument('--dann_hidden_dropout', type =float, nargs='?', default = None, help ='')
    parser.add_argument('--dann_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--dann_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--dann_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--training_scheme', type = str,nargs='?', default = 'training_scheme_1', help ='')
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--hparam_path', type=str, nargs='?', default=None, help ='')
    parser.add_argument('--opt_metric', type=str, nargs='?', default='val-balanced_acc', help ='The metric used for early stopping as well as optimizes in hp search. Should be formatted as it appears in neptune (split-metricname)')

    run_file = parser.parse_args()
    print(run_file.class_key, run_file.batch_key)
    working_dir = run_file.working_dir
    print(f'working directory : {working_dir}')

    project = neptune.init_project(
            project="becavin-lab/benchmark",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        mode="read-only",
            )# For checkpoint
    
    runs_table_df = project.fetch_runs_table(query = '`parameters/task`:string = "task_1"').to_pandas()# , columns = ['parameters/dataset_name',
                            # 'parameters/training_scheme',
                            # 'parameters/clas_loss_name',
                            # 'parameters/use_hvg',
                            # 'parameters/task',
                            # 'parameters/model',
                            # 'parameters/test_fold_nb',
                            # 'parameters/val_fold_nb',
                            # 'parameters/deprecated_status',
                            # 'parameters/debug_status']).to_pandas()
    print('run table : ')
    print(runs_table_df.shape)
    print(runs_table_df['parameters/debug_status'].head())
    project.stop()
    
    experiment = Workflow(run_file=run_file, working_dir=working_dir)

    experiment.process_dataset()

    experiment.mode = "entire_condition"

    random_seed = 2
    model = 'scPermut'
    
    X = experiment.dataset.adata.X
    classes = experiment.dataset.adata.obs[experiment.class_key]
    groups = experiment.dataset.adata.obs[experiment.batch_key]

    n_batches = len(groups.unique())
    nfold_test = max(1,round(n_batches/5)) # if less than 8 batches, this comes to 1 batch per fold, otherwise, 20% of the number of batches for test
    kf_test = GroupShuffleSplit(n_splits=3, test_size=nfold_test, random_state=random_seed)
    test_split_key = experiment.dataset.test_split_key

    test_obs_json = load_json(working_dir + 'experiment_script/benchmark/hp_test_obs.json')
    fixed_test_obs = test_obs_json[run_file.dataset_name]

    for i, (train_index, test_index) in enumerate(kf_test.split(X, classes, groups)):
        test_obs = list(groups.iloc[test_index].unique()) # the batches that go in the test set, old

        if set(test_obs) == set(fixed_test_obs):
            experiment.test_obs = test_obs
            experiment.split_train_test()
            # experiment.dataset.test_split(test_obs = test_obs) # splits the train and test dataset
            nfold_val = max(1,round((n_batches-len(test_obs))/5)) # represents 20% of the remaining train set
            kf_val = GroupShuffleSplit(n_splits=5, test_size=nfold_val, random_state=random_seed)

            X_train_val = experiment.dataset.adata_train_extended.X
            classes_train_val = experiment.dataset.adata_train_extended.obs[experiment.class_key]
            groups_train_val = experiment.dataset.adata_train_extended.obs[experiment.batch_key]

            for j, (train_index, val_index) in enumerate(kf_val.split(X_train_val, classes_train_val, groups_train_val)):
                groups_sub = groups_train_val
                experiment.keep_obs = list(groups_train_val[train_index].unique()) # keeping only train idx
                val_obs = list(groups_train_val[val_index].unique())

                print(f"Fold {i,j}:")
                print(f"train = {list(groups_train_val.iloc[train_index].unique())}, len = {len(groups_train_val.iloc[train_index].unique())}")
                print(f"val = {list(groups_train_val.iloc[val_index].unique())}, len = {len(groups_train_val.iloc[val_index].unique())}")
                print(f"test = {list(groups.iloc[test_index].unique())}, len = {len(groups.iloc[test_index].unique())}")

                
                print(set(groups_train_val.iloc[train_index].unique()) & set(groups.iloc[test_index].unique()))
                print(set(groups_train_val.iloc[train_index].unique()) & set(groups_train_val.iloc[val_index].unique()))
                print(set(groups_train_val.iloc[val_index].unique()) & set(groups.iloc[test_index].unique()))

                experiment.split_train_val()
                checkpoint={'parameters/dataset_name': experiment.dataset_name, 
                            'parameters/training_scheme': experiment.training_scheme,
                            'parameters/clas_loss_name': experiment.clas_loss_name,
                            'parameters/use_hvg': experiment.use_hvg,
                            'parameters/task': 'task_1',
                            'parameters/model': model, 
                            'parameters/test_fold_nb':i,
                            'parameters/val_fold_nb':j,
                            'parameters/deprecated_status': False,
                            'parameters/debug_status': 'fixed_1'}
                # for k, v in vars(run_file).items():
                #     checkpoint['parameters/' + k] = v
                print(f'checkpoint : {checkpoint}')
                result = runs_table_df[runs_table_df[list(checkpoint.keys())].eq(list(checkpoint.values())).all(axis=1)]
                if result.empty:
                    print(f'Running {model}')
                    experiment.start_neptune_log()
                    experiment.make_experiment()
                    experiment.add_custom_log('test_fold_nb',i)
                    experiment.add_custom_log('val_fold_nb',j)
                    experiment.add_custom_log('test_obs',test_obs)
                    experiment.add_custom_log('val_obs',val_obs)
                    experiment.add_custom_log('train_obs',experiment.keep_obs)
                    experiment.add_custom_log('task','task_1')
                    experiment.add_custom_log('deprecated_status', False)
                    experiment.add_custom_log('debug_status', "fixed_1")
                    experiment.stop_neptune_log()
