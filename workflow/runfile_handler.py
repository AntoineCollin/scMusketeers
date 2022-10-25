import pandas as pd
import yaml
import nbformat as nbf
from jinja2 import Environment, FileSystemLoader
import numpy as np
import os
import itertools
import shutil
try :
    from .load import load_runfile
except ImportError:
    from load import load_runfile

class RunFile:
    def __init__(self,working_dir,
                      workflow_ID = None,
                       dataset = "dataset",
                       dataset_name = None,
                       class_key = None,
                       dataset_normalize = 'dataset_normalize',
                       filter_min_counts = None,
                       normalize_size_factors = None,
                       scale_input = None,
                       logtrans_input = None ,
                       model_spec = 'model_spec',
                       model_name = None,
                       ae_type = None,
                       hidden_size = None,
                       hidden_dropout = None,
                       batchnorm = None,
                       activation = None,
                       init = None,
                       model_training_spec = 'model_training_spec',
                       epochs = None,
                       reduce_lr = None,
                       early_stop = None,
                       batch_size = None,
                       optimizer = None,
                       verbose = None,
                       threads = None,
                       learning_rate = None,
                       n_perm = 1,
                       permute = True,
                       change_perm = True,
                       semi_sup = False,
                       unlabeled_category = None,
                       save_zinb_param = False,
                       use_raw_as_output = False,
                       contrastive_margin = None,
                       same_class_pct = None,
                       dataset_train_split = 'dataset_train_split',
                       use_TEST = True,
                       mode = None,
                       pct_split = None,
                       obs_key = None,
                       n_keep = None,
                       keep_obs = None,
                       train_test_random_seed = None,
                       obs_subsample = None,
                       dataset_fake_annotation = 'dataset_fake_annotation',
                       make_fake = None,
                       true_celltype = None,
                       false_celltype = None,
                       pct_false = None,
                       predictor_spec = 'predictor_spec',
                       predictor_model = None ,
                       predict_key = None ,
                       predictor_hidden_sizes = None,
                       predictor_epochs = None,
                       predictor_batch_size = None,
                       predictor_activation = None):
        self.runfile_dict = locals()
        del self.runfile_dict['self']
        del self.runfile_dict['working_dir']
        del self.runfile_dict['dataset']
        del self.runfile_dict['dataset_normalize']
        del self.runfile_dict['model_spec']
        del self.runfile_dict['model_training_spec']
        del self.runfile_dict['dataset_train_split']
        del self.runfile_dict['dataset_fake_annotation']
        del self.runfile_dict['predictor_spec']
        self.workflow_ID = workflow_ID
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.class_key = class_key
        self.dataset_normalize = dataset_normalize
        self.filter_min_counts = filter_min_counts
        self.normalize_size_factors = normalize_size_factors
        self.scale_input = scale_input
        self.logtrans_input = logtrans_input
        self.model_spec = model_spec
        self.model_name = model_name
        self.ae_type = ae_type
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.model_training_spec = model_training_spec
        self.epochs = epochs
        self.reduce_lr = reduce_lr
        self.early_stop = early_stop
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.verbose = verbose
        self.threads = threads
        self.learning_rate = learning_rate
        self.n_perm = n_perm
        self.permute = permute
        self.change_perm = change_perm
        self.semi_sup = semi_sup
        self.unlabeled_category = unlabeled_category
        self.save_zinb_param = save_zinb_param
        self.use_raw_as_output = use_raw_as_output
        self.contrastive_margin = contrastive_margin
        self.same_class_pct = same_class_pct
        self.dataset_train_split = dataset_train_split
        self.use_TEST = use_TEST
        self.mode = mode
        self.pct_split = pct_split
        self.obs_key = obs_key
        self.n_keep = n_keep
        self.keep_obs = keep_obs
        self.train_test_random_seed = train_test_random_seed
        self.obs_subsample = obs_subsample
        self.dataset_fake_annotation = dataset_fake_annotation
        self.make_fake = make_fake
        self.true_celltype = true_celltype
        self.false_celltype = false_celltype
        self.pct_false = pct_false
        self.predictor_spec = predictor_spec
        self.predictor_model = predictor_model
        self.predict_key = predict_key
        self.predictor_hidden_sizes = predictor_hidden_sizes
        self.predictor_epochs = predictor_epochs
        self.predictor_batch_size = predictor_batch_size
        self.predictor_activation = predictor_activation
        
        self.working_dir = working_dir
        self.runfile_dir = self.working_dir + '/runfile_dir'
        self.runfile_template_name = 'runfile_template.yaml'
        self.runfile_csv_path = self.runfile_dir + '/runfile_list.csv'
        self.runfile_df = pd.read_csv(self.runfile_csv_path, index_col = 'index')
        self.metric_results_path = self.working_dir + '/results/metric_results.csv'
        self.metric_results_df = pd.read_csv(self.metric_results_path, index_col = 'index')        
        self.run_file_path = self.runfile_dir + f"/runfile_ID_{self.runfile_dict['workflow_ID']}.yaml"
        
    def make_runfile(self):
        file_loader = FileSystemLoader(self.runfile_dir)
        env = Environment(loader=file_loader)
        template = env.get_template(self.runfile_template_name)
        template.render(**self.runfile_dict)
        file=open(self.run_file_path, "w")
        file.write(template.render(**self.runfile_dict))
        file.close()

    def check_runfile_existance(self):
        '''
        checks if runfile already exists.
        - if it does, prints a warning and returns None
        - if it doesn't, computes a new id which is calculated to be the lowest integers which fills in any gap between two IDs (ex : if we have id 1,2,3,6 the chosen id will be 4). If there is no gap, the ID will be the max ID +1
        
        '''
        self.runfile_df = pd.read_csv(self.runfile_csv_path, index_col = 'index')
        if self.workflow_ID in self.runfile_df:
            print('ID already exists, please choose another ID.')
            return
        runfile_csv_filtered = self.runfile_df.copy()
        for key, value in self.runfile_dict.items():
            if key != 'workflow_ID':
                if type(value)== str:
                    query = f'{key} == "{value}"'
                elif value is None:
                    query = f'{key} != {key}'
                else:
                    query =  f'{key} == {value}'
                runfile_csv_filtered = runfile_csv_filtered.query(query)
        if runfile_csv_filtered.empty :
            ID_list = np.array(self.runfile_df['workflow_ID'].sort_values(), dtype = int)
            for i in range(self.runfile_df['workflow_ID'].shape[0]-1):
                if ID_list[i+1] - ID_list[i] >1.5:
                    new_id = ID_list[i] + 1
                    return new_id
            new_id = max(ID_list) + 1
            return new_id 
        if runfile_csv_filtered.shape[0] == 1:
            print('runfile already exist and has ID' + str(runfile_csv_filtered["workflow_ID"].iloc[0]))
            self.workflow_ID = runfile_csv_filtered["workflow_ID"].iloc[0]
            self.runfile_dict['workflow_ID'] = self.workflow_ID
            self.run_file_path = self.runfile_dir + f"/runfile_ID_{self.runfile_dict['workflow_ID']}.yaml"
            return 
        
    def add_entry_csv(self):
        self.runfile_df = pd.read_csv(self.runfile_csv_path, index_col = 'index')
        self.metric_results_df = pd.read_csv(self.metric_results_path, index_col = 'index')
        dict_to_pandas = {}
        for key, value in self.runfile_dict.items():
            dict_to_pandas[key] = [value]
        df_entry = pd.DataFrame.from_dict(dict_to_pandas)
        df_entry.index = [self.runfile_dict['workflow_ID']]
        df_entry.index.name = 'index'
        self.runfile_df = pd.concat([self.runfile_df, df_entry])
        self.metric_results_df = pd.concat([self.metric_results_df, df_entry])
        self.runfile_df.to_csv(self.runfile_csv_path)
        self.metric_results_df.to_csv(self.metric_results_path)
        
    def create_runfile(self, overwrite = False):
        if overwrite :
            if not self.workflow_ID:
                print("You are in overwrite mode, please specify a workflow ID to overwrite")
            else :
                self.run_file_path = self.runfile_dir + f"/runfile_ID_{self.runfile_dict['workflow_ID']}.yaml"
                self.make_runfile()
                return
        new_id = self.check_runfile_existance() # check if the file already exists
        if not new_id :
            print('Creation fail because runfile already exists')
            return
        if self.workflow_ID: # if an ID is given we keep it
            del new_id
        elif not self.workflow_ID: # otherwise, we take the proposed id
            self.workflow_ID = new_id
            self.runfile_dict['workflow_ID'] = new_id
        print(f'Creating runfile with ID {self.runfile_dict["workflow_ID"]}')
        self.run_file_path = self.runfile_dir + f"/runfile_ID_{self.runfile_dict['workflow_ID']}.yaml"
        self.make_runfile()
        self.add_entry_csv()
        

########################################################################################################
        
def read_from_ID(working_dir, workflow_ID):
    if not workflow_ID:
        print("Please specify an ID")
    else :
        yaml_name = working_dir + '/runfile_dir/runfile_ID_' + str(workflow_ID) + '.yaml'
        run_file = load_runfile(yaml_name)

        workflow_ID = 'workflow_ID'

        dataset = 'dataset'
        dataset_name = 'dataset_name'
        class_key = 'class_key'

        dataset_normalize = 'dataset_normalize'
        filter_min_counts = 'filter_min_counts'
        normalize_size_factors = 'normalize_size_factors'
        scale_input = 'scale_input'
        logtrans_input = 'logtrans_input'

        model_spec = 'model_spec'
        model_name = 'model_name'
        ae_type = 'ae_type'
        hidden_size = 'hidden_size'
        hidden_dropout = 'hidden_dropout'
        batchnorm = 'batchnorm'
        activation = 'activation'
        init = 'init'

        model_training_spec = 'model_training_spec'
        epochs = 'epochs'
        reduce_lr = 'reduce_lr'
        early_stop = 'early_stop'
        batch_size = 'batch_size'
        optimizer = 'optimizer'
        verbose = 'verbose'
        threads = 'threads'
        learning_rate = 'learning_rate'
        n_perm = 'n_perm'
        permute = 'permute'
        change_perm = 'change_perm'
        semi_sup = 'semi_sup'
        unlabeled_category = 'unlabeled_category'
        save_zinb_param = 'save_zinb_param'
        use_raw_as_output = 'use_raw_as_output'
        contrastive_margin = 'contrastive_margin'
        same_class_pct = 'same_class_pct'
        
        dataset_train_split = 'dataset_train_split'
        mode = 'mode'
        pct_split = 'pct_split'
        obs_key = 'obs_key'
        n_keep = 'n_keep'
        keep_obs = 'keep_obs'
        train_test_random_seed = 'train_test_random_seed'
        use_TEST = 'use_TEST'
        obs_subsample = 'obs_subsample'

        dataset_fake_annotation = 'dataset_fake_annotation'
        make_fake = 'make_fake'
        true_celltype = 'true_celltype'
        false_celltype = 'false_celltype'
        pct_false = 'pct_false'

        predictor_spec = 'predictor_spec'
        predictor_model = 'predictor_model' 
        predict_key = 'predict_key' 
        predictor_hidden_sizes = 'predictor_hidden_sizes'
        predictor_epochs = 'predictor_epochs'
        predictor_batch_size = 'predictor_batch_size'
        predictor_activation = 'predictor_activation'

        workflow_ID = run_file[workflow_ID]
        # dataset identifiers
        dataset_name = run_file[dataset][dataset_name]
        class_key = run_file[dataset][class_key]
        # normalization parameters
        filter_min_counts = run_file[dataset_normalize][filter_min_counts]
        normalize_size_factors = run_file[dataset_normalize][normalize_size_factors]
        scale_input = run_file[dataset_normalize][scale_input]
        logtrans_input = run_file[dataset_normalize][logtrans_input]
        # model parameters
        model_name = run_file[model_spec][model_name]
        ae_type = run_file[model_spec][ae_type]
        hidden_size = run_file[model_spec][hidden_size]
        hidden_dropout = run_file[model_spec][hidden_dropout]
        batchnorm = run_file[model_spec][batchnorm]
        activation = run_file[model_spec][activation]
        init = run_file[model_spec][init]
        # model training parameters
        epochs = run_file[model_training_spec][epochs]
        reduce_lr = run_file[model_training_spec][reduce_lr]
        early_stop = run_file[model_training_spec][early_stop]
        batch_size = run_file[model_training_spec][batch_size]
        optimizer = run_file[model_training_spec][optimizer]
        verbose = run_file[model_training_spec][verbose]
        threads = run_file[model_training_spec][threads]
        learning_rate = run_file[model_training_spec][learning_rate]
        n_perm = run_file[model_training_spec][n_perm]
        permute = run_file[model_training_spec][permute]
        change_perm = run_file[model_training_spec][change_perm]
        unlabeled_category = run_file[model_training_spec][unlabeled_category]
        save_zinb_param = run_file[model_training_spec][save_zinb_param]
        use_raw_as_output = run_file[model_training_spec][use_raw_as_output]
        # train test split
        mode = run_file[dataset_train_split][mode]
        pct_split = run_file[dataset_train_split][pct_split]
        obs_key = run_file[dataset_train_split][obs_key]
        n_keep = run_file[dataset_train_split][n_keep]
        keep_obs = run_file[dataset_train_split][keep_obs]
        train_test_random_seed = run_file[dataset_train_split][train_test_random_seed]
        use_TEST = run_file[dataset_train_split][use_TEST]
        obs_subsample = run_file[dataset_train_split][obs_subsample]
        # Create fake annotations
        make_fake = run_file[dataset_fake_annotation][make_fake]
        true_celltype = run_file[dataset_fake_annotation][true_celltype]
        false_celltype = run_file[dataset_fake_annotation][false_celltype]
        pct_false = run_file[dataset_fake_annotation][pct_false]
        # predictor parameters
        predictor_model = run_file[predictor_spec][predictor_model]
        predict_key = run_file[predictor_spec][predict_key]
        predictor_hidden_sizes = run_file[predictor_spec][predictor_hidden_sizes]
        predictor_epochs = run_file[predictor_spec][predictor_epochs]
        predictor_batch_size = run_file[predictor_spec][predictor_batch_size]
        predictor_activation = run_file[predictor_spec][predictor_activation]
        rf_dict = locals()
        del rf_dict['run_file']
        del rf_dict['yaml_name']
        return rf_dict
            
    
    ##############################################################################
    
    
class run_file_handler:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.runfile_dir = self.working_dir + '/runfile_dir'
        self.runfile_template_name = 'runfile_template.yaml'
        self.runfile_csv_path = self.working_dir + '/runfile_dir/runfile_list.csv'
        self.runfile_df = pd.read_csv(self.runfile_csv_path, index_col = 'index')        
        self.metric_results_path = self.working_dir + '/results/metric_results.csv'
        self.metric_results_df = pd.read_csv(self.metric_results_path, index_col = 'index')        

        
    def generate_runfiles(self,
                          dataset = [None],
                          dataset_name = [None],
                          class_key = [None],
                          dataset_normalize = [None],
                          filter_min_counts = [None],
                          normalize_size_factors = [None],
                          scale_input = [None],
                          logtrans_input = [None] ,
                          model_spec = [None],
                          model_name = [None],
                          ae_type = [None],
                          hidden_size = [None],
                          hidden_dropout = [None],
                          batchnorm = [None],
                          activation = [None],
                          init = [None],
                          model_training_spec = [None],
                          epochs = [None],
                          reduce_lr = [None],
                          early_stop = [None],
                          batch_size = [None],
                          optimizer = [None],
                          verbose = [None],
                          threads = [None],
                          learning_rate = [None],
                          n_perm = [None],                        
                          permute = [None],
                          change_perm = [None],
                          semi_sup = [None],
                          unlabeled_category = [None],
                          save_zinb_param = [None],
                          use_raw_as_output = [None],
                          contrastive_margin = [None],
                          same_class_pct = [None],
                          dataset_train_split = [None],
                          mode = [None],
                          pct_split = [None],
                          obs_key = [None],
                          n_keep = [None],
                          keep_obs = [None],
                          train_test_random_seed = [None],
                          use_TEST = [None],
                          obs_subsample = [None],
                          dataset_fake_annotation = [None],
                          make_fake = [None],
                          true_celltype = [None],
                          false_celltype = [None],
                          pct_false = [None],
                          predictor_spec = [None],
                          predictor_model = [None] ,
                          predict_key = [None] ,
                          predictor_hidden_sizes = [None],
                          predictor_epochs = [None],
                          predictor_batch_size = [None],        
                          predictor_activation = [None]):
        
        '''
        Takes lists of parameters for arguments and creates a runfile for every possible combination
        '''
        arg_dict = locals()
        del arg_dict['self']
        keys = list(arg_dict)
        for key, value in arg_dict.items():
            if type(value) != list:
                arg_dict[key] = [value]
        arg_list = []
        for value in itertools.product(*map(arg_dict.get, keys)):
            param_dico = dict(zip(keys, value))
            param_dico['workflow_ID'] = None
            arg_list.append(param_dico)
        for params in arg_list:
            rf = RunFile(self.working_dir, **params)
            rf.create_runfile()
            self.runfile_df = pd.read_csv(self.runfile_csv_path, index_col = 'index')
            self.metric_results_df = pd.read_csv(self.metric_results_path, index_col = 'index')

            
            
    def query_yaml(self,**kwargs):
        '''
        returns the id matching the queried parameters
        '''
        print(kwargs.items())
        runfile_csv_filtered = self.runfile_df.copy()
        for key, value in kwargs.items():
            if key != 'workflow_ID':
                if type(value) == str:
                    query = f'{key} == "{value}"'
                elif value is None:
                    query = f'{key} != {key}'
                else:
                    query =  f'{key} == {value}'
                runfile_csv_filtered = runfile_csv_filtered.query(query)
        queried_ID = list(runfile_csv_filtered['workflow_ID'])
        print(f'The IDs corresponding to this query are {", ".join([str(ID) for ID in queried_ID]) }')
        return queried_ID

    def remove_experiment(self, IDs_to_delete) -> None:
        if input('Please confirm your action (y)') == 'y':
            for ID in IDs_to_delete:
                rf_file = self.runfile_dir + f"/runfile_ID_{ID}.yaml"
                if os.path.exists(rf_file):
                    # removing the file using the os.workflow_ID() method
                    os.remove(rf_file)
                results_path = self.working_dir + f'/results/result_ID_{ID}'
                if os.path.exists(results_path):
                    # checking whether the folder is empty or not
                    shutil.rmtree(results_path)
                run_log_dir = self.working_dir + '/logs/run'
                run_log_path = run_log_dir + f'/workflow_ID_{ID}_DONE.txt'
                predicts_log_dir = self.working_dir + '/logs/predicts'
                predicts_log_path = predicts_log_dir + f'/workflow_ID_{ID}_DONE.txt'
                umap_log_dir = self.working_dir + '/logs/umap'
                umap_log_path = umap_log_dir + f'/workflow_ID_{ID}_DONE.txt'
                if os.path.exists(run_log_path):
                    # removing the file using the os.remove() method
                    os.remove(run_log_path)
                if os.path.exists(predicts_log_path):
                    # removing the file using the os.remove() method
                    os.remove(predicts_log_path)
                if os.path.exists(umap_log_path):
                    # removing the file using the os.remove() method
                    os.remove(umap_log_path)
                self.runfile_df = self.runfile_df[self.runfile_df.workflow_ID != int(ID)]
                self.runfile_df.to_csv(self.runfile_csv_path)
                self.metric_results_df = self.metric_results_df[self.metric_results_df.workflow_ID != int(ID)]
                self.metric_results_df.to_csv(self.metric_results_path)

            
    def rerun_experiment(self, IDs_to_rerun):
        '''
        Only deletes the output of the workflow, keeps the runfile. Use to relaunch an experiment after changing the code.
        '''
        if input('Please confirm your action (y)') == 'y':
            for ID in IDs_to_rerun:
                results_path = self.working_dir + f'/results/result_ID_{ID}'
                if os.path.exists(results_path):
                    # checking whether the folder is empty or not
                    shutil.rmtree(results_path)
                run_log_dir = self.working_dir + '/logs/run'
                run_log_path = run_log_dir + f'/workflow_ID_{ID}_DONE.txt'
                predicts_log_dir = self.working_dir + '/logs/predicts'
                predicts_log_path = predicts_log_dir + f'/workflow_ID_{ID}_DONE.txt'
                umap_log_dir = self.working_dir + '/logs/umap'
                umap_log_path = umap_log_dir + f'/workflow_ID_{ID}_DONE.txt'
                if os.path.exists(run_log_path):
                    # removing the file using the os.remove() method
                    os.remove(run_log_path)
                if os.path.exists(predicts_log_path):
                    # removing the file using the os.remove() method
                    os.remove(predicts_log_path)
                if os.path.exists(umap_log_path):
                    # removing the file using the os.remove() method
                    os.remove(umap_log_path)
                self.metric_results_df.loc[ID,:] = self.runfile_df.loc[ID,:]
                self.metric_results_df.to_csv(self.metric_results_path)
            

    def add_argument_runfile(self, argument, value = 'NA'):
        # Adding new argument to csv 
        self.runfile_df[argument] = value
        self.runfile_df.to_csv(self.runfile_csv_path)
        self.metric_results_df[argument] = value
        self.metric_results_df.to_csv(self.metric_results_path)

        # Adding new argument = NA for existing runfiles
        for wf_id in self.runfile_df['workflow_ID']:
            runfile_kwargs = read_from_ID(self.working_dir, wf_id)
            runfile_kwargs[argument] = value
            rf = RunFile(**runfile_kwargs)
            rf.create_runfile(overwrite=True)
            
#     def add_entry_csv(self,)
        
        
    


#     def parse_runfile(self)
        
        
