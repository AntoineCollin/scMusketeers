import yaml

def load_runfile(runfile_path):
    workflow_ID= 1
    yaml_config_dict= {'dataset':'dataset', #keys are the name in the runfile, values are the "official" keys
                     'dataset_name':'dataset_name',
                     'filter_min_counts':'filter_min_counts',
                     'normalize_size_factors':'normalize_size_factors',
                     'scale_input':'scale_input',
                     'logtrans_input':'logtrans_input',
                     'class_key':'class_key',
                     'model_spec':'model_spec',
                     'ae_type':'ae_type',
                     'hidden_size':'hidden_size',
                     'hidden_dropout':'hidden_dropout',
                     'batchnorm':'batchnorm',
                     'activation':'activation',
                     'init':'init',
                     'training_spec':'training_spec',
                     'epochs':'epochs',
                     'reduce_lr':'reduce_lr',
                     'early_stop':'early_stop',
                     'batch_size':'batch_size',
                     'optimizer':'optimizer',
                     'verbose':'verbose',
                     'threads':'threads',
                     'learning_rate':'learning_rate',
                     'n_perm':'n_perm',
                     'permute':'permute',
                     'semi_sup':'semi_sup',
                     'unlabelled_category':'unlabelled_category',
                     'train_split':'train_split',
                     'mode':'mode',
                     'pct_split':'pct_split',
                     'obs_key':'obs_key',
                     'n_keep':'n_keep',
                     'keep_obs':'keep_obs',
                     'random_seed':'random_seed',
                     'obs_subsample':'obs_subsample',
                     'train_test_random_seed':'train_test_random_seed',
                     'fake_annotation':'fake_annotation',
                     'true_celltype':'true_celltype',
                     'false_celltype':'false_celltype',
                     'pct_false':'pct_false',
                     'predictor_activation':'predictor_activation'}
    yml = open(runfile_path)
    parsed_yml = yaml.load(yml, Loader=yaml.FullLoader)
    n_hidden = parsed_yml['model_spec']['hidden_size'] 
    for key in parsed_yml.keys():
        if type(parsed_yml[key]) == dict:
            for ki in parsed_yml[key].keys():
                if parsed_yml[key][ki] == 'None':
                    parsed_yml[key][ki] = None
                if type(parsed_yml[key][ki]) == str and '(' in parsed_yml[key][ki]: # Tuples are parsed as string, we convert them back to tuples
                    parsed_yml[key][ki] = tuple([int(i) for i in parsed_yml[key][ki].strip('(').strip(')').replace(' ', '').split(',')])
    return parsed_yml

