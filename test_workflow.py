from runfile_handler import RunFile
from workflow import Workflow

working_dir = '/data/dca_permuted'

workflow_ID = None
dataset_name = 'discovair_V7'
class_key = 'celltype_lv2'
filter_min_counts = True
normalize_size_factors = True
scale_input = True
logtrans_input = True 
model_name = 'dca_permuted'
ae_type = 'zinb-conddisp'
hidden_size = 128
hidden_dropout = False
batchnorm = False
activation = 'relu'
init = 'glorot_uniform'
epochs = 100
reduce_lr = 10
early_stop = 5
batch_size = 32
optimizer = 'RMSprop'
verbose = True
threads = None
learning_rate = None
n_perm = 1
permute = True
mode = 'percentage'
pct_split = 0.8
obs_key = None
n_keep = None
keep_obs = None
train_test_random_seed = 42
obs_subsample = None
make_fake = False
true_celltype = None
false_celltype = None
pct_false = None
predictor_model = 'MLP'
predict_key = 'celltype' 
predictor_hidden_sizes = 64
predictor_epochs = 50
predictor_batch_size = 32

rf = RunFile(working_dir = working_dir,
             workflow_ID = workflow_ID,
            dataset_name = dataset_name,
            class_key = class_key,
            filter_min_counts = filter_min_counts,
            normalize_size_factors = normalize_size_factors,
            scale_input = scale_input,
            logtrans_input = logtrans_input ,
            model_name = model_name,
            ae_type = ae_type,
            hidden_size = hidden_size,
            hidden_dropout = hidden_dropout,
            batchnorm = batchnorm,
            activation = activation,
            init = init,
            epochs = epochs,
            reduce_lr = reduce_lr,
            early_stop = early_stop,
            batch_size = batch_size,
            optimizer = optimizer,
            verbose = verbose,
            threads = threads,
            learning_rate = learning_rate,
            n_perm = n_perm,
            permute = permute,
            mode = mode,
            pct_split = pct_split,
            obs_key = obs_key,
            n_keep = n_keep,
            keep_obs = keep_obs,
            train_test_random_seed = train_test_random_seed,
            obs_subsample = obs_subsample,
            make_fake = make_fake,
            true_celltype = true_celltype,
            false_celltype = false_celltype,
            pct_false = pct_false,
            predictor_model = predictor_model ,
            predict_key = predict_key ,
            predictor_hidden_sizes = predictor_hidden_sizes,
            predictor_epochs = predictor_epochs,
            predictor_batch_size = predictor_batch_size)

rf.create_runfile()

workflow = Workflow(yaml_name = rf.run_file_path, working_dir = working_dir)
workflow.make_experiment()
workflow.save_results()
