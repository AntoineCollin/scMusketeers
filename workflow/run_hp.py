import argparse
import subprocess
from ax.service.managed_loop import optimize
import os
import gc
import time
from workflow_hp import *
import tensorflow as tf

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class MakeExperiment:
    def __init__(self, run_file, working_dir):
        # super().__init__()
        self.run_file = run_file
        self.working_dir = working_dir
        self.workflow = None

    def train(self, params):
        # cuda.select_device(0)
        # device = cuda.get_current_device()
        # device.reset()
        
        self.workflow = Workflow(run_file=self.run_file, working_dir=self.working_dir)
        mcc = self.workflow.make_experiment(params)
        del self.workflow  # Should not be necessary
        return mcc


n_gpu = '01'
def train_cmd(params):
    print(params)
    run_file.ae_bottleneck_activation =  params['ae_bottleneck_activation']
    run_file.ae_activation = params['ae_activation']
    run_file.clas_w =  params['clas_w']
    run_file.dann_w = params['dann_w']
    run_file.rec_w =  1
    run_file.learning_rate = params['learning_rate']
    run_file.weight_decay =  params['weight_decay']
    run_file.warmup_epoch =  params['warmup_epoch']
    dropout =  params['dropout']
    layer1 = params['layer1']
    layer2 =  params['layer2']
    bottleneck = params['bottleneck']

    run_file.ae_hidden_size = [layer1, layer2, bottleneck, layer2, layer1]

    run_file.dann_hidden_dropout, run_file.class_hidden_dropout, run_file.ae_hidden_dropout = dropout, dropout, dropout
    
    cmd = ['sbatch', '--wait']

    # global n_gpu
    # print(n_gpu)
    # if ("hlca" in run_file.dataset_name) or ("discovair" in run_file.dataset_name) or ("htap" in run_file.dataset_name):
    #     cmd += ['--nodelist', f'gpu{n_gpu}']
    #     n_gpu = '01' if n_gpu!='01' else '03'
        
    cmd += ['/home/acollin/dca_permuted_workflow/workflow/run_workflow_cmd.sh']
    for k, v in run_file.__dict__.items():
        cmd += ([f'--{k}'])
        if type(v) == list:
            cmd += ([str(i) for i in v])
        else :
            cmd += ([str(v)])
    print(cmd)
    subprocess.Popen(cmd).wait()
    working_dir = '/home/acollin/dca_permuted_workflow/'
    with open(working_dir + '/experiment_script/mcc_res.txt', 'r') as my_file:
        mcc = float(my_file.read())
    os.remove(working_dir + '/experiment_script/mcc_res.txt')
    gc.collect()
    time.sleep(10)
    return mcc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--run_file', type = , default = , help ='')
    # parser.add_argument('--workflow_ID', type = , default = , help ='')
    parser.add_argument('--dataset_name', type = str, default = 'disco_ajrccm_downsampled', help ='Name of the dataset to use, should indicate a raw h5ad AnnData file')
    parser.add_argument('--class_key', type = str, default = 'celltype_lv2_V3', help ='Key of the class to classify')
    parser.add_argument('--batch_key', type = str, default = 'manip', help ='Key of the batches')
    parser.add_argument('--filter_min_counts', type=str2bool, nargs='?',const=True, default=True, help ='Filters genes with <1 counts')# TODO :remove, we always want to do that
    parser.add_argument('--normalize_size_factors', type=str2bool, nargs='?',const=True, default=True, help ='Weither to normalize dataset or not')
    parser.add_argument('--scale_input', type=str2bool, nargs='?',const=False, default=False, help ='Weither to scale input the count values')
    parser.add_argument('--logtrans_input', type=str2bool, nargs='?',const=True, default=True, help ='Weither to log transform count values')
    parser.add_argument('--use_hvg', type=int, nargs='?', const=10000, default=None, help = "Number of hvg to use. If no tag, don't use hvg.")
    # parser.add_argument('--reduce_lr', type = , default = , help ='')
    # parser.add_argument('--early_stop', type = , default = , help ='')
    parser.add_argument('--batch_size', type = int, nargs='?', default = 256, help = 'Training batch size')
    # parser.add_argument('--verbose', type = , default = , help ='')
    # parser.add_argument('--threads', type = , default = , help ='')
    parser.add_argument('--mode', type = str, default = 'percentage', help ='Train test split mode to be used by Dataset.train_split')
    parser.add_argument('--pct_split', type = float,nargs='?', default = 0.9, help ='')
    parser.add_argument('--obs_key', type = str,nargs='?', default = 'manip', help ='')
    parser.add_argument('--n_keep', type = int,nargs='?', default = 0, help ='')
    parser.add_argument('--split_strategy', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--keep_obs', type = str,nargs='+',default = None, help ='')
    parser.add_argument('--train_test_random_seed', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--obs_subsample', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--make_fake', type=str2bool, nargs='?',const=False, default=False, help ='')
    parser.add_argument('--true_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--false_celltype', type = str,nargs='?', default = None, help ='')
    parser.add_argument('--pct_false', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--clas_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default = 'categorical_crossentropy' , help ='Loss of the classification branch')
    parser.add_argument('--dann_loss_name', type = str,nargs='?', choices = ['categorical_crossentropy'], default ='categorical_crossentropy', help ='Loss of the DANN branch')
    parser.add_argument('--rec_loss_name', type = str,nargs='?', choices = ['MSE'], default ='MSE', help ='Reconstruction loss of the autoencoder')
    parser.add_argument('--weight_decay', type = float,nargs='?', default = 1e-4, help ='Weight decay applied by th optimizer')
    parser.add_argument('--learning_rate', type = float,nargs='?', default = 0.001, help ='Starting learning rate for training')
    parser.add_argument('--optimizer_type', type = str, nargs='?',choices = ['adam','adamw','rmsprop', 'sgd', 'adafactor'], default = 'sgd', help ='Name of the optimizer to use')
    parser.add_argument('--clas_w', type = float,nargs='?', default = 0.1, help ='Wight of the classification loss')
    parser.add_argument('--dann_w', type = float,nargs='?', default = 0.1, help ='Wight of the DANN loss')
    parser.add_argument('--rec_w', type = float,nargs='?', default = 0.8, help ='Wight of the reconstruction loss')
    parser.add_argument('--warmup_epoch', type = int,nargs='?', default = 50, help ='Wight of the reconstruction loss')
    parser.add_argument('--ae_hidden_size', type = int,nargs='+', default = [128,64,128], help ='Hidden sizes of the successive ae layers')
    parser.add_argument('--ae_hidden_dropout', type =float, nargs='?', default = 0, help ='')
    parser.add_argument('--ae_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--ae_bottleneck_activation', type = str ,nargs='?', default = 'linear' , help ='activation of the bottleneck layer')    
    parser.add_argument('--ae_output_activation', type = str,nargs='?', default = 'relu', help ='activation of the output layer. Defaults to relu since we expect values >0')
    parser.add_argument('--ae_init', type = str,nargs='?', default = 'glorot_uniform', help ='')
    parser.add_argument('--ae_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--ae_l1_enc_coef', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--ae_l2_enc_coef', type = float,nargs='?', default = 0, help ='')
    parser.add_argument('--class_hidden_size', type = int,nargs='+', default = [64], help ='Hidden sizes of the successive classification layers')
    parser.add_argument('--class_hidden_dropout', type =float, nargs='?', default = 0, help ='')
    parser.add_argument('--class_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--class_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--class_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--dann_hidden_size', type = int,nargs='?', default = [64], help ='')
    parser.add_argument('--dann_hidden_dropout', type =float, nargs='?', default = 0, help ='')
    parser.add_argument('--dann_batchnorm', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--dann_activation', type = str ,nargs='?', default = 'relu' , help ='')
    parser.add_argument('--dann_output_activation', type = str,nargs='?', default = 'softmax', help ='')
    parser.add_argument('--training_scheme', type = str,nargs='?', default = 'training_scheme_1', help ='')
    parser.add_argument('--log_neptune', type=str2bool, nargs='?',const=True, default=True , help ='')
    parser.add_argument('--workflow_id', type=str, nargs='?', default='default', help ='')
    # parser.add_argument('--epochs', type=int, nargs='?', default=100, help ='')

    run_file = parser.parse_args()
    # working_dir = '/home/acollin/dca_permuted_workflow/'
    # experiment = MakeExperiment(run_file=run_file, working_dir=working_dir)
    # workflow = Workflow(run_file=run_file, working_dir=working_dir)
    print("Workflow loaded")

    hparams = [
        #{"name": "use_hvg", "type": "range", "bounds": [5000, 10000], "log_scale": False},
        {"name": "ae_bottleneck_activation", "type": "choice", "values": ['linear', 'relu']},
        {"name": "ae_activation", "type": "choice", "values": ['sigmoid', 'relu']},
        {"name": "clas_w", "type": "range", "bounds": [1e-4, 1e2], "log_scale": True},
        {"name": "dann_w", "type": "range", "bounds": [1e-4, 1e2], "log_scale": True},
        {"name": "learning_rate", "type": "range", "bounds": [1e-4, 1e-2], "log_scale": True},
        {"name": "weight_decay", "type": "range", "bounds": [1e-8, 1e-4], "log_scale": True},
        {"name": "warmup_epoch", "type": "range", "bounds": [1, 50]},
        {"name": "dropout", "type": "range", "bounds": [0.0, 0.5]},
        {"name": "bottleneck", "type": "range", "bounds": [32, 64]},
        {"name": "layer2", "type": "range", "bounds": [64, 512]},
        {"name": "layer1", "type": "range", "bounds": [512, 1024]},

    ]
    
    working_dir = '/home/acollin/dca_permuted_workflow/'
    experiment = MakeExperiment(run_file=run_file, working_dir=working_dir)
    
    best_parameters, values, experiment, model = optimize(
        parameters=hparams,
        evaluation_function=experiment.train,
        objective_name='mcc',
        minimize=False,
        total_trials=50,
        random_seed=40,
    )

