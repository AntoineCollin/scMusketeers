#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/scanvi_sin.sif"
dataset_name=$1
class_key=$2
batch_key=$3

module load singularity


singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode fixed_number --obs_key $batch_key &> /home/acollin/dca_permuted_workflow/experiment_script/benchmark/$dataset_name.log