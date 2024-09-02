#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/scPermut"
singularity_working_dir="/data/scPermut"
singularity_path=$working_dir"/scanvi_sin.sif"
dataset_name=$1
class_key=$2
batch_key=$3
gpu_models=$4

module load singularity


singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/04_label_transfer_spatial.py --dataset_name $dataset_name --class_key $class_key --use_hvg 0 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models &> $working_dir/experiment_script/benchmark/logs/task4_$dataset_name'_'$gpu_models.log