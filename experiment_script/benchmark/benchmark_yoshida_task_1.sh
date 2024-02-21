#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=benchmark_yoshida_1      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/scanvi_sin.sif"

module load singularity


singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name yoshida_2021 --class_key Original_annotation --use_hvg 3000 --batch_key batch --mode entire_condition --obs_key batch --keep_obs AN1 AN11 AN12 AN13 AN3 AN6 AN7 --test_split_key TRAIN_TEST_split_batch &> /home/acollin/dca_permuted_workflow/experiment_script/benchmark/yoshida_1.log