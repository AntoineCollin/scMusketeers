#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=yosh1      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log

working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/singularity_scPermut.sif"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/workflow/workflow_bash.py --dataset_name yoshida_2021 --class_key Original_annotation --batch_key batch --mode entire_condition --obs_key batch --keep_obs AN1 AN11 AN12 AN13 AN3 AN6 AN7 --training_scheme training_scheme_1 --test_split_key TRAIN_TEST_split_batch &> /home/acollin/dca_permuted_workflow/experiment_script/hp_runs/logs/yoshida_hyperparam_scheme_1.log

