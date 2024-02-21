#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=tosti_1      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/hlca_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/singularity_scPermut.sif"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/workflow/workflow_bash.py --dataset_name tosti_2021 --class_key Original_annotation --batch_key batch --mode entire_condition --obs_key batch --keep_obs AFES448 TUM_13 TUM_C1 --test_split_key TRAIN_TEST_split_batch --training_scheme training_scheme_1 --hparam_path generic_r1 &> /home/acollin/dca_permuted_workflow/experiment_script/celltypist_datasets/tosti_1.log