#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=pbmc_2      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/singularity_scPermut.sif"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/workflow/workflow_bash.py --dataset_name pbmc8k_68k --class_key celltype --batch_key dataset --use_hvg 5000 --mode entire_condition --obs_key dataset --keep_obs pbmc68k --training_scheme training_scheme_2 --hparam_path generic_r1 &> /home/acollin/dca_permuted_workflow/experiment_script/htap_hyperparam_scheme_2.log
