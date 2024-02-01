#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=htap_debug      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu03
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/singularity_scPermut.sif"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/workflow/workflow_bash.py --dataset_name discovair_ajrccm_small --class_key celltype_lv1_V3 --batch_key dataset --use_hvg 5000 --mode entire_condition --obs_key dataset --keep_obs 'discovair' --training_scheme training_scheme_2 --hparam_path generic_r1_debug &> /home/acollin/dca_permuted_workflow/experiment_script/hyperparam_scheme_debug.log
