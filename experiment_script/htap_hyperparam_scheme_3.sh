#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=htap_debug      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/singularity_scPermut.sif"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/workflow/workflow_hp_debug.py --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 10000 --mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 --training_scheme training_scheme_2 &> /home/acollin/dca_permuted_workflow/experiment_script/htap_hyperparam_scheme_2.log