#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=hlca_1    # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
singularity_path=$working_dir"/singularity_scPermut.sif"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python /home/acollin/dca_permuted_workflow/workflow/workflow_run.py --dataset_name hlca_par_dataset_harmonized --class_key ann_finest_level --batch_key dataset --use_hvg 5000 --mode entire_condition --obs_key dataset --keep_obs Banovich_Kropski_2020 Krasnow_2020 Lafyatis_Rojas_2019_10Xv1 Misharin_2021 --training_scheme training_scheme_1 &> /home/acollin/dca_permuted_workflow/experiment_script/single_runs/hlca_par_hyperparam_scheme_1.log