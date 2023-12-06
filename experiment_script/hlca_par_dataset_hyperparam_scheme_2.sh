#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=hlca_2      # The job name.
#SBATCH --partition=cpucourt
#SBATCH --time=70:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

conda activate scPermut2

python /home/acollin/dca_permuted_workflow/workflow/run_hp.py --dataset_name hlca_par_dataset_harmonized --class_key ann_finest_level --batch_key dataset --use_hvg 5000 --mode entire_condition --obs_key dataset --keep_obs Banovich_Kropski_2020 Krasnow_2020 Lafyatis_Rojas_2019_10Xv1 Misharin_2021 --training_scheme training_scheme_2 &> /home/acollin/dca_permuted_workflow/experiment_script/hlca_par_hyperparam_scheme_2.log