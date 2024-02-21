#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=disco_ajrccm_3      # The job name.
#SBATCH --partition=cpucourt
#SBATCH --time=70:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

conda activate scPermut2

python /home/acollin/dca_permuted_workflow/workflow/run_hp.py --dataset_name discovair_ajrccm_small --class_key celltype_lv2_V3 --batch_key dataset --use_hvg 5000 --mode entire_condition --obs_key dataset --keep_obs discovair --training_scheme training_scheme_3 &> /home/acollin/dca_permuted_workflow/experiment_script/discovair_ajrccm_small_3.log