#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=htap_2      # The job name.
#SBATCH --partition=cpucourt
#SBATCH --time=70:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

conda activate scPermut2

python /home/acollin/dca_permuted_workflow/workflow/run_hp.py --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 5000 --mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 --training_scheme training_scheme_2 &> /home/acollin/dca_permuted_workflow/experiment_script/htap_hyperparam_scheme_2.log