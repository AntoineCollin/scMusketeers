#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=ajrccm_2      # The job name.
#SBATCH --partition=cpucourt
#SBATCH --time=70:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/acollin/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

conda activate scPermut2

python /home/acollin/dca_permuted_workflow/workflow/run_hp.py --dataset_name ajrccm_by_batch --class_key celltype --batch_key manip --use_hvg 5000 --mode entire_condition --obs_key manip --keep_obs D322_Biop_Int1 D322_Biop_Nas1 D322_Biop_Pro1 D326_Biop_Int1 D326_Biop_Pro1 D339_Biop_Int1 D339_Biop_Pro1 D339_Brus_Dis1 D344_Biop_Int1 D344_Biop_Nas1 D344_Biop_Pro1 D344_Brus_Dis1 D353_Biop_Int2 D353_Biop_Pro1 D353_Brus_Dis1 D353_Brus_Nas1 D354_Biop_Int2 D363_Biop_Int2 D363_Biop_Pro1 D363_Brus_Dis1 D367_Brus_Dis1 D367_Brus_Nas1 D372_Biop_Int1 D372_Brus_Dis1 D372_Brus_Nas1 --training_scheme training_scheme_2 &> /home/acollin/dca_permuted_workflow/experiment_script/ajrccm_hyperparam_scheme_2.log