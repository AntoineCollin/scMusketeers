#!/bin/sh
#
conda activate scPermut2

python /home/acollin/dca_permuted_workflow/workflow/workflow_bash.py --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 5000 --mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 --training_scheme training_scheme_1 &> /home/acollin/dca_permuted_workflow/htap_hyperparam_scheme_1.log