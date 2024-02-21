#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --job-name=cb_htap_hp_1      # The job name.
#SBATCH --gres=gpu:1           # Request 1 gpu (Up to 2 gpus per GPU node)
#SBATCH --partition=gpu
#SBATCH --time=35:00:00           # The time the job will take to run in D-HH:MM
#SBATCH --output=/home/cbecavin/temp/htap_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


script_dir="/home/cbecavin/scPermut"
working_dir="/workspace/cell/sc_Permut"
singularity_working_dir="/data/scPermut/"
singularity_path=$working_dir"/scPermut_gpu_jupyter.sif"
python_script=$script_dir/scPermut/__main__.py
log_file=$working_dir"/experiment_script/htap_scheme_hp_1.log"
json_path=$script_dir"/experiment_script/hp_ranges/htap_r2_sch1.json"

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $python_script \
--working_dir $singularity_working_dir --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 5000 \
--mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 \
--training_scheme training_scheme_1 --hparam_path $json_path &> $log_file
