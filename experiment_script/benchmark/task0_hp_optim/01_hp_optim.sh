#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 


working_dir="/home/acollin/scPermut/"
singularity_working_dir="/data/scPermut/"
singularity_path=$working_dir"/singularity_scPermut.sif"
dataset_name=$1
class_key=$2
batch_key=$3

json_test=$(cat $working_dir/experiment_script/benchmark/hp_test_obs.json)
test_obs=$(echo "$json_test" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
test_obs=$(echo "$test_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo test_obs=$test_obs

json_train=$(cat $working_dir/experiment_script/benchmark/hp_train_obs.json)
keep_obs=$(echo "$json_train" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
keep_obs=$(echo "$keep_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo train_obs=$keep_obs

module load singularity

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/scpermut/run_hp.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --keep_obs $keep_obs --working_dir $working_dir &> $working_dir/experiment_script/benchmark/logs/hp_optim_$dataset_name.log