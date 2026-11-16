#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.

module load miniconda
source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

echo "Env activated :"$(which python)

working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"
dataset_name=$1
class_key=$2
batch_key=$3
model=$4

python ${working_dir}/task2/task2_allmodels_label_transfer_pct_split.py --dataset_name $dataset_name --class_key $class_key \
    --use_hvg 3000 --batch_key $batch_key --mode percentage --obs_key $batch_key \
    --model $model 

## Singularity version
# module load singularity
#singularity_working_dir="/data/scPermut"
#singularity_path=$working_dir"/scanvi_sin.sif"
# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/02_label_transfer_pct_split.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode percentage --obs_key $batch_key --gpu_models $gpu_models &> $working_dir/experiment_script/benchmark/logs/task2_$dataset_name'_'$gpu_models.log