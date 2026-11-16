#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.

module load miniconda
source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

echo "Env activated :"$(which python)

working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"
dataset_name=$1
class_key=$2
batch_key=$3
model=$4
test_fold_nb=$5
# Optional: specific j-fold indices as remaining positional args (e.g. 0 2 4)
j_args="${@:6}"
j_flag=""
[ -n "$j_args" ] && j_flag="--test_fold_j_nb $j_args"

python ${working_dir}/task1_New/task1_New_allmodels_label_transfer_between_batch.py --dataset_name $dataset_name \
    --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode entire_condition \
    --obs_key $batch_key --model $model --test_fold_nb $test_fold_nb $j_flag

# singularity version
# module load singularity
# singularity_working_dir="/data/scPermut"
# singularity_path=$working_dir"/scanvi_sin_copy.sif"
# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name $dataset_name --class_key $class_key --use_hvg 3000 --batch_key $batch_key --mode entire_condition --obs_key $batch_key --gpu_models $gpu_models &> $working_dir/experiment_script/benchmark/logs/task1_$dataset_name'_'$gpu_models.log