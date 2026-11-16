#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
dataset_name=$1
class_key=$2
batch_key=$3
test_fold_nb=$4
pct_split_nb=$5
# Optional: specific seed indices as remaining positional args (e.g. 0 2 4)
seed_args="${@:6}"
seed_flag=""
[ -n "$seed_args" ] && seed_flag="--random_seed_nb $seed_args"
task="task2"

module load miniconda
source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

echo "Env activated :"$(which python)

working_dir="/workspace/cell/scMusketeers/"
python_script=${working_dir}/experiment_script/benchmark/task2_New/task2_New_scMusk_label_transfer_pct_split.py

# # best_hp=$working_dir"experiment_script/benchmark/best_hp.csv"
# # dataset_hp=$(awk -F',' '$1 == "'$dataset_name'"' "$best_hp")

# # IFS=',' read -r -a hp_list <<< "$dataset_hp"

# # use_hvg=${hp_list[2]}
# # batch_size=${hp_list[3]}
# # clas_w=${hp_list[4]}
# # dann_w=${hp_list[5]}
# # rec_w=${hp_list[6]}
# # ae_bottleneck_activation=${hp_list[7]}
# # clas_loss_name=${hp_list[8]}
# # size_factor=${hp_list[9]}
# # weight_decay=${hp_list[10]}
# # learning_rate=${hp_list[11]}
# # warmup_epoch=${hp_list[12]}
# # dropout=${hp_list[13]}
# # layer1=${hp_list[14]}
# # layer2=${hp_list[15]}
# # bottleneck=${hp_list[16]}
# # training_scheme=${hp_list[17]}

########## DEFAULT PARAMS ##############

best_hp=$working_dir"experiment_script/default_df_t10.csv"
dataset_hp=$(sed -n '2p' $best_hp)

IFS=',' read -r -a hp_list <<< "$dataset_hp"

use_hvg=${hp_list[2]}
batch_size=${hp_list[3]}
clas_w=${hp_list[4]}
dann_w=${hp_list[5]}
rec_w=${hp_list[6]}
ae_bottleneck_activation=${hp_list[7]}
clas_loss_name=${hp_list[8]}
size_factor=${hp_list[9]}
weight_decay=${hp_list[10]}
learning_rate=${hp_list[11]}
warmup_epoch=${hp_list[12]}
dropout=${hp_list[13]}
layer1=${hp_list[14]}
layer2=${hp_list[15]}
bottleneck=${hp_list[16]}
training_scheme=${hp_list[17]}

echo "Run "${python_script}
python ${python_script} --dataset_name $dataset_name \
    --class_key $class_key --batch_key $batch_key --test_fold_nb $test_fold_nb --pct_split_nb $pct_split_nb $seed_flag --mode entire_condition --obs_key $batch_key \
    --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w \
    --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay \
    --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 \
    --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True
