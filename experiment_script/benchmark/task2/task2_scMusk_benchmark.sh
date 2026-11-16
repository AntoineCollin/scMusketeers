#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
dataset_name=$1
class_key=$2
batch_key=$3
task="task2"

module load miniconda
source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

echo "Env activated :"$(which python)

working_dir="/workspace/cell/scMusketeers/"
python_script=${working_dir}/experiment_script/benchmark/${task}/${task}_scMusk_label_transfer_pct_split.py

json_test=$(cat $working_dir/experiment_script/benchmark/hp_test_obs.json)
test_obs=$(echo "$json_test" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
test_obs=$(echo "$test_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo test_obs=$test_obs

best_hp=$working_dir"experiment_script/benchmark/best_hp.csv"
dataset_hp=$(awk -F',' '$1 == "'$dataset_name'"' "$best_hp")

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

# echo use_hvg=$use_hvg
# echo batch_size=$batch_size
# echo clas_w=$clas_w
# echo dann_w=$dann_w
# echo rec_w=$rec_w
# echo ae_bottleneck_activation=$ae_bottleneck_activation
# echo clas_loss_name=$clas_loss_name
# echo size_factor=$size_factor
# echo learning_rate=$learning_rate
# echo weight_decay=$weight_decay
# echo warmup_epoch=$warmup_epoch
# echo dropout=$dropout
# echo layer1=$layer1
# echo layer2=$layer2
# echo bottleneck=$bottleneck
# echo training_scheme=$training_scheme

echo "Run "${python_script}
python ${python_script} --dataset_name $dataset_name \
    --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key \
    --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w \
    --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay \
    --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 \
    --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True 




## Singularity part
# module load singularity
# singularity_working_dir="/data/scPermut/"
# singularity_path=$working_dir"/singularity_scPermut.sif"
# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/02_label_transfer_pct_split_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task2_focal_$dataset_name.log

# ############## DEFAULT ####################
# best_hp=$working_dir"experiment_script/benchmark/default_df_t10.csv"
# dataset_hp=$(sed -n '2p' $best_hp)

# IFS=',' read -r -a hp_list <<< "$dataset_hp"

# use_hvg=${hp_list[2]}
# batch_size=${hp_list[3]}
# clas_w=${hp_list[4]}
# dann_w=${hp_list[5]}
# rec_w=${hp_list[6]}
# ae_bottleneck_activation=${hp_list[7]}
# clas_loss_name=${hp_list[8]}
# size_factor=${hp_list[9]}
# weight_decay=${hp_list[10]}
# learning_rate=${hp_list[11]}
# warmup_epoch=${hp_list[12]}
# dropout=${hp_list[13]}
# layer1=${hp_list[14]}
# layer2=${hp_list[15]}
# bottleneck=${hp_list[16]}
# training_scheme=${hp_list[17]}

# module load singularity

# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/02_label_transfer_pct_split_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task2_focal_$dataset_name.log