#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.

module load miniconda
source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

echo "Env activated :"$(which python)

working_dir="/workspace/cell/scMusketeers/"
dataset_name=$1
class_key=$2
batch_key=$3
test_fold_nb=$4
# Optional: specific j-fold indices as remaining positional args (e.g. 0 2 4)
j_args="${@:5}"
j_flag=""
[ -n "$j_args" ] && j_flag="--test_fold_j_nb $j_args"
task="task1"


# best_hp=$working_dir"experiment_script/benchmark/best_hp.csv"
# dataset_hp=$(awk -F',' '$1 == "'$dataset_name'"' "$best_hp")

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

# ######## BEST PARAMS ###############

# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task1_focal_$dataset_name.log

######### BEST PARAMS, CUSTOM SCHEME ############

# training_scheme=training_scheme_11

# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task1_focal_$dataset_name.log


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

python $working_dir/experiment_script/benchmark/task1_New/task1_New_scMusk_label_transfer_between_batch.py --task=${task} --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key \
    --test_fold_nb $test_fold_nb $j_flag --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w \
    --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay \
    --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck \
    --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True



######### DEFAULT PARAMS, CUSTOM SCHEME ############



# for training_scheme in "training_scheme_17" "training_scheme_18" "training_scheme_12" "training_scheme_13" "training_scheme_14" "training_scheme_15" "training_scheme_8" "training_scheme_9" "training_scheme_16" "training_scheme_4" "training_scheme_11" "training_scheme_19" "training_scheme_20" "training_scheme_22" "training_scheme_23" "training_scheme_24" "training_scheme_5" "training_scheme_6" "training_scheme_7" "training_scheme_25" "training_scheme_26"
# do
#     singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/01_label_transfer_between_batch_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task1_focal_$dataset_name.log
# done
