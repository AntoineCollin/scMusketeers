#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/ajrccm_hyperparam.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 

module load singularity

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

########## DEFAULT PARAMS ##############

if [ $dataset_name -eq tenx_hlca_par ]
then
best_hp=$working_dir"experiment_script/benchmark/default_df_t10_SPATIAL.csv"
else
best_hp=$working_dir"experiment_script/benchmark/default_df_t10.csv"
fi

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

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/04_label_transfer_spatial_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task4_focal_$dataset_name.log


######### DEFAULT PARAMS, CUSTOM SCHEME ############



# for training_scheme in "training_scheme_9" "training_scheme_19" "training_scheme_20" 
# do
#     singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/experiment_script/benchmark/04_label_transfer_spatial_scPermut.py --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --working_dir $working_dir --use_hvg $use_hvg --batch_size $batch_size --clas_w $clas_w --dann_w $dann_w --rec_w $rec_w --ae_bottleneck_activation $ae_bottleneck_activation --size_factor $size_factor --weight_decay $weight_decay --learning_rate $learning_rate --warmup_epoch $warmup_epoch --dropout $dropout --layer1 $layer1 --layer2 $layer2 --bottleneck $bottleneck --training_scheme $training_scheme --clas_loss_name $clas_loss_name --balance_classes True &> $working_dir/experiment_script/benchmark/logs/scPermut_task4_focal_$dataset_name.log
# done
