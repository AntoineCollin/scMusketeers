script_dir="/home/becavin/scPermut"
working_dir="/data/analysis/data_becavin/scpermut_test/"
singularity_working_dir="/data/dca_permuted_workflow/"
singularity_path=$working_dir"/scPermut_gpu_jupyter.sif"
#python_script=$script_dir/scpermut/scpermut_optimize.py
python_script=$script_dir/scpermut/__main__.py
log_file=$working_dir"/experiment_script/htap_scheme_1.log"
json_path="/home/becavin/scPermut/experiment_script/hp_ranges/generic_r1.json"

dataset_name=tran_2021
class_key=Original_annotation
batch_key=batch

log_file=$working_dir"/experiment_script/${dataset_name}_hp.log"

json_test=$(cat $script_dir/experiment_script/benchmark/hp_test_obs.json)
test_obs=$(echo "$json_test" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
test_obs=$(echo "$test_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo test_obs=$test_obs

json_train=$(cat $script_dir/experiment_script/benchmark/hp_train_obs.json)
keep_obs=$(echo "$json_train" | grep -o "\"$dataset_name\": \[[^]]*\]" | cut -d '[' -f 2 | cut -d ']' -f 1)
keep_obs=$(echo "$keep_obs" | tr -d '[:space:]' | tr -d '"' | tr ',' ' ')
echo train_obs=$keep_obs

python $python_script hyperparameter --dataset_name $dataset_name --class_key $class_key --batch_key $batch_key --test_obs $test_obs --mode entire_condition --obs_key $batch_key --keep_obs $keep_obs --working_dir $working_dir &> ${log_file}

# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $python_script \
# --working_dir $singularity_working_dir --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 5000 \
# --mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 \
# --training_scheme training_scheme_1 &> $log_file

# --hparam_path htap_r2_sch1 