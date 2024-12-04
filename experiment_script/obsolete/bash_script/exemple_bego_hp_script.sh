script_dir="/home/becavin/scPermut"
working_dir="/data/analysis/data_becavin/scpermut_test/"
singularity_working_dir="/data/dca_permuted_workflow/"
singularity_path=$working_dir"/scPermut_gpu_jupyter.sif"
#python_script=$script_dir/scpermut/scpermut_optimize.py
python_script=$script_dir/scpermut/__main__.py
log_file=$working_dir"/experiment_script/htap_scheme_1.log"
json_path="/home/becavin/scPermut/experiment_script/hp_ranges/generic_r1.json"

python $python_script \
--working_dir $working_dir --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 5000 \
--mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 \
--training_scheme training_scheme_1 --test_split_key TRAIN_TEST_split --log_neptune False --hparam_path $json_path &> $log_file

# singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $python_script \
# --working_dir $singularity_working_dir --dataset_name htap_final_by_batch --class_key celltype --batch_key donor --use_hvg 5000 \
# --mode entire_condition --obs_key donor --keep_obs PAH_688194 PAH_691048 PAH_693770 PAH_698029 PRC_16 PRC_18 PRC_3 PRC_8 \
# --training_scheme training_scheme_1 &> $log_file

# --hparam_path htap_r2_sch1 