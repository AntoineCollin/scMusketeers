#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"

task="task2_New"
sh_file=${working_dir}/${task}/${task}_allmodels_benchmark.sh

model_list_cpu=("uce")
#model_list_cpu=("celltypist" "scmap_cells" "scmap_cluster" "pca_svm" )
#model_list_gpu=("scanvi" "harmony_svm")


jobname="t2n_allmod"
cpu_sbatch="--account=cell --partition=cpucourt --time=71:00:00"
gpu_sbatch="--account=cell --partition=gpu --gres=gpu:1 --time=35:00:00"


# dataset="CellCards-Lung"
# model="uce"
# class_key="cell_type"
# batch_key="donor_id"
# sh ${sh_file} ${dataset} $class_key $batch_key ${model} 

####### sbatch for all models
for dataset in "CellCards-Lung"; # "TS-Skin";
# "SmallIntestine-20k" ;
#for dataset in "Ageing-Mouse-All" "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin" "SmallIntestine-All" "SmallIntestine-20k";
do
    class_key="cell_type"
    batch_key="donor_id"
    for model in "${model_list_cpu[@]}"; do
        for test_fold_nb in 0 1 2; do
            for pct_split_nb in 0 1 2 3; do
                log_out=${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${test_fold_nb}_pct${pct_split_nb}.log
                sbatch ${cpu_sbatch} --output ${log_out} \
                    --job-name ${jobname}_${dataset}_${model}_${test_fold_nb}_${pct_split_nb} \
                    ${sh_file} $dataset $class_key $batch_key ${model} ${test_fold_nb} ${pct_split_nb}
            done
        done
    done
    # for model in "${model_list_gpu[@]}"; do
    #     for test_fold_nb in 0 1 2; do
    #         for pct_split_nb in 0 1 2 3; do
    #             log_out=${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${test_fold_nb}_pct${pct_split_nb}.log
    #             sbatch ${gpu_sbatch} --output ${log_out} \
    #                 --job-name ${jobname}_${dataset}_${model}_${test_fold_nb}_${pct_split_nb} \
    #                 ${sh_file} $dataset $class_key $batch_key ${model} ${test_fold_nb} ${pct_split_nb}
    #         done
    #     done
    # done
done
