#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"

# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# srun -A cell -p cpucourt -t 10:00:00 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

#sc-musketeers --version


task="task1_New"
sh_file=${working_dir}/${task}/${task}_allmodels_benchmark.sh

model_list_cpu=("uce")
#model_list_cpu=("celltypist" "scmap_cells" "scmap_cluster" "pca_svm" )
#model_list_gpu=("scanvi" "harmony_svm")

jobname="t1_allmod"
cpu_sbatch="--partition=cpucourt --time=24:00:00"
gpu_sbatch="--partition=gpu --time=24:00:00"


# Run sh file directly
### ONLY FOR TESTING #####
# dataset="SmallIntestine-All"
# model="pca_svm"
# class_key="cell_type"
# batch_key="donor_id"
# sh ${sh_file} ${dataset} $class_key $batch_key ${model} 
# &> ${working_dir}/logs/scMusk_${task}_${dataset}_${model}.log
#nohup sh ${sh_file} ${dataset} $class_key $batch_key ${model} &> ${working_dir}/logs/scMusk_${task}_${dataset}_${model}.log &

# for model in "${model_list_cpu[@]}"; do
#     echo "Processing model: $model"
#     echo sh ${sh_file} ajrccm_by_batch celltype manip ${model} &> ${working_dir}/logs/scMusk_${task}_${model}_${dataset}.log
# done

#"Ageing-Mouse-All" "CellCards-Lung" "HLCA-full" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Immune" "TS-Liver" "TS-Neural" "TS-Skin"

####### sbatch for scMusketeers

#for dataset in "Ageing-Mouse-All" "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin";
#for dataset in "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin" "SmallIntestine-All" "SmallIntestine-20k";
for dataset in "SmallIntestine-All" "CellCards-Lung" ;
do
    class_key="cell_type"
    batch_key="donor_id"
    for model in "${model_list_cpu[@]}"; do
        for test_fold_nb in 0 1 2; do
            log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${test_fold_nb}.log
            sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_${dataset}_${model}_${test_fold_nb} ${sh_file} $dataset $class_key $batch_key ${model} ${test_fold_nb}
        done
    done
    for model in "${model_list_gpu[@]}"; do
        for test_fold_nb in 0 1 2; do
            log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${test_fold_nb}.log
            sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_${dataset}_${model}_${test_fold_nb} ${sh_file} $dataset $class_key $batch_key ${model} ${test_fold_nb}
        done
    done
done

