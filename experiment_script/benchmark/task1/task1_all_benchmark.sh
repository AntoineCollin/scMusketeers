#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"

# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# srun -A cell -p cpucourt -t 10:00:00 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

#sc-musketeers --version


task="task1"
sh_file=${working_dir}/${task}/${task}_allmodels_benchmark.sh

model_list_cpu=("uce" "celltypist" "scmap_cells" "scmap_cluster" "pca_svm" "pca_knn")
model_list_gpu=("scanvi" "harmony_svm")

jobname="t1_allmod"
cpu_sbatch="--partition=cpucourt --time=71:00:00"
gpu_sbatch="--partition=gpu --time=24:00:00"

dataset=yoshida_2021

# Iterate through the array
model="harmony_svm"
class_key="Original_annotation"
batch_key="batch"
#sh ${sh_file} ${dataset} $class_key $batch_key ${model} 
# &> ${working_dir}/logs/scMusk_${task}_${dataset}_${model}.log
#nohup sh ${sh_file} ${dataset} $class_key $batch_key ${model} &> ${working_dir}/logs/scMusk_${task}_${dataset}_${model}.log &

# for model in "${model_list_cpu[@]}"; do
#     echo "Processing model: $model"
#     echo sh ${sh_file} ajrccm_by_batch celltype manip ${model} &> ${working_dir}/logs/scMusk_${task}_${model}_${dataset}.log
# done



####### sbatch for scMusketeers

# for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
for dataset in "tran_2021" ;
do
    class_key="Original_annotation"
    batch_key="batch"
    # for model in "${model_list_cpu[@]}"; do
    #     log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    #     sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
    # done
    for model in "${model_list_gpu[@]}"; do
        log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
        sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
    done
done

dataset="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
# for model in "${model_list_cpu[@]}"; do
#     log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
#     sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
# done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done



dataset="htap"
class_key="ann_finest_level"
batch_key="donor"
# for model in "${model_list_cpu[@]}"; do
#     log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
#     sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
# done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done


dataset="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
# for model in "${model_list_cpu[@]}"; do
#     log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
#     sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
# done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done

dataset="hlca_trac_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
# for model in "${model_list_cpu[@]}"; do
#     log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
#     sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
# done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done
