#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"

# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# srun -A cell -p cpucourt -t 10:00:00 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate
# python /workspace/cell/scMusketeers/experiment_script/benchmark/01_label_transfer_between_batch.py --dataset_name ajrccm_by_batch --class_key celltype --use_hvg 3000 --batch_key manip --mode entire_condition --obs_key manip --gpu_models True

sc-musketeers --version


task="task2"
sh_file=${working_dir}/${task}/${task}_allmodels_benchmark.sh

all_models=("uce" "celltypist" "scmap_cells" "scmap_cluster" "pca_svm" "pca_knn" "scanvi" "harmony" "scMusketeers")
model_list_cpu=("uce" "celltypist" "scmap_cells" "scmap_cluster" "pca_svm" "pca_knn")
model_list_gpu=()

jobname="t2_allmod"
cpu_sbatch="--partition=cpucourt --time=71:00:00"
gpu_sbatch="--partition=gpu --gres=gpu:1 --time=35:00:00"

#for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
#do
#    echo $dataset
#    #${sh_file} $dataset Original_annotation batch
#done

#dataset=ajrccm_by_batch

# Iterate through the array
#model="pca_svm"
#sh ${sh_file} ajrccm_by_batch celltype manip ${model} &> ${working_dir}/logs/scMusk_${task}_${model}_${dataset}.log

# for model in "${model_list_cpu[@]}"; do
#     echo "Processing model: $model"
#     echo sh ${sh_file} ajrccm_by_batch celltype manip ${model} &> ${working_dir}/logs/scMusk_${task}_${model}_${dataset}.log
# done

#sh ${sh_file} ajrccm_by_batch celltype manip False &> ${working_dir}/logs/scMusk_${task}_${dataset}.log
#${sh_file}  htap ann_finest_level donor
#${sh_file} hlca_par_dataset_harmonized ann_finest_level dataset
#${sh_file} hlca_trac_dataset_harmonized ann_finest_level dataset


####### sbatch for scMusketeers

for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
do
    class_key="Original_annotation"
    batch_key="batch"
    for model in "${model_list_cpu[@]}"; do
        log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
        sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
    done
    for model in "${model_list_gpu[@]}"; do
        log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
        sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
    done
done

dataset="ajrccm_by_batch"
class_key="celltype"
batch_key="manip"
for model in "${model_list_cpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done



dataset="htap"
class_key="ann_finest_level"
batch_key="donor"
for model in "${model_list_cpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done


dataset="hlca_par_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
for model in "${model_list_cpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done

dataset="hlca_trac_dataset_harmonized"
class_key="ann_finest_level"
batch_key="dataset"
for model in "${model_list_cpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done
for model in "${model_list_gpu[@]}"; do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log
    sbatch ${gpu_sbatch} --output ${log_out} --job-name ${jobname}_$dataset_$model ${sh_file} $dataset $class_key $batch_key ${model}
done
