#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"
task="task1"
sh_file=${working_dir}/${task}/${task}_scMusk_benchmark.sh
jobname="t1_scm"

# for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
# do
#     echo $dataset
#     #${sh_file} $dataset Original_annotation batch
# done

#dataset=ajrccm_by_batch
sh ${sh_file} ajrccm_by_batch celltype manip &> ${working_dir}/logs/scMusk_${task}_${dataset}.log
#${sh_file}  htap ann_finest_level donor
#${sh_file} hlca_par_dataset_harmonized ann_finest_level dataset
#${sh_file} hlca_trac_dataset_harmonized ann_finest_level dataset


####### sbatch for scPermut

for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
do
    log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
    sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name ${jobname}_${dataset} ${sh_file} $dataset Original_annotation batch
done

dataset="ajrccm_by_batch"
log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name ${jobname}_${dataset} ${sh_file} ${dataset} celltype manip

dataset="htap"
log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name ${jobname}_${dataset} ${sh_file} ${dataset} ann_finest_level donor

dataset="hlca_par_dataset_harmonized"
log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name ${jobname}_${dataset} ${sh_file} ${dataset} ann_finest_level dataset

dataset="hlca_trac_dataset_harmonized"
log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}.log
sbatch --partition=gpu --gres=gpu:1 --output ${log_out} --time=35:00:00 --job-name ${jobname}_${dataset} ${sh_file} ${dataset} ann_finest_level dataset