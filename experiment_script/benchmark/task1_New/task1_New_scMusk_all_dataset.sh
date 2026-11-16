#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/"
task="task1_New"
sh_file=${working_dir}/${task}/${task}_scMusk_benchmark.sh

#for dataset in "Ageing-Mouse-All" "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin";
# for dataset in "TS-Liver" ;
# do
#     echo $dataset
#     sh ${sh_file} $dataset cell_type donor_id 
#     #&> ${working_dir}/logs/scMusk_${task}_${dataset}.log
# done

#dataset=ajrccm_by_batch
#sh ${sh_file} ajrccm_by_batch celltype manip &> ${working_dir}/logs/scMusk_${task}_${dataset}.log
#${sh_file}  htap ann_finest_level donor
#${sh_file} hlca_par_dataset_harmonized ann_finest_level dataset
#${sh_file} hlca_trac_dataset_harmonized ann_finest_level dataset


####### sbatch for scPermut
jobname="t1_scm"
cpu_sbatch="--partition=cpucourt --time=24:00:00"
gpu_sbatch="--partition=gpu --time=24:00:00"

for dataset in "SmallIntestine-All" "SmallIntestine-20k" ;
#for dataset in "Ageing-Mouse-All" "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin" "SmallIntestine-All" "SmallIntestine-20k";
do
    class_key="cell_type"
    batch_key="donor_id"
    for test_fold_nb in 0 1 2; do
        log_out=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/scMusk_sbatch_${task}_${dataset}_fold${test_fold_nb}.log
        #scancel -n ${jobname}_${dataset}_${test_fold_nb}
        sbatch ${cpu_sbatch} --output ${log_out} --job-name ${jobname}_${dataset}_${test_fold_nb} ${sh_file} $dataset $class_key $batch_key $test_fold_nb
    done
done
