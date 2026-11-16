#!/bin/sh
#
working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"
task="task2_New"
sh_file=${working_dir}/${task}/${task}_scMusk_benchmark.sh

####### sbatch for scMusketeers
jobname="t2_scm"
cpu_sbatch="--partition=cpucourt --time=24:00:00"
gpu_sbatch="--partition=gpu --time=24:00:00"

for dataset in "CellCards-Lung" "SmallIntestine-All" ;
#for dataset in "SmallIntestine-All" "SmallIntestine-20k" ;
#for dataset in "Ageing-Mouse-All" "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin" "SmallIntestine-All" "SmallIntestine-20k";
do
    class_key="cell_type"
    batch_key="donor_id"
    for test_fold_nb in 0 1 2; do
        for pct_split_nb in 0 1 2 3; do
            log_out=${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_fold${test_fold_nb}_pct${pct_split_nb}.log
            sbatch ${cpu_sbatch} --output ${log_out} \
                --job-name ${jobname}_${dataset}_${test_fold_nb}_${pct_split_nb} \
                ${sh_file} $dataset $class_key $batch_key $test_fold_nb $pct_split_nb
        done
    done
done
