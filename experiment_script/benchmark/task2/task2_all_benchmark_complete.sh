#!/bin/sh
#
# Submit sbatch jobs for all missing task2 benchmark runs.
# Missing runs are read from the CSV produced by 00_benchmark_sbatch_review.py.

working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"
task="task2"
csv_file="${working_dir}/paper_review/missing_benchmark_runs_task${task}.csv"
scmusk_sh="${working_dir}/${task}/${task}_scMusk_benchmark.sh"
allmodels_sh="${working_dir}/${task}/${task}_allmodels_benchmark.sh"

jobname="t2_complete"
cpu_sbatch="--account=cell --partition=cpucourt --time=71:00:00"
gpu_sbatch="--account=cell --partition=gpu --gres=gpu:1 --time=35:00:00"

if [ ! -f "$csv_file" ]; then
    echo "Missing CSV not found: $csv_file"
    echo "Run 00_benchmark_sbatch_review.py first."
    exit 1
fi

# Extract unique dataset,model pairs from CSV (skip header, columns 1 and 3)
tail -n +2 "$csv_file" | awk -F',' '{print $1","$3}' | sort -u | while IFS=',' read -r dataset model; do

    case "$dataset" in
        ajrccm_by_batch)
            class_key="celltype"; batch_key="manip" ;;
        htap)
            class_key="ann_finest_level"; batch_key="donor" ;;
        hlca_par_dataset_harmonized|hlca_trac_dataset_harmonized)
            class_key="ann_finest_level"; batch_key="dataset" ;;
        *)
            class_key="Original_annotation"; batch_key="batch" ;;
    esac

    log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}.log"

    if [ "$model" = "scMusketeers" ]; then
        sbatch $gpu_sbatch \
            --output "$log_out" --job-name "${jobname}_${dataset}" \
            "$scmusk_sh" "$dataset" "$class_key" "$batch_key"
    elif [ "$model" = "scanvi" ] || [ "$model" = "harmony_svm" ]; then
        sbatch $gpu_sbatch \
            --output "$log_out" --job-name "${jobname}_${dataset}_${model}" \
            "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model"
    else
        sbatch $cpu_sbatch \
            --output "$log_out" --job-name "${jobname}_${dataset}_${model}" \
            "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model"
    fi

done
