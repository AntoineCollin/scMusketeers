#!/bin/sh
#
# Submit sbatch jobs for all missing task1 benchmark runs.
# Missing runs are read from the CSV produced by 00_benchmark_sbatch_review.py.

working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"
task="task1"
csv_file="${working_dir}/paper_review/missing_benchmark_runs_${task}.csv"
scmusk_sh="${working_dir}/${task}/${task}_scMusk_benchmark.sh"
allmodels_sh="${working_dir}/${task}/${task}_allmodels_benchmark.sh"

jobname="t1_complete"
cpu_sbatch="--account=cell --partition=cpucourt --time=71:00:00"
gpu_sbatch="--account=cell --partition=gpu --gres=gpu:1 --time=35:00:00"

if [ ! -f "$csv_file" ]; then
    echo "Missing CSV not found: $csv_file"
    echo "Run 00_benchmark_sbatch_review.py first."
    exit 1
fi

# Group missing runs by (dataset, model, fold_i), collecting the missing j values
# per group, so only the missing folds are resubmitted (not the whole dataset).
# Parameter format for task1 is "i_j" (e.g. "0_2").
awk -F',' '
NR > 1 {
    split($4, parts, "_")
    key = $1 SUBSEP $3 SUBSEP parts[1]
    if (!(key in seen)) { order[++n] = key; seen[key] = 1 }
    js[key] = (js[key] == "" ? parts[2] : js[key] " " parts[2])
}
END {
    for (k = 1; k <= n; k++) {
        split(order[k], f, SUBSEP)
        printf "%s,%s,%s,%s\n", f[1], f[2], f[3], js[order[k]]
    }
}' "$csv_file" | while IFS=',' read -r dataset model fold_i fold_js; do

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

    log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold_i}.log"

    if [ "$model" = "scMusketeers" ]; then
        # scMusketeers always runs the hp_test_obs-selected test fold; fold_i is
        # passed for the record, fold_js restricts which val folds are run.
        sbatch $gpu_sbatch \
            --output "$log_out" --job-name "${jobname}_${dataset}_${fold_i}" \
            "$scmusk_sh" "$dataset" "$class_key" "$batch_key" "$fold_i" $fold_js
    elif [ "$model" = "scanvi" ] || [ "$model" = "harmony_svm" ]; then
        sbatch $gpu_sbatch \
            --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold_i}" \
            "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold_i" $fold_js
    else
        sbatch $cpu_sbatch \
            --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold_i}" \
            "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold_i" $fold_js
    fi

done
