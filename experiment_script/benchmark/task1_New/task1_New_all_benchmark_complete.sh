#!/bin/sh
#
# Submit sbatch jobs for all missing task1_New benchmark runs.
# Missing runs are read from the CSV produced by 00_benchmark_sbatch_review.py.

working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"
task="task1_New"
csv_file="${working_dir}/paper_review/missing_benchmark_runs_${task}.csv"
scmusk_sh="${working_dir}/${task}/${task}_scMusk_benchmark.sh"
allmodels_sh="${working_dir}/${task}/${task}_allmodels_benchmark.sh"

jobname="t1n_complete"
cpu_sbatch="--account=cell --partition=cpucourt --time=71:00:00"
gpu_sbatch="--account=cell --partition=gpu --gres=gpu:1 --time=35:00:00 --mem=64G"
# The GPU account quota (MaxGRESPerAccount) limits how many GPU jobs run at once,
# so the GPU-capable models (scMusketeers, scanvi, harmony_svm, celltypist) are
# run on CPU instead. The model code auto-detects the device; harmony/scmap are
# CPU anyway, scanvi/scMusketeers/celltypist fall back to CPU (slower).
# cpucourt has both 192 GB and 384 GB nodes (35 of the latter), so request
# generously: 128 GB normally, and 370 GB for the two largest datasets. 370 GB
# forces SLURM onto a 384 GB node and uses nearly all of it (requesting 384G
# would exceed the node's 384000 MB RealMemory and never schedule).
model_cpu_sbatch="--account=cell --partition=cpucourt --time=71:00:00 --mem=128G"
model_cpu_highmem_sbatch="--account=cell --partition=cpucourt --time=71:00:00 --mem=370G"
# CPU models (scmap, pca) also load the full matrix and densify it during HVG
# selection; on the two largest datasets 128 GB is not enough, so use 180 GB
# (cpucourt nodes have 190 GB).
cpu_highmem_sbatch="--account=cell --partition=cpucourt --time=71:00:00 --mem=180G"
# uce loads the full precomputed-embedding AnnData into RAM (the CellCards-Lung /
# SmallIntestine-All uce_input files are 9-14 GB on disk) and densifies it, so it
# needs a lot of memory. The default ~4.75 GB allocation gets OOM-killed (exit 9).
# Small/medium datasets fit in 128 GB on cpucourt; the two largest datasets
# (CellCards-Lung 348k cells, SmallIntestine-All 265k cells) exceed even 128 GB,
# so they go to the smp big-memory partition (smp01 = 1.5 TB).
uce_sbatch="--account=cell --partition=cpucourt --time=71:00:00 --mem=128G"
uce_smp_sbatch="--account=cell --partition=smp --time=71:00:00 --mem=200G"

# smp UCE jobs are collected into a manifest and submitted as a single job array
# with a concurrency cap, so they don't monopolise the (single) smp big-memory
# node. At most uce_smp_throttle array tasks run at once.
uce_smp_throttle=6
uce_array_runner="${working_dir}/uce_array_runner.sh"
uce_smp_manifest="${working_dir}/sbatch_logs/uce_smp_manifest_${task}_$(date +%Y%m%d_%H%M%S).txt"

if [ ! -f "$csv_file" ]; then
    echo "Missing CSV not found: $csv_file"
    echo "Run 00_benchmark_sbatch_review.py first."
    exit 1
fi

# All task1_New datasets share the same keys
class_key="cell_type"
batch_key="donor_id"

# Group missing runs by (dataset, model, fold_i), collecting the missing j values per group.
# This avoids re-running already completed j folds within the same fold_i.
# Parameter format for task1_New is "i_j" (e.g. "0_2").
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

    # GPU-capable models run on CPU (GPU quota saturated); pick the memory tier
    # by dataset size.
    case "$dataset" in
        CellCards-Lung|SmallIntestine-All) model_opts="$model_cpu_highmem_sbatch" ;;
        *)                                 model_opts="$model_cpu_sbatch" ;;
    esac

    for j in $fold_js; do
        if [ "$model" = "scMusketeers" ]; then
            log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_fold${fold_i}_j${j}.log"
            sbatch $model_opts \
                --output "$log_out" --job-name "${jobname}_${dataset}_${fold_i}_j${j}" \
                "$scmusk_sh" "$dataset" "$class_key" "$batch_key" "$fold_i" $j
        elif [ "$model" = "scanvi" ] || [ "$model" = "harmony_svm" ] || [ "$model" = "celltypist" ]; then
            # celltypist auto-detects GPU and uses CPU SGD when none is present
            log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold_i}_j${j}.log"
            sbatch $model_opts \
                --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold_i}_j${j}" \
                "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold_i" $j
        elif [ "$model" = "uce" ]; then
            # high-memory CPU job; the two largest datasets need the smp partition.
            case "$dataset" in
                CellCards-Lung|SmallIntestine-All)
                    # collect into the manifest -> submitted as a throttled array below
                    echo "$dataset $class_key $batch_key $model $fold_i $j" >> "$uce_smp_manifest"
                    ;;
                *)
                    log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold_i}_j${j}.log"
                    sbatch $uce_sbatch \
                        --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold_i}_j${j}" \
                        "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold_i" $j
                    ;;
            esac
        else
            # CPU models (scmap, pca): the two largest datasets need more memory
            case "$dataset" in
                CellCards-Lung|SmallIntestine-All) cpu_opts="$cpu_highmem_sbatch" ;;
                *)                                 cpu_opts="$cpu_sbatch" ;;
            esac
            log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold_i}_j${j}.log"
            sbatch $cpu_opts \
                --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold_i}_j${j}" \
                "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold_i" $j
        fi
    done

done

# Submit the collected smp UCE tasks as a single throttled job array so they do
# not all run on the smp node at once (max uce_smp_throttle concurrent).
if [ -s "$uce_smp_manifest" ]; then
    n_uce=$(wc -l < "$uce_smp_manifest")
    echo "Submitting $n_uce smp UCE task(s) as a job array (max ${uce_smp_throttle} concurrent)"
    sbatch $uce_smp_sbatch \
        --array=0-$((n_uce - 1))%${uce_smp_throttle} \
        --output "${working_dir}/sbatch_logs/uce_smp_${task}_%A_%a.log" \
        --job-name "${jobname}_uce_smp" \
        "$uce_array_runner" "$uce_smp_manifest" "$allmodels_sh"
else
    rm -f "$uce_smp_manifest"
fi
