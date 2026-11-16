#!/bin/sh
#
# Submit sbatch jobs for all missing task2_New benchmark runs.
# Missing runs are read from the CSV produced by 00_benchmark_sbatch_review.py.

working_dir="/workspace/cell/scMusketeers/experiment_script/benchmark"
task="task2_New"
csv_file="${working_dir}/paper_review/missing_benchmark_runs_${task}.csv"
scmusk_sh="${working_dir}/${task}/${task}_scMusk_benchmark.sh"
allmodels_sh="${working_dir}/${task}/${task}_allmodels_benchmark.sh"

jobname="t2n_complete"
cpu_sbatch="--account=cell --partition=cpucourt --time=24:00:00 --mem=128G"
# GPU-capable models (scMusketeers, scanvi, harmony_svm, celltypist) run on the
# gpu partition. 64 GB host RAM normally; 200 GB for the two largest datasets.
gpu_sbatch="--account=cell --partition=gpu --gres=gpu:1 --time=24:00:00 --mem=64G"
gpu_highmem_sbatch="--account=cell --partition=gpu --gres=gpu:1 --time=24:00:00 --mem=200G"
# CPU models (scmap, pca) also load the full matrix and densify it during HVG
# selection; on the two largest datasets 128 GB is not enough, so use a 250 GB
# cpucourt node.
cpu_highmem_sbatch="--account=cell --partition=cpucourt --time=24:00:00 --mem=250G"
# uce loads the full precomputed-embedding AnnData into RAM (the CellCards-Lung /
# SmallIntestine-All uce_input files are 9-14 GB on disk) and densifies it, so it
# needs a lot of memory. The default ~4.75 GB allocation gets OOM-killed (exit 9).
# Small/medium datasets fit in 128 GB on cpucourt; the largest datasets
# (CellCards-Lung, SmallIntestine-All, Ageing-Mouse-All) go to the smp big-memory
# partition (smp01 = 1.5 TB) as a throttled job array.
uce_sbatch="--account=cell --partition=cpucourt --time=24:00:00 --mem=128G"
uce_smp_sbatch="--account=cell --partition=smp --time=24:00:00 --mem=300G"

# smp UCE jobs are collected into a manifest and submitted as a single job array
# with a concurrency cap, so they don't monopolise the (single) smp node.
uce_smp_throttle=6
uce_array_runner="${working_dir}/uce_array_runner.sh"
uce_smp_manifest="${working_dir}/sbatch_logs/uce_smp_manifest_${task}_$(date +%Y%m%d_%H%M%S).txt"

if [ ! -f "$csv_file" ]; then
    echo "Missing CSV not found: $csv_file"
    echo "Run 00_benchmark_sbatch_review.py --task2-new first."
    exit 1
fi

# All task2_New datasets share the same keys
class_key="cell_type"
batch_key="donor_id"

# Group missing runs by (dataset, model, fold, pct), collecting missing seed indices per group.
# This avoids re-running already completed seeds within the same fold+pct combination.
# pct list: 0.05=0  0.1=1  0.5=2  0.9=3  — seed index = seed_value - 30
# Parameter format: fold_pct_seed (e.g. "0_0.1_31")
awk -F',' '
BEGIN {
    pct_nb["0.05"]=0; pct_nb["0.1"]=1; pct_nb["0.5"]=2; pct_nb["0.9"]=3
}
NR > 1 {
    dataset=$1; model=$3; param=$4
    split(param, parts, "_")
    fold=parts[1]; pct=parts[2]; seed_val=parts[3]+0
    seed_idx=seed_val-30
    key=dataset SUBSEP model SUBSEP fold SUBSEP pct_nb[pct]
    if (!(key in seen)) { order[++n]=key; seen[key]=1 }
    seeds[key]=(seeds[key]=="" ? seed_idx : seeds[key]" "seed_idx)
}
END {
    for (k=1; k<=n; k++) {
        split(order[k], f, SUBSEP)
        printf "%s,%s,%s,%s,%s\n", f[1], f[2], f[3], f[4], seeds[order[k]]
    }
}' "$csv_file" | while IFS=',' read -r dataset model fold pct_nb_val seed_ids; do

    case "$dataset" in
        CellCards-Lung|SmallIntestine-All) cpu_opts="$cpu_highmem_sbatch"; gpu_opts="$gpu_highmem_sbatch" ;;
        *)                                 cpu_opts="$cpu_sbatch";         gpu_opts="$gpu_sbatch" ;;
    esac
    for s in $seed_ids; do
        if [ "$model" = "scMusketeers" ]; then
            log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_fold${fold}_pct${pct_nb_val}_s${s}.log"
            sbatch $gpu_opts \
                --output "$log_out" --job-name "${jobname}_${dataset}_${fold}_${pct_nb_val}_s${s}" \
                "$scmusk_sh" "$dataset" "$class_key" "$batch_key" "$fold" "$pct_nb_val" $s
        elif [ "$model" = "scanvi" ] || [ "$model" = "harmony_svm" ] || [ "$model" = "celltypist" ]; then
            # gpu partition; celltypist auto-detects the GPU
            log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold}_pct${pct_nb_val}_s${s}.log"
            sbatch $gpu_opts \
                --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold}_${pct_nb_val}_s${s}" \
                "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold" "$pct_nb_val" $s
        elif [ "$model" = "uce" ]; then
            # high-memory job; the largest datasets go to the smp throttled array,
            # the rest run on cpucourt with 128 GB.
            case "$dataset" in
                CellCards-Lung|SmallIntestine-All|Ageing-Mouse-All)
                    # collect into the manifest -> submitted as a throttled array below
                    echo "$dataset $class_key $batch_key $model $fold $pct_nb_val $s" >> "$uce_smp_manifest"
                    ;;
                *)
                    log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold}_pct${pct_nb_val}_s${s}.log"
                    sbatch $uce_sbatch \
                        --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold}_${pct_nb_val}_s${s}" \
                        "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold" "$pct_nb_val" $s
                    ;;
            esac
        else
            # CPU models (scmap, pca); cpu_opts already picks the memory tier
            log_out="${working_dir}/sbatch_logs/scMusk_sbatch_${task}_${dataset}_${model}_fold${fold}_pct${pct_nb_val}_s${s}.log"
            sbatch $cpu_opts \
                --output "$log_out" --job-name "${jobname}_${dataset}_${model}_${fold}_${pct_nb_val}_s${s}" \
                "$allmodels_sh" "$dataset" "$class_key" "$batch_key" "$model" "$fold" "$pct_nb_val" $s
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
