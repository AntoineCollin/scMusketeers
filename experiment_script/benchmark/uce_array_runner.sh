#!/bin/sh
#
# SLURM job-array runner.
#
# Executes a single line of a manifest file, selected by $SLURM_ARRAY_TASK_ID.
# Each manifest line is the complete positional-argument list for a model job
# script (e.g. task1_New_allmodels_benchmark.sh). Used to throttle heavy smp
# UCE jobs: the complete scripts collect one manifest line per (dataset, fold,
# j/seed) and submit this runner as an array with a concurrency cap (--array=...%N).
#
# Usage (via sbatch --array):
#   sbatch --array=0-<N-1>%<throttle> uce_array_runner.sh <manifest> <job_sh>

manifest="$1"
job_sh="$2"

if [ -z "$manifest" ] || [ -z "$job_sh" ]; then
    echo "Usage: $0 <manifest> <job_sh>  (run as a SLURM array job)"
    exit 1
fi

# SLURM_ARRAY_TASK_ID is 0-based; sed lines are 1-based.
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$manifest")
if [ -z "$line" ]; then
    echo "No manifest line for array index ${SLURM_ARRAY_TASK_ID} in ${manifest}"
    exit 1
fi

echo "Array task ${SLURM_ARRAY_TASK_ID}: ${job_sh} ${line}"
# Invoke through the shell so the model job script does not need the execute
# bit (these *_benchmark.sh scripts are normally launched via sbatch, which
# does not require +x). Word-splitting of $line is intentional: it expands to
# the positional args.
# shellcheck disable=SC2086
exec sh "$job_sh" $line
