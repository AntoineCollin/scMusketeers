#!/bin/bash

bash_script_dir="/home/acollin/jobs/dca_jobs/workflow_jobs"
working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
runfile_dir=$working_dir"/runfile_dir"
metrics_log_dir=$working_dir"/logs/metrics"

# Computes metrics for all workflows, should take a while, don't use lightly.

for runfile_path in `find $runfile_dir -maxdepth 1 -type f  -name "run*[0-9].yaml"`
do
    NUMBER=$(echo $runfile_path | grep -o -E '[0-9]+')
    log_file=$metrics_log_dir"/workflow_ID_"$NUMBER"_DONE.txt"
    if [ ! -f $log_file ]; then
        sbatch --job-name=$NUMBER"_metrics" $bash_script_dir'/dca_metrics.sh' $working_dir $NUMBER
    fi
done
