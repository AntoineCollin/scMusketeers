#!/bin/bash

bash_script_dir="/home/acollin/jobs/dca_jobs/workflow_jobs"
working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
runfile_dir=$working_dir"/runfile_dir"
umap_log_dir=$working_dir"/logs/umap"

for runfile_path in `find $runfile_dir -maxdepth 1 -type f  -name "run*[0-9].yaml"`
do
    NUMBER=$(echo $runfile_path | grep -o -E '[0-9]+')
    log_file=$umap_log_dir"/workflow_ID_"$NUMBER"_DONE.txt"
    if [ ! -f $log_file ]; then
        sbatch --job-name=$NUMBER"_umap" $bash_script_dir'/dca_umap.sh' $working_dir $runfile_path
    fi
done
