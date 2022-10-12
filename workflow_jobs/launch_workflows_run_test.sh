#!/bin/bash

bash_script_dir="/home/acollin/jobs/dca_jobs/workflow_jobs"
working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
runfile_dir=$working_dir"/runfile_dir"
run_log_dir=$working_dir"/logs/run"
dca_sin_path=$working_dir"/singularity_dca.sif"
scvi_sin_path=$working_dir"/scvi_sin.sif"


for runfile_path in `find $runfile_dir -maxdepth 1 -type f  -name "run*[0-9].yaml"`
do
    model=`grep -A1 'model_spec:' $runfile_path | tail -n1 | awk '{ print $2}'`
    NUMBER=$(echo $runfile_path | grep -o -E '[0-9]+')
    log_file=$run_log_dir"/workflow_ID_"$NUMBER"_DONE.txt"
    if [ ! -f $log_file ]; then
        if [ $model = "scanvi" ]; then
            sbatch $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $scvi_sin_path
        fi
        if [ $model = "dca_permuted" ]; then
            sbatch $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
        fi
    fi
done