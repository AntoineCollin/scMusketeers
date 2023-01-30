#!/bin/bash

bash_script_dir="/home/acollin/jobs/dca_jobs/workflow_jobs"
working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
runfile_dir=$working_dir"/runfile_dir"
run_log_dir=$working_dir"/logs/run"
dca_sin_path=$working_dir"/singularity_dca.sif"
scvi_sin_path=$working_dir"/scvi_sin.sif"
scvi_scarches_LCA_path=$working_dir"/singularity_scarches_LCA.sif"
# scvi_sin_path=$working_dir"/singularity_dca_pytorch.sif"

for runfile_path in `find $runfile_dir -maxdepth 1 -type f  -name "run*[0-9].yaml"`
do
    model=`grep -A1 'model_spec:' $runfile_path | tail -n1 | awk '{ print $2}'`
    dataset=`grep -A1 'model_spec:' $runfile_path | tail -n1 | awk '{ print $2}'`
    NUMBER=$(echo $runfile_path | grep -o -E '[0-9]+')
    log_file=$run_log_dir"/workflow_ID_"$NUMBER"_DONE.txt"
    if [ ! -f $log_file ]; then
        if [[ "$dataset" == *"disco"* ]]; then 
            if [ $model = "scanvi" ]; then
                sbatch --job-name=$NUMBER"_workflow" --mem-per-gpu=150G $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $scvi_sin_path
            fi
            if [ $model = "dca_permuted" ]; then
                sbatch --job-name=$NUMBER"_workflow" --mem-per-gpu=150G $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
            fi
            if [ $model = "dca_into_perm" ]; then
                sbatch --job-name=$NUMBER"_workflow" --mem-per-gpu=150G $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
            fi
            if [ $model = "contrastive" ]; then
                sbatch --job-name=$NUMBER"_workflow" --mem-per-gpu=150G $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
            fi
            if [ $model = "scarches_scanvi_LCA" ]; then
                echo 'in da loop'
                sbatch --job-name=$NUMBER"_workflow" --mem-per-gpu=150G $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $scarches_LCA_sin_path
            fi
        else
            if [ $model = "scanvi" ]; then
                sbatch --job-name=$NUMBER"_workflow" $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $scvi_sin_path
            fi
            if [ $model = "dca_permuted" ]; then
                sbatch --job-name=$NUMBER"_workflow" $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
            fi
            if [ $model = "dca_into_perm" ]; then
                sbatch --job-name=$NUMBER"_workflow" $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
            fi
            if [ $model = "contrastive" ]; then
                sbatch --job-name=$NUMBER"_workflow" $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $dca_sin_path
            fi
            if [ $model = "scarches_scanvi_LCA" ]; then
                echo 'in da loop'
                sbatch --job-name=$NUMBER"_workflow" $bash_script_dir'/dca_workflow.sh' $working_dir $singularity_working_dir $runfile_path $scarches_LCA_sin_path
            fi
        fi
    fi

done