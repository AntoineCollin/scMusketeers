#!/bin/bash

#SBATCH --job-name=test_workflow
#SBATCH --output=test_workflow.txt

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


module purge
module load singularity
module load cuda/10.0

working_dir="/home/acollin/dca_permuted_workflow"
singularity_working_dir="/data/dca_permuted_workflow"
runfile_ID="runfile_ID_"$1
runfile_path=$working_dir"/runfile_dir/"$runfile_ID".yaml"
singularity_path=$working_dir"/singularity_dca.sif"

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/run_workflow.py $singularity_working_dir $runfile_path &> log/log_workflow_$runfile_ID.txt
