#!/bin/bash

#SBATCH --output=test_workflow.txt

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu


module purge
module load singularity
module load cuda/10.0

working_dir=$1
singularity_working_dir=$2
runfile_ID=$(basename $3 .yaml)
singularity_path=$working_dir"/singularity_dca.sif"

singularity exec --nv --bind $working_dir:$singularity_working_dir $singularity_path python $working_dir/run_predict.py $singularity_working_dir $3 &> log/predict/log_predict_$runfile_ID.txt
