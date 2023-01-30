#!/bin/bash

#SBATCH --output=logs_DCA_t1_umap.txt

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt


module purge
module load miniconda

conda activate discovair

working_dir=$1
runfile_ID=$2


python $working_dir/compute_single_metrics.py $working_dir $runfile_ID &> log/metrics/log_metrics_$runfile_ID.txt