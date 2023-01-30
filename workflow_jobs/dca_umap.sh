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
runfile_ID=$(basename $2 .yaml)


python $working_dir/run_umap.py $working_dir $2 &> log/umap/log_umap_$runfile_ID.txt