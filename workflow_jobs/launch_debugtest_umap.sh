#!/bin/bash

#SBATCH --job-name=t1_umap
#SBATCH --output=logs_DCA_t1_umap.txt

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt


module purge
module load miniconda

conda activate discovair

working_dir="/home/acollin/dca_permuted_workflow"
runfile_ID="runfile_ID_"$1
runfile_path=$working_dir"/runfile_dir/"$runfile_ID".yaml"

python $working_dir/run_umap.py $working_dir $runfile_path &> log/log_umap_$runfile_ID.txt