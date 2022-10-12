#!/bin/bash

#SBATCH --job-name=t1_umap
#SBATCH --output=logs_DCA_t1_umap.txt

#SBATCH --ntasks=1
#SBATCH --time=0-02:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt

#SBATCH --mail-user=collin@ipmc.cnrs.fr
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load miniconda

conda activate discovair

python /home/acollin/dca_permuted/tasks_scripts/task_1_umap.py &> log/log_dca_semi_sup_t1_umap.txt
