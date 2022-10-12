#!/bin/bash

#SBATCH --job-name=leidenx
#SBATCH --output=leidenx.txt

#SBATCH --ntasks=1
#SBATCH --time=0-15:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt

#SBATCH --mail-user=collin@ipmc.cnrs.fr
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load miniconda

conda activate discovair

python /home/acollin/dca_permuted_workflow/compute_leiden.py &> full_log_compute_leiden.txt
