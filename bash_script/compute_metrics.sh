#!/bin/bash

#SBATCH --output=comput_metrics.log

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt


module purge
module load miniconda

conda activate discovair

python /home/acollin/dca_permuted_workflow/compute_semisup_metrics.py &> log_metrics.log
