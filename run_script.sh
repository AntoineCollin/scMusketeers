#!/bin/bash

#SBATCH --output=batch_entropy.log

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt


module purge
module load miniconda

conda activate discovair

python /home/acollin/dca_permuted_workflow/Batch_entropy_mixing_setup2.py &> Batch_entropy_mixing_setup2.log
