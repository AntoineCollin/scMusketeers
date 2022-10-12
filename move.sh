#!/bin/bash


#SBATCH --job-name=menage
#SBATCH --output=menage.txt

#SBATCH --ntasks=1
#SBATCH --time=0-15:00:00
#SBATCH --account=cell
#SBATCH --partition=cpucourt

#SBATCH --mail-user=collin@ipmc.cnrs.fr
#SBATCH --mail-type=BEGIN,END,FAIL

mv `cat /home/acollin/dca_permuted_workflow/corrected_counts_list.txt` /workspace/cell/dca_permuted_archive/results
