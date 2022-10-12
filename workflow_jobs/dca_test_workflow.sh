#!/bin/bash

#SBATCH --job-name=test_workflow
#SBATCH --output=test_workflow.txt

#SBATCH --ntasks=1
#SBATCH --time=0-20:00:00
#SBATCH --account=cell
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH --mail-user=collin@ipmc.cnrs.fr
#SBATCH --mail-type=BEGIN,END,FAIL

module purge
module load singularity
module load cuda/10.0

singularity exec --nv --bind /home/acollin/dca_permuted:/data/dca_permuted /home/acollin/dca_permuted/singularity_dca.sif python /home/acollin/dca_permuted/workflow/test_workflow.py &> log/log_test_workflow.txt
