#!/bin/sh
#
#SBATCH --account=cell     # The account name for the job.
#SBATCH --partition=cpucourt
#SBATCH --time=10:00:00           # The time the job will take to run in D-HH:MM

dataset=$1 

conda activate scanpy_recent 

python compute_PCA_harmony.py $dataset &> pca_$dataset.log
