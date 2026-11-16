#!/bin/bash
#SBATCH --account=cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --job-name=scmusk_tuto
#SBATCH --output=tutorial/sbatch_tutoriel_%j.out
#SBATCH --error=tutorial/sbatch_tutoriel_%j.out

# Activate the sc-musketeers conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scmusk-new

# Run the tutorial
bash tutorial/tutorial.sh
