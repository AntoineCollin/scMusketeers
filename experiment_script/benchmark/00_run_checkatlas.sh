#!/bin/sh
#SBATCH --account=cell
#SBATCH --partition=cpucourt
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --output=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/checkatlas_cellxgene.log
#SBATCH --job-name=checkatlas_cellxgene

module purge
module load miniconda

conda activate /home/cbecavin/.conda/envs/checkatlas

#checkatlas run /workspace/cell/scMusketeers/data/cellxgene_datasets/
checkatlas run /workspace/cell/scMusketeers/data/cellxgene_datasets/ --metric_clust none  --metric_dimred none --metric_annot none

conda deactivate