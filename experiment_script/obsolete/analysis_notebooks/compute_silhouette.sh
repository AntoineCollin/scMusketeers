#!/bin/sh
#SBATCH --account=cell     # The account name for the job.
#SBATCH --output=/home/acollin/compute_metrics.log # Important to retrieve the port where the notebook is running, if not included a slurm file with the job-id will be outputted. 
#SBATCH --partition=cpucourt 
#SBATCH --time=71:00:00 
#SBATCH --job-name compute_metrics
#SBATCH --output /home/acollin/scPermut/analysis_notebooks/log.log 


python /home/acollin/scPermut/analysis_notebooks/compute_silhouette.py &> /home/acollin/scPermut/analysis_notebooks/compute_silhouette.log 