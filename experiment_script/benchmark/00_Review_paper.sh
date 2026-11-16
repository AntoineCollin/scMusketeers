#srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate
sc-musketeers --version

# Running task1 for scmusketeers and all datasets
sh experiment_script/benchmark/task1/task1_scMusk_all_dataset.sh 
#01_scPermut_all_dataset.sh and 01_all_benchmark.sh are equivalent

# Running task1 for all other models and all datasets
sh experiment_script/benchmark/task1/task1_all_benchmark.sh

# Running task2 for scmusketeers and all datasets
sh experiment_script/benchmark/task2/task2_scMusk_all_dataset.sh 
#01_scPermut_all_dataset.sh and 01_all_benchmark.sh are equivalent in their function and structure

# Running task2 for all other models and all datasets
sh experiment_script/benchmark/task2/task2_all_benchmark.sh

# Get the list of jobs which should have been run for task1 and task2
python experiment_script/benchmark/00_benchmark_sbatch_review.py

# Rerun missing jobs
sh experiment_script/benchmark/task1_New/task1_New_all_benchmark_complete.sh
sh experiment_script/benchmark/task2_New/task2_New_all_benchmark_complete.sh

# Run checkatlas on all datasets
sbatch experiment_script/benchmark/00_run_checkatlas.sh

# Create figures new and old
python analysis_notebooks/figure_generation.py --process_csv --task1_new
python analysis_notebooks/figure_generation.py --process_csv --task2_new

python analysis_notebooks/figure_generation.py --task1
python analysis_notebooks/figure_generation.py --task2
python analysis_notebooks/figure_generation.py --task1_new
python analysis_notebooks/figure_generation.py --task2_new

python analysis_notebooks/generate_old_figure_2.py


