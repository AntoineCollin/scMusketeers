#!/bin/sh
#SBATCH --account=cell
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=test_conda
#SBATCH --output=/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs/test_conda.log

source /softs/miniconda3/etc/profile.d/conda.sh
conda activate UCE

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
