#!/bin/sh
#

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_tenx benchmark_task_4.sh tenx_hlca ann_finest_level dataset True

# sbatch ---partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method

# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_tenx benchmark_task_4.sh tenx_hlca ann_finest_level dataset False

# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method False
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method False

sbatch --partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method False
sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method False

sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method  True
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method True
