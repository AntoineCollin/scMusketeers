#!/bin/sh
#

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_bench_tenx_lv3 benchmark_task_4.sh tenx_hlca_par ann_level_3 dataset True
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_bench_tenx_lv3 benchmark_task_4.sh tenx_hlca_par ann_level_3 dataset False

sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_cell_lv3 benchmark_task_4.sh tenx_hlca_par_cell ann_level_3 dataset True
sbatch --partition=cpucourt --time=71:00:00 --job-name t4_cell_lv3 benchmark_task_4.sh tenx_hlca_par_cell ann_level_3 dataset False

sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_nuc_lv3 benchmark_task_4.sh tenx_hlca_par_nuc ann_level_3 dataset True
sbatch --partition=cpucourt --time=71:00:00 --job-name t4_nuc_lv3 benchmark_task_4.sh tenx_hlca_par_nuc ann_level_3 dataset False

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_bench_tenx_lv4 benchmark_task_4.sh tenx_hlca_par ann_level_4 dataset True
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_bench_tenx_lv4 benchmark_task_4.sh tenx_hlca_par ann_level_4 dataset False

# sbatch ---partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method


# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method False
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method False

# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method False
# sbatch --partition=cpucourt --time=71:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method False

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_wmb benchmark_task_4.sh wmb_full class library_method  True
# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t4_it_et benchmark_task_4.sh wmb_it_et subclass library_method True
