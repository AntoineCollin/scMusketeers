#!/bin/sh
#

sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_cell_tenx_lv3 04_scPermut_benchmark.sh tenx_hlca_par_cell ann_level_3 dataset
# sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_tenx_lv4 04_scPermut_benchmark.sh tenx_hlca_par ann_level_4 dataset
sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_nuc_tenx_lv3 04_scPermut_benchmark.sh tenx_hlca_par_nuc ann_level_3 dataset
# sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_tenx_lv4 04_scPermut_benchmark.sh tenx_hlca_par ann_level_4 dataset

# sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_wmb 04_scPermut_benchmark.sh wmb_full class library_method
# sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_it_et 04_scPermut_benchmark.sh wmb_it_et subclass library_method
