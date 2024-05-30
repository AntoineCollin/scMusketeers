#!/bin/sh
#

sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_tenx_hlca 04_scPermut_benchmark.sh tenx_hlca ann_finest_level dataset
sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_wmb 04_scPermut_benchmark.sh wmb_full class library_method
sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name t4_it_et 04_scPermut_benchmark.sh wmb_it_et subclass library_method
