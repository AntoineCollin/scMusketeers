#!/bin/sh
#

for dataset in "yoshida_2021" "tosti_2021" "lake_2021" "tabula_2022_spleen" "dominguez_2022_spleen" "dominguez_2022_lymph" "koenig_2022" "litvinukova_2020" ; #   "tran_2021"
do
    sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t1_scp_$dataset 01_scPermut_benchmark.sh $dataset Original_annotation batch
done

sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t1_scp_ajrccm_by_batch 01_scPermut_benchmark.sh ajrccm_by_batch celltype manip
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t1_scp_htap 01_scPermut_benchmark.sh htap ann_finest_level donor
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t1_scper_hlca_par_dataset_harmonized 01_scPermut_benchmark.sh hlca_par_dataset_harmonized ann_finest_level dataset
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t1_scper_hlca_trac_dataset_harmonized 01_scPermut_benchmark.sh hlca_trac_dataset_harmonized ann_finest_level dataset