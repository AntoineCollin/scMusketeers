#!/bin/sh
#

for dataset in "tabula_2022_spleen" "dominguez_2022_lymph" "tran_2021" "dominguez_2022_spleen"  "lake_2021" "yoshida_2021" #"tosti_2021"    "litvinukova_2020" ; #  "koenig_2022"
do
    sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t2_scper_$dataset 02_scPermut_benchmark.sh $dataset Original_annotation batch
done

sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t2_scper_ajrccm_by_batch 02_scPermut_benchmark.sh ajrccm_by_batch celltype manip
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t2_scper_htap 02_scPermut_benchmark.sh htap ann_finest_level donor
# # sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t2_scper_hlca_par_dataset_harmonized 02_scPermut_benchmark.sh hlca_par_dataset_harmonized ann_finest_level dataset
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t2_scper_hlca_trac_dataset_harmonized 02_scPermut_benchmark.sh hlca_trac_dataset_harmonized ann_finest_level dataset