#!/bin/sh
#

for dataset in "litvinukova_2020"  # "tran_2021" "tabula_2022_spleen"  "yoshida_2021" "dominguez_2022_spleen" "tosti_2021" "dominguez_2022_lymph" "lake_2021" "koenig_2022";  #    ; # 
do
    sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t3_scper_$dataset 03_scPermut_benchmark.sh $dataset Original_annotation batch
done

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t3_scper_ajrccm_by_batch 03_scPermut_benchmark.sh ajrccm_by_batch celltype manip
# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t3_scper_htap 03_scPermut_benchmark.sh htap ann_finest_level donor
sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t3_scper_hlca_par_dataset_harmonized 03_scPermut_benchmark.sh hlca_par_dataset_harmonized ann_finest_level dataset
# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name t3_scper_hlca_trac_dataset_harmonized 03_scPermut_benchmark.sh hlca_trac_dataset_harmonized ann_finest_level dataset