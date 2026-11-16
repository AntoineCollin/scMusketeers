#!/bin/sh
#

# for dataset in  "dominguez_2022_lymph" "tosti_2021"; #   #"tran_2021" "tabula_2022_spleen"   # "yoshida_2021"
# do
#     sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name hp_$dataset 01_hp_optim.sh $dataset Original_annotation batch
# done

# for dataset in  "litvinukova_2020" ; #"koenig_2022"; #"lake_2021" # "dominguez_2022_spleen" #  ; # 
# do
#     sbatch --partition=gpu --gres=gpu:1  --nodelist=gpu03 --time=36:00:00 --job-name hp_$dataset 01_hp_optim.sh $dataset Original_annotation batch
# done

# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name hp_ajrccm_by_batch 01_hp_optim.sh ajrccm_by_batch celltype manip
# sbatch --partition=gpu --gres=gpu:1 --time=35:00:00 --job-name hp_htap 01_hp_optim.sh htap ann_finest_level donor
sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name hp_hlca_par_dataset_harmonized 01_hp_optim.sh hlca_par_dataset_harmonized ann_finest_level dataset
# sbatch --partition=gpu --gres=gpu:1 --nodelist=gpu03 --time=35:00:00 --job-name hp_hlca_trac_dataset_harmonized 01_hp_optim.sh hlca_trac_dataset_harmonized ann_finest_level dataset
