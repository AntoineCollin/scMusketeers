#!/bin/sh
#
conda activate scanpy_recent

for dataset in "yoshida_2021" "tran_2021" "tosti_2021" "litvinukova_2020" "lake_2021" "dominguez_2022_lymph" "dominguez_2022_spleen" "tabula_2022_spleen" "koenig_2022" "ajrccm_by_batch" "htap_final_by_batch" "hlca_par_dataset_harmonized" "hlca_trac_dataset_harmonized";  
do
    sbatch --job-name pca_$dataset one_pca_sbatch.sh $dataset
done
