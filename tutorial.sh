dataset=data/ajrccm_by_batch.h5ad
outdir=/data/analysis/data_becavin/scpermut_test/tutorial

python scpermut/__main__.py transfer $dataset --class_key=celltype --unlabeled_category=Basal --batch_key=manip --out=$outdir
# echo scpermut transfer $dataset --class_key=celltype --batch_key=manip --out=$outdir


create test dataset with unlabeled cells
Create an argument to automatically unlabeled cells.
