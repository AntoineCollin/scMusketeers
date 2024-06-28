dataset=data/ajrccm_by_batch.h5ad
outdir=/home/becavin/scPermuts

python scpermut/__main__.py transfer $dataset --class_key=celltype --unlabeled_category=Basal --batch_key=manip --out=$outdir
# echo scpermut transfer $dataset --class_key=celltype --batch_key=manip --out=$outdir