dataset=/home/acollin/data/ajrccm_by_batch.h5ad
outdir=/home/acollin/scPermut/tutorial

python scpermut/__main__.py transfer $dataset --class_key=celltype --batch_key=manip --out=$outdir
# echo scpermut transfer $dataset --class_key=celltype --batch_key=manip --out=$outdir