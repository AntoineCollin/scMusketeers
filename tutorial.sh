dataset=data/ajrccm_by_batch.h5ad
outdir=/data/analysis/data_becavin/scpermut_test/tutorial

##### Sampling_percentage 20%
# Transfer Cell annotation to all Unknown cells
python scpermut/__main__.py transfer $dataset --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

# Transfer Cell annotation and remove batch to query adata
#python scpermut/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

##### Sampling_percentage 40%
# Transfer Cell annotation to all Unknown cells
#python scpermut/__main__.py transfer $dataset --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

# Transfer Cell annotation and remove batch to query adata
# python scpermut/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

