dataset=data/Deprez-2020-unknown-0.2.h5ad
ref_dataset=data/Deprez-2020-ref-batch-0.2.h5ad
query_dataset=data/Deprez-2020-query-batch-0.2.h5ad
outdir=/data/analysis/data_becavin/scpermut_test/tutorial
outname="Deprez-2020-unknown-0.2-pred"
outname_query="Deprez-2020-query-0.2-pred"
warmup_epoch=5   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=5   # default = 100, help = Number of epoch to train full model
permonly_epoch=5   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=5   # default = 50, help = Number of epoch to train te classifier only


##### Sampling_percentage 20%
# Transfer Cell annotation to all Unknown cells
#sc-musketeers transfer ${dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name=${outname}
# python sc-musketeers/__main__.py transfer ${dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name=${outname}

# Transfer Cell annotation and remove batch to query adata
sc-musketeers transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name=${outname_query} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
# python sc-musketeers/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name={outname_query}

##### Sampling_percentage 40%
# Transfer Cell annotation to all Unknown cells
#python sc-musketeers/__main__.py transfer $dataset --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

# Transfer Cell annotation and remove batch to query adata
# python sc-musketeers/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir
