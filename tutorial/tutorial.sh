outdir="/data/analysis/data_becavin/scmusketeers"
dataset=${outdir}"/data/Deprez-Lung-unknown-0.2.h5ad"
outname=Deprez-Lung-unknown-0.2-pred
classkey="celltype"
unlabeled="Unknown"
batchkey="donor"
#dataset=${outdir}"/data/CellTypist-Lung-unknown-0.2.h5ad"
#outname="CellTypist-Lung-unknown-0.2-pred"
#classkey="cell_type"
#batchkey="donor_id"

ref_dataset=data/Deprez-2020-ref-batch-0.2.h5ad
query_dataset=data/Deprez-2020-query-batch-0.2.h5ad
outname_query="Deprez-2020-query-0.2-pred"

warmup_epoch=2   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=2   # default = 100, help = Number of epoch to train full model
permonly_epoch=5   # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=2   # default = 50, help = Number of epoch to train te classifier only

log_neptune=False
neptune_name="sc-musketeers"

##### Sampling_percentage 20%
# Transfer Cell annotation to all Unknown cells
#sc-musketeers transfer ${dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name=${outname}
# sc-musketeers transfer ${dataset} --class_key=${classkey} --unlabeled_category=${unlabeled} --batch_key=${batchkey} --out_dir=${outdir} --out_name=${outname} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
python scmusketeers/__main__.py transfer ${dataset} --log_neptune=${log_neptune} --neptune_name=${neptune_name} --class_key=${classkey} --unlabeled_category=${unlabeled} --batch_key=${batchkey} --out_dir=${outdir} --out_name=${outname} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
# python sc-musketeers/__main__.py transfer ${dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name=${outname}

# Transfer Cell annotation and remove batch to query adata
#sc-musketeers transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name=${outname_query} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
# python sc-musketeers/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out_dir=${outdir} --out_name={outname_query}

##### Sampling_percentage 40%
# Transfer Cell annotation to all Unknown cells
#python sc-musketeers/__main__.py transfer $dataset --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

# Transfer Cell annotation and remove batch to query adata
# python sc-musketeers/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir
