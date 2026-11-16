# Get GPU
# srun -A cell -p gpu -t 10:00:00 --gres=gpu:1 --pty bash -i
# source ~/.cache/pypoetry/virtualenvs/sc-musketeers-voskaBul-py3.12/bin/activate

outdir="/workspace/cell/scMusketeers"
#outdir="/data/analysis/data_becavin/scMusketeers-data"
#scmusk_path="/data/analysis/data_becavin/scMusketeers"
scmusk_path=${outdir}
dataset=${outdir}"/data/Deprez-Lung-unknown-0.2.h5ad"
outname=Deprez-Lung-unknown-0.2-pred
classkey="celltype"
unlabeled="Unknown"
batchkey="donor"
#dataset=${outdir}"/data/CellTypist-Lung-unknown-0.2.h5ad"
#outname="CellTypist-Lung-unknown-0.2-pred"
#classkey="cell_type"
#batchkey="donor_id"

ref_dataset=${outdir}/data/Deprez-Lung-ref-batch-0.2.h5ad
query_dataset=${outdir}/data/Deprez-Lung-query-batch-0.2.h5ad
outname_query=${outdir}/data/Deprez-Lung-transfer-0.2-pred

#warmup_epoch=2   # default 100, help - Number of epoch to warmup DANN
#fullmodel_epoch=2   # default = 100, help = Number of epoch to train full model
#permonly_epoch=2 # default = 100, help = Number of epoch to train in permutation only mode
#classifier_epoch=2   # default = 50, help = Number of epoch to train te classifier only
warmup_epoch=20   # default 100, help - Number of epoch to warmup DANN
fullmodel_epoch=50   # default = 100, help = Number of epoch to train full model
permonly_epoch=50 # default = 100, help = Number of epoch to train in permutation only mode
classifier_epoch=50   # default = 50, help = Number of epoch to train te classifier only

training_scheme="training_scheme_8"
bestparam_path=${scmusk_path}"/experiment_script/hyperparam/hp_best_ajrccm.csv"
#bestparam_path=${scmusk_path}"/experiment_script/hyperparam/hp_best_ajrccm.csv"
#bestparam_path=${scmusk_path}"/experiment_script/hyperparam/default_df_t11.csv"

##### Sampling_percentage 20%
# Transfer Cell annotation to all Unknown cells
sc-musketeers transfer ${dataset} --debug --training_scheme=${training_scheme} --class_key=${classkey} --unlabeled_category=${unlabeled} --batch_key=${batchkey} --out_dir=${outdir} --out_name=${outname} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
# With neptune log
#sc-musketeers transfer ${dataset} --debug --log_neptune=${log_neptune} --neptune_name=${neptune_name} --class_key=${classkey} --unlabeled_category=${unlabeled} --batch_key=${batchkey} --out_dir=${outdir} --out_name=${outname} --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
# With best param
#sc-musketeers transfer ${dataset} --debug --log_neptune=${log_neptune} \
#    --neptune_name=${neptune_name} --training_scheme=${training_scheme} \
#    --class_key=${classkey} --unlabeled_category=${unlabeled} --batch_key=${batchkey} --out_dir=${outdir} --out_name=${outname} \
#    --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch}
    # > nohup_transfer.out &

# Transfer Cell annotation and remove batch to query adata
#nohup sc-musketeers transfer ${ref_dataset} --query_path ${query_dataset} --debug --log_neptune=${log_neptune} \
#    --neptune_name=${neptune_name} --bestparam_path=${bestparam_path} --training_scheme=${training_scheme} \
#    --class_key=${classkey} --unlabeled_category=${unlabeled} --batch_key=${batchkey} --out_dir=${outdir} --out_name=${outname_query} \
#    --warmup_epoch=${warmup_epoch} --fullmodel_epoch=${fullmodel_epoch} --permonly_epoch=${permonly_epoch} --classifier_epoch=${classifier_epoch} > nohup_transfer_ref_query.out &


##### Sampling_percentage 40%
# Transfer Cell annotation to all Unknown cells
#python sc-musketeers/__main__.py transfer $dataset --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir

# Transfer Cell annotation and remove batch to query adata
# python sc-musketeers/__main__.py transfer ${ref_dataset} --query_path ${query_dataset} --class_key=celltype --unlabeled_category="Unknown" --batch_key=manip --out=$outdir