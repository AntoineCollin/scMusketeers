#!/bin/sh
#
# Generate X_uce embeddings for all new benchmark datasets using UCE
#

UCE_DIR="/workspace/cell/scMusketeers/experiment_script/benchmark/UCE"
UCE_SCRIPT="${UCE_DIR}/eval_single_anndata.py"
DATA_DIR="/workspace/cell/scMusketeers/data/cellxgene_datasets"
MODEL_LOC="${UCE_DIR}/model_files/33l_8ep_1024t_1280.torch"
BATCH_SIZE=25

#module load miniconda && conda activate UCE
#cd ${UCE_DIR}

for dataset in "TS-Neural"; do
#for dataset in "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin"; do
    input_path=${DATA_DIR}/${dataset}_uce_input.h5ad
    python3 -c "
import anndata as ad
adata = ad.read_h5ad('${DATA_DIR}/${dataset}.h5ad')
adata.var_names = adata.var['feature_name']
adata.var.index.name = None
adata.write_h5ad('${input_path}')
print('Saved preprocessed:', '${input_path}')
"
    cd ${UCE_DIR} && python ${UCE_SCRIPT} \
        --adata_path ${input_path} \
        --dir ${DATA_DIR}/ \
        --species human \
        --model_loc ${MODEL_LOC} \
        --nlayers 33 \
        --batch_size ${BATCH_SIZE}
done

# Ageing-Mouse-All uses mouse species
# dataset="Ageing-Mouse-All"
# input_path=${DATA_DIR}/${dataset}_uce_input.h5ad
# python3 -c "
# import anndata as ad
# adata = ad.read_h5ad('${DATA_DIR}/${dataset}.h5ad')
# adata.var_names = adata.var['feature_name']
# adata.var.index.name = None
# adata.write_h5ad('${input_path}')
# print('Saved preprocessed:', '${input_path}')
# "
# python ${UCE_SCRIPT} \
#     --adata_path ${input_path} \
#     --dir ${DATA_DIR} \
#     --species mouse \
#     --model_loc ${MODEL_LOC} \
#     --nlayers 33 \
#     --batch_size ${BATCH_SIZE}
