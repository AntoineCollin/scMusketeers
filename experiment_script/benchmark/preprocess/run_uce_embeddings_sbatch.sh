#!/bin/sh
#
# Generate X_uce embeddings for all new benchmark datasets using UCE (sbatch)
#

UCE_DIR="/workspace/cell/scMusketeers/experiment_script/benchmark/UCE"
UCE_SCRIPT="${UCE_DIR}/eval_single_anndata.py"
DATA_DIR="/workspace/cell/scMusketeers/data/cellxgene_datasets"
MODEL_LOC="${UCE_DIR}/model_files/33l_8ep_1024t_1280.torch"
BATCH_SIZE=25

sbatch_gpu="--account=cell --partition=gpu --gres=gpu:1 --time=35:00:00"
#cpu_sbatch="--account=cell --partition=cpucourt --time=24:00:00"
log_dir="/workspace/cell/scMusketeers/experiment_script/benchmark/sbatch_logs"
CONDA_INIT="source /softs/miniconda3/etc/profile.d/conda.sh && conda activate UCE"
CONDA_INIT_V100="source /softs/miniconda3/etc/profile.d/conda.sh && conda activate UCE_v100"

for dataset in "SmallIntestine-20k"; do
#for dataset in "CellCards-Lung" "PBMC-Lee" "TS-Blood" "TS-BoneMarrow" "TS-Liver" "TS-Neural" "TS-Skin"; do
    input_path=${DATA_DIR}/${dataset}_uce_input.h5ad
    log_out=${log_dir}/uce_${dataset}.log
    echo sbatch ${sbatch_gpu} --job-name=UCE_${dataset} --output=${log_out} \
        --wrap="${CONDA_INIT_V100} && \
            python3 -c \"
import anndata as ad
adata = ad.read_h5ad('${DATA_DIR}/${dataset}.h5ad')
adata.var_names = adata.var['feature_name']
adata.var.index.name = None
adata.write_h5ad('${input_path}')
\" && \
            cd ${UCE_DIR} && \
            python ${UCE_SCRIPT} \
                --adata_path ${input_path} \
                --dir ${DATA_DIR}/ \
                --species human \
                --model_loc ${MODEL_LOC} \
                --nlayers 33 \
                --batch_size ${BATCH_SIZE}"
done

### Ageing-Mouse-All uses mouse species
# dataset="Ageing-Mouse-All"
# input_path=${DATA_DIR}/${dataset}_uce_input.h5ad
# log_out=${log_dir}/uce_${dataset}.log
# sbatch ${sbatch_gpu} --job-name=UCE_${dataset} --output=${log_out} \
#     --wrap="${CONDA_INIT} && \
#         python3 -c \"
# import anndata as ad
# adata = ad.read_h5ad('${DATA_DIR}/${dataset}.h5ad')
# adata.var_names = adata.var['feature_name']
# adata.var.index.name = None
# adata.write_h5ad('${input_path}')
# \" && \
#         cd ${UCE_DIR} && python ${UCE_SCRIPT} \
#             --adata_path ${input_path} \
#             --dir ${DATA_DIR}/ \
#             --species mouse \
#             --model_loc ${MODEL_LOC} \
#             --nlayers 33 \
#             --batch_size ${BATCH_SIZE}"
