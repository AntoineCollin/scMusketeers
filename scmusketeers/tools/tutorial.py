import os

import numpy as np
import scanpy as sc
from utils import split_adata


def create_tuto_data(
    sampling_percentage,
    path_adata,
    name,
    class_key,
    batch_key,
    unlabeled_category,
):
    adata = sc.read_h5ad(path_adata)
    adata_train, adata_test = split_adata(
        adata, batch_key, 1 - sampling_percentage
    )

    # Compute UMAP
    sc.tl.pca(adata_test, svd_solver="arpack")
    sc.pp.neighbors(
        adata_test, n_neighbors=10, n_pcs=40
    )  # UMAP is based on the neighbor graph; we'll compute this first
    sc.tl.umap(adata_test)
    sc.pl.umap(adata_test, color="celltype")

    # add unlabeled_category to celltype
    celltype = adata_test.obs[class_key]
    celltype = celltype.cat.add_categories(unlabeled_category)
    # print(celltype.cat.categories)

    #### Celltype annotation

    # unassigned X% of cells, randomly
    list_index = range(len(celltype))
    list_index_select = np.random.choice(
        list_index,
        size=int(len(list_index) * sampling_percentage),
        replace=False,
    )
    celltype.iloc[list_index_select] = unlabeled_category
    print(celltype)
    adata_test.obs[class_key] = celltype
    path_new_adata = os.path.join(
        "data", f"{name}-unknown-{sampling_percentage}.h5ad"
    )
    adata_test.write_h5ad(path_new_adata)

    # unassigned X% of cells, keeping cell type distribution (Useful ???)

    #### Batch correction
    adata_train, adata_test = split_adata(
        adata, batch_key, 1 - sampling_percentage
    )
    # separate cells from 2X% of batches
    print(np.arange(adata.n_obs))
    # sample adata to reduce its size
    all_batches = adata_test.obs[batch_key].unique()
    print(len(all_batches))
    test_batches = np.random.choice(
        all_batches, int(len(all_batches) * sampling_percentage), replace=False
    )
    print(test_batches)
    print(len(test_batches))
    train_batches = list(set(all_batches).difference(test_batches))
    adata_ref = adata_test[adata_test.obs[batch_key].isin(train_batches)]
    adata_query = adata_test[adata_test.obs[batch_key].isin(test_batches)]
    print(adata_ref.obs[batch_key])
    print(adata_query.obs[batch_key])

    # Compute PCA and UMAP
    sc.tl.pca(adata_ref, svd_solver="arpack")
    sc.pp.neighbors(
        adata_ref, n_neighbors=10, n_pcs=40
    )  # UMAP is based on the neighbor graph; we'll compute this first
    sc.tl.umap(adata_ref)

    sc.tl.pca(adata_query, svd_solver="arpack")
    sc.pp.neighbors(
        adata_query, n_neighbors=10, n_pcs=40
    )  # UMAP is based on the neighbor graph; we'll compute this first
    sc.tl.umap(adata_query)

    # remove celltype from query
    celltype = adata_query.obs[class_key]
    celltype = celltype.cat.set_categories([unlabeled_category])
    celltype.loc[celltype.index] = unlabeled_category
    adata_query.obs[class_key] = celltype

    ### Ref and query are separated
    path_new_adata = os.path.join(
        "data", f"{name}-ref-batch-{sampling_percentage}.h5ad"
    )
    adata_ref.write_h5ad(path_new_adata)
    path_new_adata = os.path.join(
        "data", f"{name}-query-batch-{sampling_percentage}.h5ad"
    )
    adata_query.write_h5ad(path_new_adata)

    ### Ref and query on same dataset (Useful ????)


if __name__ == "__main__":
    path_adata = "data/ajrccm_by_batch.h5ad"
    name = "Deprez-2020"
    class_key = "celltype"
    unlabeled_category = "Unknown"
    batch_key = "manip"

    sampling_percentage = 0.2
    create_tuto_data(
        sampling_percentage,
        path_adata,
        name,
        class_key,
        batch_key,
        unlabeled_category,
    )

    sampling_percentage = 0.4
    create_tuto_data(
        sampling_percentage,
        path_adata,
        name,
        class_key,
        batch_key,
        unlabeled_category,
    )
