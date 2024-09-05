import os
import sys

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

sys.path.insert(1, os.path.join(sys.path[0], ".."))

try:
    from ..tools.utils import densify
except ImportError:
    from tools.utils import densify


def get_hvg_common(
    adata_, n_hvg=2000, flavor="seurat", batch_key="manip", reduce_adata=True
):
    """
    Computes a list of hvg which are common to the most different batches.
    """
    adata = adata_.copy()
    check_nnz = np.asarray(adata.X[adata.X != 0][0])
    while len(check_nnz.shape) >= 1:  # Get first non zero element of X
        check_nnz = check_nnz[0]
    if int(check_nnz) == float(check_nnz):
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    n_batches = len(adata.obs[batch_key].unique())

    sc.pp.highly_variable_genes(
        adata=adata, n_top_genes=n_hvg, flavor=flavor, batch_key=batch_key
    )
    dispersion_nbatches = adata.var["dispersions_norm"][
        adata.var["highly_variable_nbatches"] == n_batches
    ]  # Starts with genes hvg for every batches
    dispersion_nbatches = dispersion_nbatches.sort_values(ascending=False)
    print(f"Searching for highly variable genes in {n_batches} batches")
    if (
        len(dispersion_nbatches) >= n_hvg
    ):  # If there are already enough hvg in every batch, returns them
        top_genes = list(dispersion_nbatches[:n_hvg].index)
        if reduce_adata:
            return adata[:, top_genes]
        else:
            return top_genes
    print(
        f"Found {len(dispersion_nbatches)} highly variable genes using {n_batches} batches"
    )
    top_genes = list(
        dispersion_nbatches.index
    )  # top_genes is the final selection of genes
    remaining_genes = n_hvg
    enough = False

    while not enough:
        n_batches = (
            n_batches - 1
        )  # Looks for genes hvg for one fewer batch at each iteration
        print(f"Searching for highly variable genes in {n_batches} batches")
        remaining_genes = n_hvg - len(top_genes)  # nb of genes left to find
        dispersion_nbatches = adata.var["dispersions_norm"][
            adata.var["highly_variable_nbatches"] == n_batches
        ]
        dispersion_nbatches = dispersion_nbatches.sort_values(ascending=False)
        if (
            len(dispersion_nbatches) > remaining_genes
        ):  # Enough genes to fill in the rest
            print(
                f"Found {len(dispersion_nbatches)} highly variable genes using {n_batches} batches. Selecting top {remaining_genes}"
            )
            # print(dispersion_nbatches)
            top_genes += list(dispersion_nbatches[:remaining_genes].index)
            enough = True
        else:
            print(
                f"Found {len(dispersion_nbatches)} highly variable genes using {n_batches} batches"
            )
            top_genes += list(dispersion_nbatches.index)

    if reduce_adata:
        return adata_[:, top_genes]

    return top_genes


def load_dataset(ref_path, query_path, class_key, unlabeled_category):
    if not query_path:
        adata = sc.read_h5ad(ref_path)
    else:
        ref = sc.read_h5ad(ref_path)
        query = sc.read_h5ad(query_path)
        query.obs[class_key] = unlabeled_category
        adata = ref.concatenate(query, join="inner")
    if not adata.raw:
        adata.raw = adata
    return adata


class Dataset:
    def __init__(
        self,
        adata,  # full anndata, query is the cells which have unlabeled_category as class_key.
        class_key,
        batch_key,
        filter_min_counts,
        normalize_size_factors,
        size_factor,
        scale_input,
        logtrans_input,
        use_hvg,
        unlabeled_category,
        train_test_random_seed,
    ):
        self.adata = adata
        self.class_key = class_key
        self.adata.obs[f"true_{self.class_key}"] = self.adata.obs[
            self.class_key
        ]  # Duplicate original annotation in true_class_key
        self.batch_key = batch_key
        self.filter_min_counts = filter_min_counts
        self.normalize_size_factors = normalize_size_factors
        self.size_factor = size_factor
        self.scale_input = scale_input
        self.logtrans_input = logtrans_input
        self.use_hvg = use_hvg
        # if not semi_sup:
        #     self.semi_sup = True # semi_sup defaults to True, especially for scANVI
        self.unlabeled_category = unlabeled_category
        self.adata_train_extended = (
            anndata.AnnData()
        )  # Contains the train and to-be-defined val data
        self.adata_test = anndata.AnnData()  # The query data
        self.adata_train = anndata.AnnData()
        self.adata_val = anndata.AnnData()
        self.adta_test = anndata.AnnData()
        self.train_test_random_seed = train_test_random_seed
        # self.markers_path = self.dataset_dir + '/' + f'markers/markers_{dataset_name}.csv'

    def normalize(self):
        if self.filter_min_counts:
            sc.pp.filter_genes(self.adata, min_counts=1)
            sc.pp.filter_cells(self.adata, min_counts=1)
        nonzero_genes, _ = sc.pp.filter_genes(self.adata.X, min_counts=1)
        assert (
            nonzero_genes.all()
        ), "Please remove all-zero genes before using DCA."

        if self.size_factor == "raw":  # Computing sf on raw data
            self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )

        if self.use_hvg:
            self.adata = get_hvg_common(
                self.adata, n_hvg=self.use_hvg, batch_key=self.batch_key
            )

        if (
            self.normalize_size_factors
            or self.scale_input
            or self.logtrans_input
        ):
            self.adata.raw = self.adata.copy()
        else:
            self.adata.raw = self.adata

        if self.normalize_size_factors:

            sc.pp.normalize_total(self.adata)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )

        if self.logtrans_input:
            sc.pp.log1p(self.adata)

        if self.scale_input:
            sc.pp.scale(self.adata)

        if self.size_factor == "default":  # Computing sf on preprocessed data
            self.adata.obs["n_counts"] = self.adata.X.sum(axis=1)
            self.adata.obs["size_factors"] = (
                self.adata.obs.n_counts / np.median(self.adata.obs.n_counts)
            )
        elif self.size_factor == "constant":
            self.adata.obs["size_factors"] = 1.0

    def train_val_split(self):
        self.adata_train_extended = self.adata[
            self.adata.obs[self.class_key] != self.unlabeled_category
        ]  # Contains the train and to-be-defined val data

        # Splits the train in val and train in a stratified way, 80/20
        pct_split = 0.2
        # Dirty workaround for celltypes with 1 cell only, which is a rare case
        stratification = self.adata_train_extended.obs[self.class_key].copy()
        if min(stratification.value_counts()) == 1:
            solo_ct = list(
                stratification.value_counts()[
                    stratification.value_counts() == 1
                ].index
            )
            max_ct = list(
                stratification.value_counts()[
                    stratification.value_counts()
                    == stratification.value_counts().max()
                ].index
            )[0]
            mapping = {}
            for i in solo_ct:
                w = np.where(stratification == i)[0]
                stratification.iloc[w] = max_ct
                mapping[i] = w
        train_idx, val_idx = train_test_split(
            np.arange(self.adata_train_extended.n_obs),
            train_size=pct_split,
            stratify=stratification,
            random_state=self.train_test_random_seed,
        )  # split on the index
        if min(stratification.value_counts()) == 1:
            for ct, w in mapping.items():
                # Adding the single celltype to the training dataset.
                if not w in train_idx:
                    train_idx = np.append(train_idx, w)
                    val_idx = val_idx[val_idx != w]

        spl = pd.Series(
            ["train"] * self.adata_train_extended.n_obs,
            index=self.adata_train_extended.obs.index,
        )
        spl.iloc[val_idx] = "val"
        self.adata_train_extended.obs["train_split"] = spl.values
        print(self.unlabeled_category)
        test_idx = (
            self.adata.obs[self.class_key] == self.unlabeled_category
        )  # boolean
        print(self.adata.obs[self.class_key].unique())
        print(test_idx.sum())
        split = pd.Series(
            ["train"] * self.adata.n_obs, index=self.adata.obs.index
        )
        split[test_idx] = "test"

        split[self.adata_train_extended.obs_names] = (
            self.adata_train_extended.obs["train_split"]
        )
        self.adata.obs["train_split"] = split

    def create_inputs(self):
        """
        Must be called after train_split
        """
        self.adata_train = self.adata[
            self.adata.obs["train_split"] == "train"
        ].copy()
        self.adata_val = self.adata[
            self.adata.obs["train_split"] == "val"
        ].copy()
        self.adata_test = self.adata[
            self.adata.obs["train_split"] == "test"
        ].copy()
        self.X = densify(self.adata.X)
        self.X_train = densify(self.adata_train.X)
        self.X_val = densify(self.adata_val.X)
        self.X_test = densify(self.adata_test.X)
        self.y = self.adata.obs[f"true_{self.class_key}"]
        self.y_train = self.adata_train.obs[f"true_{self.class_key}"]
        self.y_val = self.adata_val.obs[f"true_{self.class_key}"]
        self.y_test = self.adata_test.obs[f"true_{self.class_key}"]
        self.ohe_celltype = OneHotEncoder(
            handle_unknown="ignore"
        )  # TODO : handle the case where there can be unknown celltypes in val, pex adding an output node for 'UNK'
        y = np.array(self.y_train).reshape(-1, 1)
        self.ohe_celltype.fit(y)
        self.y_one_hot = (
            self.ohe_celltype.transform(np.array(self.y).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_train_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_train).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_val_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_val).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.y_test_one_hot = (
            self.ohe_celltype.transform(np.array(self.y_test).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch = self.adata.obs[self.batch_key]
        self.batch_train = self.adata_train.obs[self.batch_key]
        self.batch_val = self.adata_val.obs[self.batch_key]
        self.batch_test = self.adata_test.obs[self.batch_key]
        self.ohe_batches = OneHotEncoder()
        batch_ID = np.array(self.batch).reshape(
            -1, 1
        )  # We know batches from the whole dataset so we fit on the entire dataset
        self.ohe_batches.fit_transform(batch_ID).astype(float).todense()
        self.batch_one_hot = (
            self.ohe_batches.transform(np.array(self.batch).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch_train_one_hot = (
            self.ohe_batches.transform(
                np.array(self.batch_train).reshape(-1, 1)
            )
            .astype(float)
            .todense()
        )
        self.batch_val_one_hot = (
            self.ohe_batches.transform(np.array(self.batch_val).reshape(-1, 1))
            .astype(float)
            .todense()
        )
        self.batch_test_one_hot = (
            self.ohe_batches.transform(
                np.array(self.batch_test).reshape(-1, 1)
            )
            .astype(float)
            .todense()
        )
        self.sf = self.adata.obs["size_factors"].values  # .reshape(-1,1)
        self.sf_train = self.adata_train.obs[
            "size_factors"
        ].values  # .reshape(-1,1)
        self.sf_val = self.adata_val.obs[
            "size_factors"
        ].values  # .reshape(-1,1)
        self.sf_test = self.adata_test.obs[
            "size_factors"
        ].values  # .reshape(-1,1)


def process_dataset(workflow):
    # Loading dataset
    adata = load_dataset(
        dataset_dir=workflow.data_dir,
        dataset_name=workflow.run_file.dataset_name,
    )

    workflow.dataset = Dataset(
        adata=adata,
        class_key=workflow.run_file.class_key,
        batch_key=workflow.run_file.batch_key,
        filter_min_counts=workflow.run_file.filter_min_counts,
        normalize_size_factors=workflow.run_file.normalize_size_factors,
        size_factor=workflow.run_file.size_factor,
        scale_input=workflow.run_file.scale_input,
        logtrans_input=workflow.run_file.logtrans_input,
        use_hvg=workflow.run_file.use_hvg,
        unlabeled_category=workflow.unlabeled_category,
    )

    # Processing dataset. Splitting train/test.
    workflow.dataset.normalize()


def split_train_test(workflow):
    workflow.dataset.test_split(
        test_obs=workflow.run_file.test_obs,
        test_index_name=workflow.run_file.test_index_name,
    )


def split_train_val(workflow):
    workflow.dataset.train_split(
        mode=workflow.run_file.mode,
        pct_split=workflow.run_file.pct_split,
        obs_key=workflow.run_file.obs_key,
        n_keep=workflow.run_file.n_keep,
        keep_obs=workflow.run_file.keep_obs,
        split_strategy=workflow.run_file.split_strategy,
        obs_subsample=workflow.run_file.obs_subsample,
        train_test_random_seed=workflow.run_file.train_test_random_seed,
    )

    print("dataset has been preprocessed")
    workflow.dataset.create_inputs()
