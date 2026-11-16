#!/usr/bin/python
import anndata as ad
import pandas as pd
import scanpy as sc
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import logging

try:
    import celltypist
    import scanpy.external as sce
    import scBalance as sb
    import scBalance.scbalance_IO as ss
    import scvi
    from scmappy import common_genes, scmap_annotate
except:
    pass
import numpy as np

try:
    from ..tools.utils import densify
except ImportError:
    from scmusketeers.tools.utils import densify

# from tools.utils import densify

logger = logging.getLogger("Sc-Musketeers")

# logger.debug("Last run with scvi-tools version:", scvi.__version__)


def svm_label(X_full, y_list, assign, pred_full=True):
    X_train = X_full[assign == "train", :]
    X_test_val = X_full[assign != "train", :]

    y_train = y_list["full"][assign == "train"]
    #    y_train = y_list['train']
    # y_test = y_list['full'][assign != 'train']

    clf = svm.SVC()  # default rbf ok ? or Linear Kernel ?
    clf.fit(X_train, y_train)

    if pred_full:
        y_pred = clf.predict(X_full)
    else:
        y_pred = clf.predict(X_test_val)

    return y_pred


def knn_label(X_full, y_list, assign, pred_full=True):
    X_train = X_full[assign == "train", :]
    X_test_val = X_full[assign != "train", :]

    y_train = y_list["full"][assign == "train"]
    #    y_train = y_list['train']
    # y_test = y_list['full'][assign != 'train']

    clf = KNeighborsClassifier(
        n_neighbors=5
    )  # default rbf ok ? or Linear Kernel ?
    clf.fit(X_train, y_train)

    if pred_full:
        y_pred = clf.predict(X_full)
    else:
        y_pred = clf.predict(X_test_val)

    return y_pred


def pca_knn(X_list, y_list, batch_list, assign, adata_list, pred_full=True):
    """Perform PCA reduction and then predict cell type's
    annotation with a SVM algorithm
    return :
        - latent = X_pca
        - y_pred = prediction for all cells"""
    # adata = sc.AnnData(X = X_list['full'],
    #                    obs = pd.DataFrame({
    #                        'celltype': y_list['full'],
    #                        'batch': batch_list['full'],
    #                        'split': assign},index =  y_list['full'].index))
    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]
    y_pred = knn_label(X_pca, y_list, assign, pred_full=pred_full)

    X_pca_list = {
        group: X_pca[assign == group, :] for group in np.unique(assign)
    }
    X_pca_list["full"] = X_pca
    y_pred_list = {
        group: y_pred[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred
    return X_pca_list, y_pred_list


def pca_svm(X_list, y_list, batch_list, assign, adata_list, pred_full=True):
    """Perform PCA reduction and then predict cell type's
    annotation with a SVM algorithm
    return :
        - latent = X_pca
        - y_pred = prediction for all cells"""
    # adata = sc.AnnData(X = X_list['full'],
    #                    obs = pd.DataFrame({
    #                        'celltype': y_list['full'],
    #                        'batch': batch_list['full'],
    #                        'split': assign},index =  y_list['full'].index))
    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]
    y_pred = svm_label(X_pca, y_list, assign, pred_full=pred_full)

    X_pca_list = {
        group: X_pca[assign == group, :] for group in np.unique(assign)
    }
    X_pca_list["full"] = X_pca
    y_pred_list = {
        group: y_pred[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred
    return X_pca_list, y_pred_list


def harmony_svm(
    X_list, y_list, batch_list, assign, adata_list, pred_full=True
):  # -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Perform an integration from different dataset
    and then predict cell type's
    annotation with a SVM algorithm
    return :
        - latent = X_pca_harmony
        - y_pred = prediction for all cells"""
    # adata = sc.AnnData(X = X_list['full'],
    #                    obs = pd.DataFrame({
    #                        'celltype': y_list['full'],
    #                        'batch': batch_list['full'],
    #                        'split': assign},index =  y_list['full'].index))

    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    adata.obs["batch"] = batch_list["full"]
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    if not "X_pca_harmony" in adata.obsm:
        logger.debug("Did not find existing harmony, computing it")
        import harmonypy as hm
        ho = hm.run_harmony(adata.obsm["X_pca"], adata.obs, "batch")
        # In harmonypy>=0.2.0 the Z_corr property already returns (n_cells, n_pcs)
        adata.obsm["X_pca_harmony"] = ho.Z_corr
    X_pca_harmony = adata.obsm["X_pca_harmony"].copy()
    y_pred = svm_label(X_pca_harmony, y_list, assign, pred_full=pred_full)

    X_pca_harmony_list = {
        group: X_pca_harmony[assign == group, :] for group in np.unique(assign)
    }
    X_pca_harmony_list["full"] = X_pca_harmony
    y_pred_list = {
        group: y_pred[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred
    return X_pca_harmony_list, y_pred_list


def celltypist_model(
    X_list, y_list, batch_list, assign, adata_list, n_jobs=30, pred_full=True
):  # -> tuple:
    """Perform label transfer using CellTypist :
    - latent = X_pca
    - y_pred = prediction for all cells"""
    adata = sc.AnnData(
        X=X_list["full"].copy(),
        obs=pd.DataFrame(
            {
                "celltype": y_list["full"].copy(),
                "batch": batch_list["full"].copy(),
                "model": assign.copy(),
            }
        ),
    )
    # sc.pp.normalize_total(adata, target_sum = 1e4)
    # sc.pp.log1p(train_val)
    adata_train = adata[adata.obs["model"].isin(["train"])]

    sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]

    logger.debug("Start train model")
    # The default LBFGS logistic-regression solver does not scale: on large
    # training sets (e.g. Ageing-Mouse-All at pct 0.5/0.9, ~55k-89k cells) it
    # runs for >24h and hits the SLURM time limit. CellTypist recommends the
    # SGD/mini-batch solver above ~20k cells, so switch to it well before the
    # training set reaches 100k cells.
    # Only request the GPU SGD path when a GPU is actually present; otherwise
    # use_GPU=True crashes on CPU-only nodes (jobs are moved to CPU when the GPU
    # queue is saturated). SGD on CPU is still fine, just a little slower.
    try:
        import torch
        _use_gpu = torch.cuda.is_available()
    except Exception:
        _use_gpu = False
    if adata_train.n_obs > 20000:
        model = celltypist.train(
            adata_train,
            "celltype",
            n_jobs=n_jobs,
            use_SGD=True,
            use_GPU=_use_gpu,
            mini_batch=True,
            check_expression=False,
        )
    else:
        model = celltypist.train(
            adata_train, "celltype", n_jobs=n_jobs, check_expression=False
        )
    # .X = expect log1p normalized expression to 10000 counts per cell
    # if not -> check_expression = False

    logger.debug("Start annotate dataset")
    predictions = celltypist.annotate(adata, model=model)
    # majority_voting = False default

    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]

    X_pca_list = {
        group: X_pca[assign == group, :] for group in np.unique(assign)
    }
    X_pca_list["full"] = X_pca
    y_pred_list = {
        group: predictions.predicted_labels["predicted_labels"][
            assign == group
        ]
        for group in np.unique(assign)
    }
    y_pred_list["full"] = predictions.predicted_labels["predicted_labels"]
    return X_pca_list, y_pred_list


def scmap_cluster(
    X_list, y_list, batch_list, assign, adata_list, pred_full=True
):
    adata = sc.AnnData(
        X=densify(X_list["full"]),
        obs=pd.DataFrame(
            {"celltype": y_list["full"]}, index=y_list["full"].index
        ),
    )

    adata.var["Gene_names"] = adata.var_names
    adata_train = adata[assign == "train", :]
    adata_test_val = adata[assign.isin(["test", "val"]), :]

    adata_train, adata_test_val = common_genes(
        adata_train, adata_test_val, "Gene_names", remove_unmached=True
    )

    sc.pp.highly_variable_genes(adata_train)

    y_pred = scmap_annotate(
        adata_test_val,
        adata_train,  # train is the ref, test_val is the query
        "Gene_names",
        "celltype",
        inplace=False,
        algorithm_flavor="centroid",
        gene_selection_flavor="HVGs",
        similarity_threshold=0.7,
        key_added="scmap_annotation",
    )
    y_pred = pd.Series(y_pred, index=adata_test_val.obs_names)
    y_pred = pd.concat([y_pred, y_list["train"]])
    y_pred = y_pred[
        y_list["full"].index
    ]  # reordering them according to y_full just to be sure

    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]

    X_pca_list = {
        group: X_pca[assign == group, :] for group in np.unique(assign)
    }
    X_pca_list["full"] = X_pca
    y_pred_list = {
        group: y_pred[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred
    return X_pca_list, y_pred_list


def scmap_cells(
    X_list, y_list, batch_list, assign, adata_list, pred_full=True
):
    adata = sc.AnnData(
        X=densify(X_list["full"]),
        obs=pd.DataFrame(
            {"celltype": y_list["full"]}, index=y_list["full"].index
        ),
    )

    adata.var["Gene_names"] = adata.var_names
    adata_train = adata[assign == "train", :]
    adata_test_val = adata[assign.isin(["test", "val"]), :]

    adata_train, adata_test_val = common_genes(
        adata_train, adata_test_val, "Gene_names", remove_unmached=True
    )

    sc.pp.highly_variable_genes(adata_train)

    y_pred = scmap_annotate(
        adata_test_val,
        adata_train,  # train is the ref, test_val is the query
        "Gene_names",
        "celltype",
        inplace=False,
        algorithm_flavor="cell",
        gene_selection_flavor="HVGs",
        similarity_threshold=0.7,
        key_added="scmap_annotation",
    )
    y_pred = pd.Series(y_pred, index=adata_test_val.obs_names)
    y_pred = pd.concat([y_pred, y_list["train"]])
    y_pred = y_pred[
        y_list["full"].index
    ]  # reordering them according to y_full just to be sure

    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]

    X_pca_list = {
        group: X_pca[assign == group, :] for group in np.unique(assign)
    }
    X_pca_list["full"] = X_pca
    y_pred_list = {
        group: y_pred[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred
    return X_pca_list, y_pred_list


def uce(X_list, y_list, batch_list, assign, adata_list, pred_full=True):
    """
    Since UCE embedding is fully unsupervised and deterministic, it is computed beforehand. Hence, this function implies that a 'X_uce' field with UCE embedding already exists in adata.
    """
    X_uce = adata_list["full"].obsm["X_uce"].copy()
    y_pred = svm_label(X_uce, y_list, assign, pred_full=pred_full)

    X_uce_list = {
        group: X_uce[assign == group, :] for group in np.unique(assign)
    }
    X_uce_list["full"] = X_uce
    y_pred_list = {
        group: y_pred[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred
    return X_uce_list, y_pred_list


def scanvi(X_list, y_list, batch_list, assign, adata_list):
    """Perform scvi and scanvi integration and
    do the predict on the unknow cells with scanvi
    input :
        - anndata : .X = raw count ++ (not scale)
    return :
        - latent = latent_space from scanvi
        - y_pred = prediction for all cells"""
    unlabeled_category = "UNK"
    SCANVI_LATENT_KEY = "X_scANVI"
    SCANVI_PREDICTION_KEY = "pred_scANVI"  # "C_scANVI"

    adata = sc.AnnData(
        X=X_list["full"].copy(),
        obs=pd.DataFrame(
            {
                "celltype": y_list["full"].copy(),
                "celltype_full": y_list["full"].copy(),
                "batch": batch_list["full"].copy(),
                "split": assign.copy(),
            },
            index=assign.index,
        ),
    )
    adata.obs["celltype"] = adata.obs["celltype"].astype(str)
    adata.obs["celltype"][
        adata.obs["split"].isin(["val", "test"])
    ] = unlabeled_category
    # sorting_order = {'train': 1, 'val': 2, 'test': 3}
    # sorted_df = adata.obs.sort_values(by='split',
    #                                   key=lambda x:x.map(sorting_order))
    # adata = adata[sorted_df.index]
    # adata.obs = adata.obs.reset_index(drop=True)
    # adata.layers['count'] = adata.X

    # Run scvi
    scvi.model.SCVI.setup_anndata(
        adata,
        # layer = "count",
        batch_key="batch",
        labels_key="celltype",
    )
    scvi_model = scvi.model.SCVI(
        adata, n_layers=1, n_latent=50  # default = 10  or 50 ?
    )  # default = 1
    logger.debug("start train scvi")
    scvi_model.train(
        max_epochs=100,
        early_stopping=True,
    )

    # Run scanvi
    scanvi_model = scvi.model.SCANVI.from_scvi_model(
        scvi_model,
        adata=adata,
        unlabeled_category=unlabeled_category,
        labels_key="celltype",
    )
    logger.debug("start train scanvi")
    scanvi_model.train(
        max_epochs=20,
        n_samples_per_label=100,
        # train_size=1,
        # validation_size=None,  # ,
        # shuffle_set_split = False
    )

    adata.obsm[SCANVI_LATENT_KEY] = scanvi_model.get_latent_representation(
        adata
    )
    adata.obs[SCANVI_PREDICTION_KEY] = scanvi_model.predict(adata)

    latent_list = {
        group: np.asarray(
            adata[adata.obs["split"] == group, :].obsm[SCANVI_LATENT_KEY]
        )
        for group in np.unique(assign)
    }
    latent_list["full"] = adata.obsm[SCANVI_LATENT_KEY]
    y_pred_list = {
        group: np.asarray(
            adata[adata.obs["split"] == group, :].obs[SCANVI_PREDICTION_KEY]
        )
        for group in np.unique(assign)
    }
    y_pred_list["full"] = adata.obs[SCANVI_PREDICTION_KEY]
    return latent_list, y_pred_list


def scBalance_model(X_list, y_list, batch_list, assign, adata_list):
    full, reference, ref_label = ss.Scanpy_Obj_IO(
        test_obj=adata_list["full"],
        ref_obj=adata_list["train"],
        label_obj=y_list["train"],
        scale=False,
    )
    y_pred_full = sb.scBalance(full, reference, ref_label, "cpu")
    logger.debug(len(y_pred_full))
    adata = adata_list[
        "full"
    ]  # adding PCA to adata_list['full'] the first time and reuses it for the next function calls
    if not "X_pca" in adata.obsm:
        logger.debug("Did not find existing PCA, computing it")
        sc.tl.pca(adata)
    X_pca = adata.obsm["X_pca"]

    X_pca_list = {
        group: X_pca[assign == group, :] for group in np.unique(assign)
    }
    X_pca_list["full"] = X_pca
    y_pred_list = {
        group: y_pred_full[assign == group] for group in np.unique(assign)
    }
    y_pred_list["full"] = y_pred_full
    return X_pca_list, y_pred_list
