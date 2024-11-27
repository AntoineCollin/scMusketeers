import argparse
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from PIL import ImageColor
from sklearn.model_selection import train_test_split


def check_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def df_to_dict(df, key_column, value_column, singleton_to_str=False):
    result_dict = df.groupby(key_column)[value_column].apply(list).to_dict()
    if singleton_to_str:
        for k, v in result_dict.items():
            if len(v) == 1:
                result_dict[k] = v[0]
    return result_dict


def dict_to_df(
    dico,
    val_name="values",
    key_name="keys",
):
    keys = [k for k, v in dico.items() for _ in v]
    values = [v for v in dico.values() for v in v]
    df = pd.DataFrame({val_name: values, key_name: keys})
    return df


def densify(X):
    if (type(X) == scipy.sparse.csr_matrix) or (
        type(X) == scipy.sparse.csc_matrix
    ):
        return np.asarray(X.todense())
    else:
        return np.asarray(X)


def check_raw(X):
    return int(np.max(X)) == np.max(X)


def ann_subset(adata, obs_key, conditions):
    """
    Return a subset of the adata for cells with obs_key verifying conditions
    """
    if type(conditions) == str:
        conditions = [conditions]
    return adata[adata.obs[obs_key].isin(conditions), :].copy()


def nan_to_0(val):
    if np.isnan(val) or pd.isna(val) or type(val) == type(None):
        return 0
    else:
        return val


def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))


def hex2rgb(hexcode):
    return np.array(ImageColor.getcolor(hexcode, "RGB"))


def save_json(dico, p):
    if not p.startswith("/"):
        p = p
    if not p.endswith(".json"):
        p += ".json"
    with open(p, "w") as fichier_json:
        json.dump(dico, fichier_json)


def load_json(p):
    if not p.startswith("/"):
        p = JSON_PATH + p
    if not p.endswith(".json"):
        p += ".json"
    with open(p, "r") as fichier_json:
        dico = json.load(fichier_json)
    return dico


def scanpy_to_input(adata, keys, use_raw=False):
    """
    Converts a scanpy object to a csv count matrix + an array for each metadata specified in *args
    """
    adata_to_dict = {}
    if use_raw:
        adata_to_dict["counts"] = densify(adata.raw.X.copy())
    else:
        adata_to_dict["counts"] = densify(adata.X.copy())
    for key in keys:
        adata_to_dict[key] = adata.obs[key]
    return adata_to_dict


def input_to_scanpy(count_matrix, obs_df, obs_names=None):
    """
    Converts count matrix and metadata to a

    count_matrix : matrix type data (viable input to sc.AnnData(X=...))
    obs_df : either a pd.DataFrame or a list of pd.Series in which case they will be concatenated
    obs_name : optional, row/index/obs name for the AnnData object
    """
    if type(obs_df) != pd.DataFrame:
        obs_df = pd.concat(obs_df)
    ad = sc.AnnData(X=count_matrix, obs=obs_df)
    if obs_names:
        ad.obs_names = obs_names
    return ad


def default_value(var, val):
    """
    Returns var when val is None
    """
    if not var:
        return val
    else:
        return var


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def tuple_to_scalar(v):
    if isinstance(v, int) or isinstance(v, float):
        return v
    elif len(v) == 1:
        return float(v[0])
    else:
        return [float(i) for i in v]


def create_meta_stratify(adata, cats):
    obs = adata.obs
    meta_cats = "_".join(cats)
    meta_col = pd.Series(obs[cats[0]].astype(str), index=obs.index)
    for h in cats[1:]:
        meta_col = meta_col + "_" + obs[h].astype(str)
    obs[meta_cats] = meta_col
    adata.obs[meta_cats] = meta_col


def split_adata(adata, cats, train_size=0.8):
    """
    Split adata in train and test in a stratified manner according to cats
    cats - List of celltype, batch for sampling
    """
    if type(cats) == str:
        meta_cats = cats
    else:
        create_meta_stratify(adata, cats)
        meta_cats = "_".join(cats)
    train_idx, test_idx = train_test_split(
        np.arange(adata.n_obs),
        train_size=train_size,
        stratify=adata.obs[meta_cats],
        random_state=50,
    )
    spl = pd.Series(["train"] * adata.n_obs, index=adata.obs.index)
    spl.iloc[test_idx] = "test"
    adata.obs["TRAIN_TEST_split"] = spl.values
    adata_train = adata[adata.obs["TRAIN_TEST_split"] == "train"].copy()
    adata_test = adata[adata.obs["TRAIN_TEST_split"] == "test"].copy()
    del adata.obs["TRAIN_TEST_split"]
    return adata_train, adata_test


def result_dir(neptune_id, working_dir=None):
    if working_dir:
        save_dir = (
            working_dir + "experiment_script/results/" + str(neptune_id) + "/"
        )
    else:
        save_dir = "./experiment_script/results/" + str(neptune_id) + "/"
    return save_dir


def load_confusion_matrix(neptune_id, train_split="val", working_dir=None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(
        save_dir + f"confusion_matrix_{train_split}.csv", index_col=0
    )


def load_pred(neptune_id, working_dir=None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(
        save_dir + f"predictions_full.csv", index_col=0
    ).squeeze()


def load_proba_pred(neptune_id, working_dir=None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(
        save_dir + f"y_pred_proba_full.csv", index_col=0
    ).squeeze()


def load_split(neptune_id, working_dir=None):
    save_dir = result_dir(neptune_id, working_dir)
    return pd.read_csv(save_dir + f"split_full.csv", index_col=0).squeeze()


def load_latent_space(neptune_id, working_dir=None):
    save_dir = result_dir(neptune_id, working_dir)
    return np.load(save_dir + f"latent_space_full.npy")


def load_umap(neptune_id, working_dir=None):
    save_dir = result_dir(neptune_id, working_dir)
    return np.load(save_dir + f"umap_full.npy")


def load_expe(neptune_id, working_dir):
    save_dir = result_dir(neptune_id, working_dir)
    X = load_latent_space(neptune_id, working_dir)
    pred = load_pred(neptune_id, working_dir)
    adata = sc.AnnData(X=X, obs=pred)
    # proba_pred = load_proba_pred(neptune_id, working_dir)
    umap = load_umap(neptune_id, working_dir)
    # adata.obsm['proba_pred'] = proba_pred
    adata.obsm["X_umap"] = umap
    return adata


def plot_umap_proba(adata, celltype, **kwargs):
    adata.obs[celltype] = adata.obsm["proba_pred"][celltype]
    sc.pl.umap(adata, color=celltype, **kwargs)


def plot_size_conf_correlation(adata):
    proba_pred = adata.obsm["proba_pred"]
    class_df_dict = {
        ct: proba_pred.loc[adata.obs["true"] == ct, :]
        for ct in adata.obs["true"].cat.categories
    }  # The order of the plot is defined here (adata.obs['true_louvain'].cat.categories)
    mean_acc_dict = {ct: df.mean(axis=0) for ct, df in class_df_dict.items()}

    f, axes = plt.subplots(1, 2, figsize=(10, 5))
    f.suptitle("correlation between confidence and class size")
    pd.Series(
        {ct: class_df_dict[ct].shape[0] for ct in mean_acc_dict.keys()}
    ).plot.bar(ax=axes[0])
    pd.Series(
        {ct: mean_acc_dict[ct][ct] for ct in mean_acc_dict.keys()}
    ).plot.bar(ax=axes[1])


def plot_class_accuracy(adata, layout=True, **kwargs):
    """
    mode is either bar (average) or box (boxplot)
    """
    adata = self.latent_spaces[ID]
    workflow = self.workflow_list[ID]
    true_key = f"true_{workflow.class_key}"
    pred_key = f"{workflow.class_key}_pred"
    labels = adata.obs[true_key].cat.categories
    conf_mat = pd.DataFrame(
        confusion_matrix(
            adata.obs[true_key], adata.obs[pred_key], labels=labels
        ),
        index=labels,
        columns=labels,
    )

    n = math.ceil(np.sqrt(len(labels)))
    f, axes = plt.subplots(n, n, constrained_layout=layout)
    f.suptitle("Accuracy & confusion by celltype")
    #     plt.constrained_layout()
    i = 0
    for ct in labels:
        r = i // n
        c = i % n
        ax = axes[r, c]
        df = conf_mat.loc[ct, :] / conf_mat.loc[ct, :].sum()
        df.plot.bar(ax=ax, figsize=(20, 15), ylim=(0, 1), **kwargs)
        ax.tick_params(axis="x", labelrotation=90)
        ax.set_title(ct + f"- {conf_mat.loc[ct,:].sum()} cells")
        i += 1
