import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .notebook_tools import aestetic_data_name, test_fold_selection
from .generate_figure_task1_New import AESTHETIC_DATA_NAME as NEW_AESTHETIC_DATA_NAME
from .generate_figure_task1_New import NEW_DATASETS

logger = logging.getLogger(__name__)

# Per-task configuration: checkpoint suffix, ordered (dataset -> display) pairs,
# optional fixed test fold per dataset, and output subfolder.
#   - Task 1 (old datasets): one selected test fold per dataset.
#   - Task 1 New (new datasets): median over every fold (no fixed selection).
HEATMAP_TASKS = {
    "1": {
        "suffix": "_task_1",
        "datasets": list(aestetic_data_name.items()),
        "fold_selection": test_fold_selection,
        "out_subdir": "task_1",
    },
    "1_New": {
        "suffix": "_task_1_New",
        "datasets": [(ds, NEW_AESTHETIC_DATA_NAME.get(ds, ds)) for ds in NEW_DATASETS],
        "fold_selection": None,
        "out_subdir": "task_1_New",
    },
    "2_New": {
        "suffix": "_task_2_New",
        "datasets": [(ds, NEW_AESTHETIC_DATA_NAME.get(ds, ds)) for ds in NEW_DATASETS],
        # median over every training fraction / fold / seed
        "fold_selection": None,
        "out_subdir": "task_2_New",
    },
}

# Model display names (PCA-kNN excluded), in heatmap row order.
HEATMAP_MODEL_NAMES = {
    "scPermut":         "1.scMusketeers",
    "scPermut_default": "1.scMusketeers",
    "scanvi":           "2.scANVI",
    "uce":              "3.UCE",
    "harmony_svm":      "4.Harmony",
    "pca_svm":          "5.PCA",
    "celltypist":       "6.Celltypist",
    "scmap_cells":      "7.scmap - cells",
    "scmap_cluster":    "8.scmap - cluster",
}
HEATMAP_MODEL_ORDER = list(dict.fromkeys(HEATMAP_MODEL_NAMES.values()))


def produce_heatmap(checkpoint_base_path, working_dir, task="1", metric="test_balanced_acc"):
    """Heatmap of per-dataset model rank (by median balanced accuracy).

    Models are ranked 1..N within each dataset by their median `metric`
    (1 = best). A "Global Rank" column ranks models by their mean per-dataset
    rank. PCA-kNN is excluded.

    `task` selects the benchmark: "1" (old datasets, one fixed test fold each)
    or "1_New" (new datasets, median over every fold).
    """
    cfg = HEATMAP_TASKS.get(task)
    if cfg is None:
        logger.error(f"Unknown heatmap task '{task}'. Choose from {list(HEATMAP_TASKS)}.")
        return

    checkpoint_path = checkpoint_base_path.replace(".csv", f"{cfg['suffix']}.csv")
    if not os.path.exists(checkpoint_path):
        logger.error(
            f"Checkpoint file not found at {checkpoint_path}. Run csv_process first!"
        )
        return

    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    df = pd.read_csv(checkpoint_path)
    if df.empty:
        logger.error(f"No valid task {task} runs found in checkpoint. Cannot plot heatmap.")
        return

    # Keep only the displayed models (drops pca_knn and anything unexpected).
    df = df[df["model"].isin(HEATMAP_MODEL_NAMES.keys())].copy()

    fold_selection = cfg["fold_selection"]

    # Build a model x dataset table of median balanced accuracy.
    records = {}
    dataset_cols = []
    for ds, disp in cfg["datasets"]:
        sub = df[df["dataset_name"] == ds]
        # Task 1 restricts to one selected test fold; Task 1 New uses all folds.
        if fold_selection is not None:
            if ds not in fold_selection:
                continue
            sub = sub[sub["test_fold_nb"] == fold_selection[ds]]
        if sub.empty:
            logger.warning(f"No data for dataset '{ds}', skipping column.")
            continue
        med = sub.groupby("model")[metric].median()
        records[disp] = {
            HEATMAP_MODEL_NAMES[m]: med.get(m, np.nan) for m in HEATMAP_MODEL_NAMES
        }
        dataset_cols.append(disp)

    if not dataset_cols:
        logger.error("No datasets matched; nothing to plot.")
        return

    median_df = pd.DataFrame(records).reindex(HEATMAP_MODEL_ORDER)[dataset_cols]

    n_nan = int(median_df.isna().values.sum())
    if n_nan:
        logger.warning(
            f"{n_nan} missing model x dataset cell(s) in the checkpoint; they are "
            "left blank and ranked among the available models for that dataset."
        )

    # Rank within each dataset (1 = highest median balanced accuracy). NaN cells
    # (missing runs) stay NaN and are drawn blank.
    rank_df = median_df.rank(axis=0, ascending=False, method="min")

    # Global rank = rank of the mean per-dataset rank (1 = best overall).
    global_rank = rank_df.mean(axis=1).rank(ascending=True, method="min")

    n_models = len(HEATMAP_MODEL_ORDER)
    n_ds = len(dataset_cols)

    cmap = plt.get_cmap("Blues_r").copy()
    cmap.set_bad("#e8e8e8")  # missing cells in light grey

    fig = plt.figure(figsize=(max(12, n_ds * 1.25) + 2, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[n_ds, 1.4], wspace=0.10)
    ax = fig.add_subplot(gs[0])
    ax_g = fig.add_subplot(gs[1])

    heat_kw = dict(
        cmap=cmap, vmin=1, vmax=n_models, cbar=False,
        linewidths=1.2, linecolor="white", annot_kws={"fontsize": 32},
    )

    sns.heatmap(rank_df, annot=True, fmt=".0f", ax=ax, **heat_kw)
    ax.set_xticklabels(dataset_cols, rotation=90, fontsize=15)
    ax.set_yticklabels(HEATMAP_MODEL_ORDER, rotation=0, fontsize=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=True, bottom=False)

    sns.heatmap(global_rank.to_frame("Global Rank"), annot=True, fmt=".0f", ax=ax_g, **heat_kw)
    ax_g.set_xticklabels(["Global Rank"], rotation=90, fontsize=15)
    ax_g.set_yticks([])
    ax_g.set_ylabel("")
    ax_g.set_xlabel("")

    fig.suptitle("Model rank with balanced accuracy", fontsize=20, fontweight="bold")

    out_dir = os.path.join(working_dir, "analysis_notebooks", "figure_review", cfg["out_subdir"])
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, f"Figure_task{task}_Heatmap_rank.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(out_png.replace(".png", ".svg"), bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {out_png}")

    # Save the underlying rank table for reference (nullable ints tolerate NaN).
    table = rank_df.astype("Int64").copy()
    table["Global Rank"] = global_rank.astype("Int64")
    table.to_csv(os.path.join(out_dir, "heatmap_rank.csv"))
