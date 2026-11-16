import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import logging

logger = logging.getLogger(__name__)

NEW_DATASETS = [
    "TS-Neural",
    "TS-Skin",
    "SmallIntestine-20k",
    "TS-Liver",
    "TS-BoneMarrow",
    "PBMC-Lee",
    "TS-Blood",
    "Ageing-Mouse-All",
    "SmallIntestine-All",
    "CellCards-Lung",
]

CATEGORY_TITLES = {
    "small":  "Small (≤ 20k cells)",
    "medium": "Medium (20k – 100k cells)",
    "large":  "Large (> 100k cells)",
}

# Datasets ordered by cell count within each size group
DIFF_DICT = {
    "small":  ["TS-Neural", "TS-Skin", "SmallIntestine-20k"],
    "medium": ["TS-Liver", "TS-BoneMarrow", "PBMC-Lee", "TS-Blood"],
    "large":  ["Ageing-Mouse-All", "SmallIntestine-All", "CellCards-Lung"],
}

AESTHETIC_DATA_NAME = {
    "Ageing-Mouse-All":    "Ageing Mouse All",
    "CellCards-Lung":      "CellCards Lung",
    "PBMC-Lee":            "PBMC Lee",
    "SmallIntestine-20k":  "Small Intestine 20k",
    "SmallIntestine-All":  "Small Intestine All",
    "TS-Blood":            "TS - Blood",
    "TS-BoneMarrow":       "TS - Bone Marrow",
    "TS-Liver":            "TS - Liver",
    "TS-Neural":           "TS - Neural",
    "TS-Skin":             "TS - Skin",
}

MODEL_NAMES = {
    "scPermut":         "1-scMusketeers",
    "scPermut_default": "1-scMusketeers",
    "scanvi":           "2-scANVI",
    "uce":              "3-UCE",
    "harmony_svm":      "4-Harmony",
    "pca_svm":          "5-PCA",
    "celltypist":       "6-CellTypist",
    "scmap_cells":      "7-scmap-cells",
    "scmap_cluster":    "8-scmap-cluster",
}

MODEL_COLORS = {
    "scPermut_default": "#B15240",
    "scanvi":           "#C78C3B",
    "uce":              "#D3A53C",
    "harmony_svm":      "#5B9DC7",
    "pca_svm":          "#264D74",
    "celltypist":       "#75BAD3",
    "scmap_cells":      "#607F6A",
    "scmap_cluster":    "#707C45",
}

MODEL_ORDER = list(dict.fromkeys(MODEL_NAMES.values()))

ENTROPY_EXCLUDED_MODELS = {"6-CellTypist", "7-scmap-cells", "8-scmap-cluster"}

MODEL_COLORS_AESTHETIC = {
    aesthetic: MODEL_COLORS[orig]
    for orig, aesthetic in MODEL_NAMES.items()
    if orig in MODEL_COLORS
}


def produce_fig_task1_New(checkpoint_base_path, working_dir, run_stats=False):
    task1_new_checkpoint_path = checkpoint_base_path.replace(".csv", "_task_1_New.csv")
    if not os.path.exists(task1_new_checkpoint_path):
        logger.error(f"Checkpoint file not found at {task1_new_checkpoint_path}. Run csv_process first!")
        return

    logger.info(f"Loading checkpoint from {task1_new_checkpoint_path}...")
    runs_table_df = pd.read_csv(task1_new_checkpoint_path)

    if runs_table_df.empty:
        logger.error("No valid task 1 New runs found in checkpoint. Cannot plot figures.")
        return

    logger.info("Building DataFrames for Task 1 New plots...")
    task_dict_acc = {}

    for key, ds_list in DIFF_DICT.items():
        df_acc = pd.DataFrame()
        for ds in ds_list:
            ds_df = runs_table_df[runs_table_df["dataset_name"] == ds].copy()
            if ds_df.empty:
                logger.warning(f"No data found for dataset '{ds}', skipping.")
                continue
            for model in ds_df["model"].unique():
                if model == "pca_knn":  # PCA-kNN excluded from all figures
                    continue
                sub = ds_df[ds_df["model"] == model].copy()
                if not sub.empty:
                    sub["aestetic_data_name"] = AESTHETIC_DATA_NAME.get(ds, ds)
                    sub["aestetic_model_name"] = MODEL_NAMES.get(model, model)
                    df_acc = pd.concat([df_acc, sub])
        task_dict_acc[key] = df_acc

    figures_dir = os.path.join(working_dir, "analysis_notebooks", "figure_review", "task_1_New")
    os.makedirs(figures_dir, exist_ok=True)

    combined_df = pd.concat(
        [df.assign(category=key) for key, df in task_dict_acc.items()],
        ignore_index=True,
    )
    combined_path = os.path.join(figures_dir, "task_dict_acc.csv")
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Saved task_dict_acc to {combined_path}")

    logger.info("Generating Task 1 New sub-figures...")

    def generate_subfigures(metric_col, ylabel, output_prefix, exclude_models=None):
        for key, title_name in CATEGORY_TITLES.items():
            df_plot = task_dict_acc.get(key, pd.DataFrame())
            if df_plot.empty:
                logger.warning(f"No data for category '{key}', skipping.")
                continue

            order = [m for m in MODEL_ORDER if m in df_plot["aestetic_model_name"].unique()]
            if exclude_models:
                order = [m for m in order if m not in exclude_models]
            n_datasets = df_plot["aestetic_data_name"].nunique()
            fig_width = max(8, n_datasets * 4)

            _, ax = plt.subplots(figsize=(fig_width, 6))

            sns.boxplot(
                data=df_plot,
                x="aestetic_data_name",
                y=metric_col,
                hue="aestetic_model_name",
                showfliers=False,
                hue_order=order,
                palette=MODEL_COLORS_AESTHETIC,
                ax=ax,
            )
            sns.stripplot(
                data=df_plot,
                x="aestetic_data_name",
                y=metric_col,
                hue="aestetic_model_name",
                size=3,
                color="black",
                dodge=True,
                ax=ax,
                hue_order=order,
                legend=False,
            )

            if run_stats:
                baseline_name = "1-scMusketeers"
                pairs = []
                if baseline_name in order:
                    for dataset in df_plot["aestetic_data_name"].unique():
                        for m in order:
                            if m != baseline_name:
                                has_baseline = not df_plot[
                                    (df_plot["aestetic_data_name"] == dataset)
                                    & (df_plot["aestetic_model_name"] == baseline_name)
                                ].empty
                                has_model = not df_plot[
                                    (df_plot["aestetic_data_name"] == dataset)
                                    & (df_plot["aestetic_model_name"] == m)
                                ].empty
                                if has_baseline and has_model:
                                    pairs.append(((dataset, baseline_name), (dataset, m)))
                if pairs:
                    annot = Annotator(
                        ax,
                        pairs,
                        data=df_plot,
                        x="aestetic_data_name",
                        y=metric_col,
                        hue="aestetic_model_name",
                        hue_order=order,
                    )
                    try:
                        annot.configure(
                            test="Wilcoxon",
                            text_format="star",
                            loc="inside",
                            comparisons_correction="Benjamini-Hochberg",
                        )
                        annot.apply_test().annotate()
                    except Exception as e:
                        logger.debug(f"Stats annotation error on '{title_name}' for {metric_col}: {e}")

            ax.set_title(title_name, fontsize=14)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Dataset")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

            plt.tight_layout()
            output_path = os.path.join(figures_dir, f"{output_prefix}_{key}.png")
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            plt.savefig(output_path.replace(".png", ".svg"), bbox_inches="tight")
            logger.info(f"Saved {output_path}")
            plt.close()

    generate_subfigures(
        metric_col="test_acc",
        ylabel="Accuracy",
        output_prefix="Figure_task1_New_Accuracy",
    )

    generate_subfigures(
        metric_col="test_balanced_acc",
        ylabel="Balanced accuracy",
        output_prefix="Figure_task1_New_Balanced_Accuracy",
    )

    generate_subfigures(
        metric_col="full_batch_mixing_entropy",
        ylabel="Batch mixing entropy",
        output_prefix="Figure_task1_New_Entropy",
        exclude_models=ENTROPY_EXCLUDED_MODELS,
    )
