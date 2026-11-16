import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import logging

from .notebook_tools import diff_dict, aestetic_data_name, test_fold_selection

logger = logging.getLogger(__name__)

CATEGORY_TITLES = {
    'homo': 'Homo',
    'assays': 'Assays',
    'suspension': 'Suspension',
    'datasets': 'Datasets',
}

FIGURE2_MODEL_NAMES = {
    'scPermut':         '1-scMusketeers',
    'scPermut_default': '1-scMusketeers',
    'scanvi':           '2-scANVI',
    'uce':            '3-UCE',
    'harmony_svm':    '4-Harmony',
    'pca_svm':        '5-PCA',
    'celltypist':     '6-CellTypist',
    'scmap_cells':    '7-scmap-cells',
    'scmap_cluster':  '8-scmap-cluster',
}

MODEL_COLORS = {
          'scPermut_default' : "#B15240",
          'scanvi':"#C78C3B" ,
          'uce':"#D3A53C" ,
          'harmony_svm': "#5B9DC7",
          'pca_svm':"#264D74",
          'celltypist': "#75BAD3",
          'scmap_cells':"#607F6A" ,
          'scmap_cluster':"#707C45"}

FIGURE2_MODEL_ORDER = list(FIGURE2_MODEL_NAMES.values())

ENTROPY_EXCLUDED_MODELS = {"6-CellTypist", "7-scmap-cells", "8-scmap-cluster"}

FIGURE2_MODEL_COLORS = {
    aesthetic_name: MODEL_COLORS[orig_key]
    for orig_key, aesthetic_name in FIGURE2_MODEL_NAMES.items()
    if orig_key in MODEL_COLORS
}

def produce_fig_2(checkpoint_base_path, working_dir, run_stats=False):
    task1_checkpoint_path = checkpoint_base_path.replace('.csv', '_task_1.csv')
    if not os.path.exists(task1_checkpoint_path):
        logger.error(f"Checkpoint file not found at {task1_checkpoint_path}. Run csv_process first!")
        return

    logger.info(f"Loading checkpoint from {task1_checkpoint_path}...")
    runs_table_df = pd.read_csv(task1_checkpoint_path)

    if runs_table_df.empty:
        logger.error("No valid task 1 runs found in checkpoint. Cannot plot figures.")
        return

    logger.info("Building DataFrames for Figure 2 plots...")
    task_dict_acc = {}

    for key in diff_dict.keys():
        df_acc = pd.DataFrame()
        for ds in diff_dict[key]:
            if ds not in test_fold_selection:
                continue
            expected_test_fold = test_fold_selection[ds]
            ds_df = runs_table_df.query(f"dataset_name == '{ds}' and test_fold_nb == {expected_test_fold}")

            for model in ds_df['model'].unique():
                sub = ds_df.query(f"model == '{model}'").copy()
                if not sub.empty:
                    sub['aestetic_data_name'] = aestetic_data_name.get(ds, ds)
                    sub['aestetic_model_name'] = FIGURE2_MODEL_NAMES.get(model, model)
                    df_acc = pd.concat([df_acc, sub])
        task_dict_acc[key] = df_acc

    figures_dir = os.path.join(working_dir, 'analysis_notebooks', 'figure_review', 'task_1')
    os.makedirs(figures_dir, exist_ok=True)

    combined_df = pd.concat(
        [df.assign(category=key) for key, df in task_dict_acc.items()],
        ignore_index=True
    )
    combined_path = os.path.join(figures_dir, "task_dict_acc.csv")
    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Saved task_dict_acc to {combined_path}")

    logger.info("Generating Figure 2 sub-figures...")

    model_order = FIGURE2_MODEL_ORDER

    def generate_subfigures(metric_col, ylabel, output_prefix, exclude_models=None):
        for key, title_name in CATEGORY_TITLES.items():
            df_plot = task_dict_acc.get(key, pd.DataFrame())
            if df_plot.empty:
                logger.warning(f"No data for category '{key}', skipping.")
                continue

            order = [m for m in model_order if m in df_plot['aestetic_model_name'].unique()]
            if exclude_models:
                order = [m for m in order if m not in exclude_models]
            n_datasets = df_plot['aestetic_data_name'].nunique()
            fig_width = max(8, n_datasets * 4)

            _, ax = plt.subplots(figsize=(fig_width, 6))

            sns.boxplot(data=df_plot, x="aestetic_data_name", y=metric_col,
                        hue="aestetic_model_name", showfliers=False, hue_order=order,
                        palette=FIGURE2_MODEL_COLORS, ax=ax)
            sns.stripplot(data=df_plot, x="aestetic_data_name", y=metric_col,
                          hue="aestetic_model_name", size=3, color="black",
                          dodge=True, ax=ax, hue_order=order, legend=False)

            if run_stats:
                baseline_name = '1-scMusketeers'
                pairs = []
                if baseline_name in order:
                    for dataset in df_plot['aestetic_data_name'].unique():
                        for m in order:
                            if m != baseline_name:
                                has_baseline = not df_plot[
                                    (df_plot['aestetic_data_name'] == dataset) &
                                    (df_plot['aestetic_model_name'] == baseline_name)
                                ].empty
                                has_model = not df_plot[
                                    (df_plot['aestetic_data_name'] == dataset) &
                                    (df_plot['aestetic_model_name'] == m)
                                ].empty
                                if has_baseline and has_model:
                                    pairs.append(((dataset, baseline_name), (dataset, m)))
                if pairs:
                    annot = Annotator(ax, pairs, data=df_plot, x="aestetic_data_name",
                                      y=metric_col, hue="aestetic_model_name", hue_order=order)
                    try:
                        annot.configure(test='Wilcoxon', text_format='star', loc='inside',
                                        comparisons_correction="Benjamini-Hochberg")
                        annot.apply_test().annotate()
                    except Exception as e:
                        logger.debug(f"Stats annotation error on '{title_name}' for {metric_col}: {e}")

            ax.set_title(title_name, fontsize=14)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Dataset")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

            plt.tight_layout()
            output_path = os.path.join(figures_dir, f"{output_prefix}_{key}.png")
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
            logger.info(f"Saved {output_path}")
            plt.close()

    generate_subfigures(
        metric_col="test_balanced_acc",
        ylabel="Balanced accuracy",
        output_prefix="Figure_task1_Balanced_Accuracy",
    )

    generate_subfigures(
        metric_col="full_batch_mixing_entropy",
        ylabel="Batch mixing entropy",
        output_prefix="Figure_task1_Entropy",
        exclude_models=ENTROPY_EXCLUDED_MODELS,
    )
