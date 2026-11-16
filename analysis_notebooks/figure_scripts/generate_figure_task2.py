import os
import pandas as pd
import numpy as np
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
    'uce':              '3-UCE',
    'harmony_svm':      '4-Harmony',
    'pca_svm':          '5-PCA',
    'celltypist':       '6-CellTypist',
    'scmap_cells':      '7-scmap-cells',
    'scmap_cluster':    '8-scmap-cluster',
}

MODEL_COLORS = {
    'scPermut_default': "#B15240",
    'scanvi':           "#C78C3B",
    'uce':              "#D3A53C",
    'harmony_svm':      "#5B9DC7",
    'pca_svm':          "#264D74",
    'celltypist':       "#75BAD3",
    'scmap_cells':      "#607F6A",
    'scmap_cluster':    "#707C45",
}

FIGURE2_MODEL_COLORS = {
    aesthetic_name: MODEL_COLORS[orig_key]
    for orig_key, aesthetic_name in FIGURE2_MODEL_NAMES.items()
    if orig_key in MODEL_COLORS
}

FIGURE2_MODEL_ORDER = list(dict.fromkeys(FIGURE2_MODEL_NAMES.values()))

ENTROPY_EXCLUDED_MODELS = {"6-CellTypist", "7-scmap-cells", "8-scmap-cluster"}

METRICS = [
    ('test_balanced_acc',        'Balanced accuracy'),
    ('full_batch_mixing_entropy', 'Batch Mixing Entropy'),
]


def _extract_pct(parameter):
    """Extract pct split value from parameter string.
    scMusketeers: pct_seed (2 parts)
    others:       fold_pct_seed (3 parts)
    """
    parts = str(parameter).split('_')
    return float(parts[0] if len(parts) == 2 else parts[1])


def produce_fig_3(checkpoint_base_path, working_dir, run_stats=False):
    task2_checkpoint_path = checkpoint_base_path.replace('.csv', '_task_2.csv')
    if not os.path.exists(task2_checkpoint_path):
        logger.error(f"Checkpoint file not found at {task2_checkpoint_path}. Run csv_process first!")
        return

    logger.info(f"Loading checkpoint from {task2_checkpoint_path}...")
    runs_table_df = pd.read_csv(task2_checkpoint_path)

    if runs_table_df.empty:
        logger.error("No valid task 2 runs found in checkpoint. Cannot plot figures.")
        return

    logger.info("Building DataFrames for Figure task2 plots...")
    task_dict_acc = {}

    for key in diff_dict.keys():
        df_acc = pd.DataFrame()
        for ds in diff_dict[key]:
            if ds not in test_fold_selection:
                continue
            expected_test_fold = test_fold_selection[ds]
            ds_df = runs_table_df.query(
                f"dataset_name == '{ds}' and test_fold_nb == {expected_test_fold}"
            )
            for model in ds_df['model'].unique():
                sub = ds_df.query(f"model == '{model}'").copy()
                if not sub.empty:
                    sub['aestetic_data_name'] = aestetic_data_name.get(ds, ds)
                    sub['aestetic_model_name'] = FIGURE2_MODEL_NAMES.get(model, model)
                    sub['pct'] = sub['parameter'].apply(_extract_pct)
                    df_acc = pd.concat([df_acc, sub])
        task_dict_acc[key] = df_acc

    logger.info("Generating Figure task2 sub-figures...")

    model_order = FIGURE2_MODEL_ORDER

    figures_dir = os.path.join(working_dir, 'analysis_notebooks', 'figure_review', 'task_2')
    os.makedirs(figures_dir, exist_ok=True)

    for key, title_name in CATEGORY_TITLES.items():
        df_plot = task_dict_acc.get(key, pd.DataFrame())
        if df_plot.empty:
            logger.warning(f"No data for category '{key}', skipping.")
            continue

        datasets = df_plot['aestetic_data_name'].unique()
        n_rows = len(datasets)
        base_order = [m for m in model_order if m in df_plot['aestetic_model_name'].unique()]
        pct_order = sorted(df_plot['pct'].unique())

        fig, axes = plt.subplots(n_rows, 2, figsize=(14, n_rows * 3.5))
        if n_rows == 1:
            axes = axes.reshape(1, 2)

        fig.suptitle(title_name, fontsize=16)

        for col_idx, (_, col_title) in enumerate(METRICS):
            axes[0, col_idx].set_title(col_title, fontsize=11)

        for row_idx, dataset in enumerate(datasets):
            ds_df = df_plot[df_plot['aestetic_data_name'] == dataset]

            for col_idx, (metric_col, ylabel) in enumerate(METRICS):
                ax = axes[row_idx, col_idx]
                order = base_order if metric_col != 'full_batch_mixing_entropy' else \
                    [m for m in base_order if m not in ENTROPY_EXCLUDED_MODELS]

                sns.boxplot(
                    data=ds_df,
                    x='pct',
                    y=metric_col,
                    hue='aestetic_model_name',
                    order=pct_order,
                    hue_order=order,
                    palette=FIGURE2_MODEL_COLORS,
                    showfliers=False,
                    ax=ax,
                )
                sns.stripplot(
                    data=ds_df,
                    x='pct',
                    y=metric_col,
                    hue='aestetic_model_name',
                    order=pct_order,
                    hue_order=order,
                    size=3,
                    color='black',
                    dodge=True,
                    legend=False,
                    ax=ax,
                )

                ax.set_ylabel(ylabel if col_idx == 0 else '')
                ax.set_xlabel('Training split (pct)')
                ax.set_title('')

                # Dataset label on the right side of the row
                if col_idx == 1:
                    ax.annotate(
                        dataset, xy=(1.02, 0.5), xycoords='axes fraction',
                        va='center', ha='left', fontsize=9, rotation=270
                    )

                # Legend only on right column
                if col_idx == 0:
                    ax.get_legend().remove()
                else:
                    ax.legend(
                        bbox_to_anchor=(1.12, 1), loc='upper left',
                        borderaxespad=0., fontsize=7, title='Model', title_fontsize=8
                    )

                if run_stats:
                    baseline_name = FIGURE2_MODEL_NAMES.get('scPermut_default', '1-scMusketeers')
                    pairs = []
                    for pct in pct_order:
                        pct_df = ds_df[ds_df['pct'] == pct]
                        for m in order:
                            if m == baseline_name:
                                continue
                            if (not pct_df[pct_df['aestetic_model_name'] == baseline_name].empty
                                    and not pct_df[pct_df['aestetic_model_name'] == m].empty):
                                pairs.append(((pct, baseline_name), (pct, m)))
                    if pairs:
                        annot = Annotator(ax, pairs, data=ds_df, x='pct', y=metric_col,
                                          hue='aestetic_model_name', order=pct_order,
                                          hue_order=order)
                        try:
                            annot.configure(test='Wilcoxon', text_format='star', loc='inside',
                                            comparisons_correction="Benjamini-Hochberg")
                            annot.apply_test().annotate()
                        except Exception as e:
                            logger.debug(f"Stats error on '{dataset}' for {metric_col}: {e}")

        plt.tight_layout()
        output_path = os.path.join(figures_dir, f"Figure_task2_{key}.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.savefig(output_path.replace('.png', '.svg'), bbox_inches='tight')
        logger.info(f"Saved {output_path}")
        plt.close()
