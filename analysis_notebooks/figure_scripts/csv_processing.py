import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# scMusketeers folders use this legacy display name in the output DataFrames
_MODEL_DISPLAY_NAME = {
    "scMusketeers": "scPermut_default",
}

# Reverse mapping, to write failed runs back in the original CSV naming
_MODEL_ORIGINAL_NAME = {v: k for k, v in _MODEL_DISPLAY_NAME.items()}

# A run whose test accuracy AND balanced accuracy are both exactly 0 is treated
# as a failed run: the job crashed / produced no metric, so _read_metrics
# defaulted it to 0.0. (Low-but-nonzero scores are kept as genuine results.)


def _load_completed_entries(paper_review_dir):
    """Read completed_benchmark_runs_taskX.csv files and return list of (dataset, task, model, parameter) tuples."""
    entries = []
    for task_num in ["1", "1_New", "2", "2_New"]:
        csv_path = os.path.join(paper_review_dir, f"completed_benchmark_runs_task{task_num}.csv")
        if not os.path.isfile(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                entries.append((row["Dataset Name"], str(row["Task"]), row["Model"], row["Parameter"]))
        except Exception as e:
            logger.warning(f"Could not read {csv_path}: {e}")
    return entries


def _reconstruct_folder_name(dataset, task, model, parameter):
    """Reconstruct the result folder name from CSV row fields.

    Naming conventions (mirroring 00_benchmark_sbatch_review.py):
      task1 scMusketeers : {dataset}_task_1_{parameter}
      task1 others       : {dataset}_task_1_{model}_{parameter}
      task2 any model    : {dataset}_task_2_{model}_{parameter}
    """
    if task == "1" and model == "scMusketeers":
        return f"{dataset}_task_1_{parameter}"
    elif task in ("1", "1_New"):
        return f"{dataset}_task_1_{model}_{parameter}"
    else:
        return f"{dataset}_task_2_{model}_{parameter}"


def _extract_test_fold_nb(parameter):
    """Extract integer test fold number from the first component of a parameter string."""
    if parameter and "_" in parameter:
        try:
            first_part = parameter.split("_")[0]
            if "." not in first_part:
                return int(first_part)
        except ValueError:
            pass
    return -1


def _read_metrics(metrics_file):
    metrics = pd.read_csv(metrics_file, index_col=0, header=None).squeeze("columns")
    return {
        "val_acc": metrics.get("evaluation/val/acc", 0.0),
        "test_acc": metrics.get("evaluation/test/acc", 0.0),
        "val_balanced_acc": metrics.get("evaluation/val/balanced_acc", 0.0),
        "test_balanced_acc": metrics.get("evaluation/test/balanced_acc", 0.0),
        "full_batch_mixing_entropy": metrics.get("evaluation/full/batch_mixing_entropy", 0.0),
    }


def detect_failed_runs(runs_table_df, paper_review_dir):
    """Flag runs whose test accuracy AND balanced accuracy are both exactly 0.

    These indicate a failed job (crashed, ran out of memory, etc.) whose metric
    defaulted to 0.0 rather than a genuine, low-but-nonzero result.

    Writes missing_benchmark_runs_task{T}_failed.csv per task in
    `paper_review_dir`, using the same Dataset/Task/Model/Parameter columns as
    the missing-runs CSVs so the failed runs can be re-submitted with the
    complete scripts.

    Returns the failed-runs DataFrame.
    """
    if runs_table_df.empty:
        return pd.DataFrame()

    failed = runs_table_df[
        (runs_table_df["test_acc"] == 0)
        & (runs_table_df["test_balanced_acc"] == 0)
    ].copy()

    if failed.empty:
        logger.info(
            "No failed runs detected (no run has both test_acc and "
            "test_balanced_acc == 0)."
        )
        return failed

    logger.warning(
        f"Detected {len(failed)} failed run(s) with test_acc AND "
        f"test_balanced_acc both == 0:"
    )
    for _, r in failed.iterrows():
        logger.warning(
            f"  {r['dataset_name']} | {r['task']} | {r['model']} | "
            f"{r['parameter']} (acc={r['test_acc']:.3f}, "
            f"bal_acc={r['test_balanced_acc']:.3f})"
        )

    # Write per-task CSVs in the missing-runs format so they can be re-submitted.
    os.makedirs(paper_review_dir, exist_ok=True)
    out = failed.copy()
    out["Dataset Name"] = out["dataset_name"]
    out["Task"] = out["task"].str.replace("task_", "", regex=False)
    out["Model"] = out["model"].map(_MODEL_ORIGINAL_NAME).fillna(out["model"])
    out["Missing Parameter"] = out["parameter"]
    cols = ["Dataset Name", "Task", "Model", "Missing Parameter"]
    for task_label, g in out.groupby("Task"):
        path = os.path.join(
            paper_review_dir, f"missing_benchmark_runs_task{task_label}_failed.csv"
        )
        g[cols].to_csv(path, index=False)
        logger.warning(f"Wrote {len(g)} failed run(s) to {path}")

    return failed


def csv_process(results_dir, checkpoint_path, task_filter=None):
    """Process result metrics into per-task checkpoint CSVs.

    Args:
        task_filter: optional list of task numbers to process, e.g. ["1"] or
                     ["1_New", "2", "2_New"]. None means process all tasks.
    """
    filter_label = f" (tasks: {task_filter})" if task_filter else " (all tasks)"
    logger.info(f"Parsing result directories to build metrics table{filter_label}...")
    data = []

    if not os.path.exists(results_dir):
        logger.error(f"Results directory not found: {results_dir}")
        return pd.DataFrame()

    paper_review_dir = os.path.abspath(
        os.path.join(results_dir, "..", "benchmark", "paper_review")
    )
    completed_entries = _load_completed_entries(paper_review_dir)

    if task_filter is not None:
        completed_entries = [e for e in completed_entries if e[1] in task_filter]

    if not completed_entries:
        logger.error(
            f"No completed runs CSVs found in {paper_review_dir}. "
            "Run 00_benchmark_sbatch_review.py first to generate them."
        )
        return pd.DataFrame()

    logger.info(f"Loaded {len(completed_entries)} completed runs from {paper_review_dir}.")
    for dataset, task, model, parameter in completed_entries:
        folder_name = _reconstruct_folder_name(dataset, task, model, parameter)
        metrics_file = os.path.join(results_dir, folder_name, "all_metrics.csv")
        if not os.path.isfile(metrics_file):
            logger.debug(f"Metrics file missing: {metrics_file}")
            continue
        try:
            data.append({
                "dataset_name": dataset,
                "task": f"task_{task}",
                "model": _MODEL_DISPLAY_NAME.get(model, model),
                "test_fold_nb": _extract_test_fold_nb(parameter),
                "parameter": parameter,
                **_read_metrics(metrics_file),
            })
        except Exception as e:
            logger.debug(f"Failed to process {folder_name}: {e}")

    runs_table_df = pd.DataFrame(data)
    logger.info(f"Finished parsing. Extracted {len(runs_table_df)} valid runs.")

    # Flag failed runs (accuracy and balanced accuracy both 0) and drop them so
    # they don't pollute the checkpoints / figures.
    failed = detect_failed_runs(runs_table_df, paper_review_dir)
    if not failed.empty:
        runs_table_df = runs_table_df.drop(failed.index)
        logger.info(
            f"Removed {len(failed)} failed run(s) from the checkpoint "
            f"({len(runs_table_df)} runs kept)."
        )

    if not runs_table_df.empty:
        for t, group_df in runs_table_df.groupby("task"):
            sorted_df = group_df.sort_values(by=["dataset_name", "model", "test_fold_nb"])
            task_checkpoint_path = checkpoint_path.replace(".csv", f"_{t}.csv")
            logger.info(f"Saving checkpoint for {t} to {task_checkpoint_path}")
            sorted_df.to_csv(task_checkpoint_path, index=False)

    return runs_table_df
