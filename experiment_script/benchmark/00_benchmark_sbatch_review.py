import os
import sys
import csv
import argparse

MODELS = ["scMusketeers", "celltypist", "scmap_cells", "scmap_cluster", "pca_knn", "pca_svm", "scanvi", "harmony_svm", "uce"]

# Populated by scan_results() from the results directory
DATASETS = ['ajrccm_by_batch', 'dominguez_2022_lymph', 'dominguez_2022_spleen', 'hlca_par_dataset_harmonized', 'hlca_trac_dataset_harmonized', 'htap', 'koenig_2022', 'lake_2021', 'litvinukova_2020', 'tabula_2022_spleen', 'tosti_2021', 'yoshida_2021']
DATASETS_TASK1_NEW = ['Ageing-Mouse-All', 'CellCards-Lung', 'PBMC-Lee', 'SmallIntestine-20k', 'SmallIntestine-All', 'TS-Blood', 'TS-BoneMarrow', 'TS-Liver', 'TS-Neural', 'TS-Skin']

TASKS = ['1', '2']
TASK1_TESTS = ['0', '1', '2']   # test fold indices (i) for task1
TASK1_VALS = ['0', '1', '2', '3', '4']    # val fold indices (j) for task1
TASK2_FOLDS = ['0', '1', '2']   # test fold indices (i) for task2 non-scMusk
TASK2_PCTS = ['0.05', '0.1', '0.5', '0.9']    # pct_split values for task2
TASK2_SEEDS = ['30', '31', '32', '33', '34', '35']   # random_seed values for task2

# task1_New shares the same fold structure as task1
TASK1_NEW_TESTS = ['0', '1', '2']
TASK1_NEW_VALS = ['0', '1', '2', '3', '4']


def _datasets_for_task(task):
    """Return the dataset list that applies to a given task identifier."""
    if task in ("1_New", "2_New"):
        return DATASETS_TASK1_NEW
    return DATASETS


def parse_folder_name(folder_name):
    """Parse a result folder name into a dict with dataset/task/model/params info.

    Folder naming conventions (task1_New reuses the same _task_1_ pattern as task1;
    the task is disambiguated by checking the dataset name against DATASETS_TASK1_NEW):
      task1 / task1_New scMusketeers : {dataset}_task_1_{i}_{j}
      task1 / task1_New others       : {dataset}_task_1_{model}_{i}_{j}
      task2 scMusketeers             : {dataset}_task_2_scMusketeers_{pct}_{seed}
      task2 others                   : {dataset}_task_2_{model}_{fold}_{pct}_{seed}
    """
    for task_num in ["1", "2"]:
        sep = f"_task_{task_num}_"
        if sep not in folder_name:
            continue
        idx = folder_name.index(sep)
        dataset = folder_name[:idx]
        rest = folder_name[idx + len(sep):]

        model = None
        params_str = rest

        # task1 / task1_New scMusketeers folders have no model prefix: rest starts with a digit
        if task_num in ("1", "1_New") and rest and rest[0].isdigit():
            model = "scMusketeers"
        else:
            for m in sorted(MODELS, key=len, reverse=True):
                if rest.startswith(m + "_"):
                    model = m
                    params_str = rest[len(m) + 1:]
                    break

        if model is None:
            continue

        parts = params_str.split("_") if params_str else []

        if task_num == "1":
            if len(parts) == 2:
                # Promote to task 1_New when the dataset belongs to the new set
                effective_task = "1_New" if dataset in DATASETS_TASK1_NEW else "1"
                return {"dataset": dataset, "task": effective_task, "model": model,
                        "test": parts[0], "val": parts[1]}
        elif task_num == "2":
            # Promote to task 2_New when the dataset belongs to the new set
            effective_task = "2_New" if dataset in DATASETS_TASK1_NEW else "2"
            if model == "scMusketeers" and len(parts) == 2:
                # Legacy task2 format: pct_seed (no fold).
                # Not valid for task2_New where fold is required.
                if dataset in DATASETS_TASK1_NEW:
                    return None
                return {"dataset": dataset, "task": "2", "model": model,
                        "fold": None, "pct": parts[0], "seed": parts[1]}
            elif len(parts) == 3:
                # task2_New format for all models: fold_pct_seed
                return {"dataset": dataset, "task": effective_task, "model": model,
                        "fold": parts[0], "pct": parts[1], "seed": parts[2]}
    return None


def scan_results(results_dir):
    """Scan results folder and populate global DATASETS, TASKS, and parameter lists."""
    global DATASETS, TASKS, TASK1_TESTS, TASK1_VALS, TASK1_NEW_TESTS, TASK1_NEW_VALS
    global TASK2_FOLDS, TASK2_PCTS, TASK2_SEEDS

    datasets, tasks = set(), set()
    t1_tests, t1_vals = set(), set()
    t1n_tests, t1n_vals = set(), set()
    t2_folds, t2_pcts, t2_seeds = set(), set(), set()

    for entry in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, entry)):
            continue
        parsed = parse_folder_name(entry)
        if parsed is None:
            continue
        datasets.add(parsed["dataset"])
        tasks.add(parsed["task"])
        if parsed["task"] == "1":
            t1_tests.add(parsed["test"])
            t1_vals.add(parsed["val"])
        elif parsed["task"] == "1_New":
            t1n_tests.add(parsed["test"])
            t1n_vals.add(parsed["val"])
        elif parsed["task"] in ("2", "2_New"):
            if parsed["fold"] is not None:
                t2_folds.add(parsed["fold"])
            t2_pcts.add(parsed["pct"])
            # Seeds are integers (30-35); skip legacy float seeds (0.1, 0.9)
            try:
                if float(parsed["seed"]) >= 1:
                    t2_seeds.add(parsed["seed"])
            except ValueError:
                t2_seeds.add(parsed["seed"])

    DATASETS = sorted(d for d in datasets if d not in DATASETS_TASK1_NEW)
    TASKS = sorted(tasks)
    if t1_tests:
        TASK1_TESTS = sorted(t1_tests, key=int)
    if t1_vals:
        TASK1_VALS = sorted(t1_vals, key=int)
    if t1n_tests:
        TASK1_NEW_TESTS = sorted(t1n_tests, key=int)
    if t1n_vals:
        TASK1_NEW_VALS = sorted(t1n_vals, key=int)
    TASK2_FOLDS = sorted(t2_folds, key=int)
    TASK2_PCTS = sorted(t2_pcts, key=float)
    TASK2_SEEDS = sorted(t2_seeds, key=float)


def get_expected_parameters(task, model=None, selected_tests=None):
    """Return expected parameter strings for a given task and model.

    selected_tests restricts the task1 test folds (used for scMusketeers task1,
    which only ran the single selected "most detrimental" fold per dataset; see
    hp_test_obs.json and task1/task1_scMusk_label_transfer_between_batch.py).
    """
    expected_params = []
    if task in ("1", "1_New"):
        tests = TASK1_NEW_TESTS if task == "1_New" else TASK1_TESTS
        vals = TASK1_NEW_VALS if task == "1_New" else TASK1_VALS
        # task1 (not task1_New) scMusketeers was run on only one test fold per
        # dataset: the fold whose test batches match hp_test_obs.json[dataset].
        # That fold index varies per dataset and can't be recomputed here without
        # the AnnData, so it is derived from the completed runs via selected_tests.
        if task == "1" and model == "scMusketeers" and selected_tests is not None:
            tests = selected_tests
        for test in tests:
            for val in vals:
                expected_params.append(f"{test}_{val}")
    elif task in ("2", "2_New"):
        if model == "scMusketeers" and task == "2":
            # Legacy task2: no fold in parameter
            for pct in TASK2_PCTS:
                for seed in TASK2_SEEDS:
                    expected_params.append(f"{pct}_{seed}")
        else:
            # task2_New (all models) and task2 non-scMusk: fold_pct_seed
            for fold in TASK2_FOLDS:
                for pct in TASK2_PCTS:
                    for seed in TASK2_SEEDS:
                        expected_params.append(f"{fold}_{pct}_{seed}")
    return expected_params


def find_completed_runs(results_dir):
    """Search for folders containing all_metrics.csv and extract run info."""
    completed_runs = []
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        metrics_path = os.path.join(folder_path, "all_metrics.csv")
        if not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0:
            continue
        parsed = parse_folder_name(folder)
        if parsed is None:
            completed_runs.append({"dataset": folder, "task": "Unknown",
                                   "model": "Unknown", "parameter": "Unknown"})
            continue
        if parsed["task"] in ("1", "1_New"):
            parameter = f"{parsed['test']}_{parsed['val']}"
        elif parsed["task"] in ("2", "2_New"):
            if parsed["fold"] is None:
                parameter = f"{parsed['pct']}_{parsed['seed']}"
            else:
                parameter = f"{parsed['fold']}_{parsed['pct']}_{parsed['seed']}"
        completed_runs.append({
            "dataset": parsed["dataset"],
            "task": parsed["task"],
            "model": parsed["model"],
            "parameter": parameter,
        })
    return completed_runs


def save_to_csv(completed_runs, output_dir, selected_tasks):
    """Save completed runs data to separate CSV files by task."""
    if not completed_runs:
        return
    os.makedirs(output_dir, exist_ok=True)

    runs_by_task = {}
    for r in completed_runs:
        task = r["task"]
        if task not in selected_tasks:
            continue
        runs_by_task.setdefault(task, []).append(r)

    for task, runs in runs_by_task.items():
        csv_path = os.path.join(output_dir, f"completed_benchmark_runs_task{task}.csv")
        sorted_runs = sorted(runs, key=lambda x: (x["dataset"], x["model"], x["parameter"]))
        try:
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Dataset Name", "Task", "Model", "Parameter"])
                for r in sorted_runs:
                    writer.writerow([r["dataset"], r["task"], r["model"], r["parameter"]])
            print(f"Successfully saved table to {csv_path}")
        except Exception as e:
            print(f"Failed to save CSV {csv_path}: {e}")


def find_missing_runs(completed_runs, output_dir, selected_tasks):
    """Identify missing runs per dataset/task/model and save to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Build lookup: {dataset: {task: {model: set(parameters)}}}
    completed_mapped = {}
    for r in completed_runs:
        ds, task, model, param = r["dataset"], r["task"], r["model"], r["parameter"]
        if task not in selected_tasks:
            continue
        completed_mapped.setdefault(ds, {}).setdefault(task, {}).setdefault(model, set()).add(param)

    for task in selected_tasks:
        datasets = _datasets_for_task(task)
        if not datasets:
            continue
        missing_runs = []
        for ds in datasets:
            for model in MODELS:
                selected_tests = None
                if task == "1" and model == "scMusketeers":
                    # Restrict expected runs to the single selected test fold(s)
                    # actually run for this dataset (see get_expected_parameters).
                    done = completed_mapped.get(ds, {}).get(task, {}).get(model, set())
                    selected_tests = sorted({p.split("_")[0] for p in done})
                    # If scMusketeers never ran for this dataset we can't know its
                    # selected fold, so fall back to expecting fold 0.
                    if not selected_tests:
                        selected_tests = ["0"]
                for expected_param in get_expected_parameters(task, model, selected_tests):
                    completed = (ds in completed_mapped
                                 and task in completed_mapped[ds]
                                 and model in completed_mapped[ds][task]
                                 and expected_param in completed_mapped[ds][task][model])
                    if not completed:
                        missing_runs.append({"dataset": ds, "task": task,
                                             "model": model, "parameter": expected_param})

        if missing_runs:
            csv_path = os.path.join(output_dir, f"missing_benchmark_runs_task{task}.csv")
            try:
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Dataset Name", "Task", "Model", "Missing Parameter"])
                    for r in missing_runs:
                        writer.writerow([r["dataset"], r["task"], r["model"], r["parameter"]])
                print(f"Successfully saved missing runs to {csv_path}")
            except Exception as e:
                print(f"Failed to save CSV {csv_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Review benchmark run completeness.")
    parser.add_argument("--task1", action="store_true", help="Process task1 runs")
    parser.add_argument("--task1-new", action="store_true", dest="task1_new",
                        help="Process task1_New runs (new CellxGene datasets)")
    parser.add_argument("--task2", action="store_true", help="Process task2 runs")
    parser.add_argument("--task2-new", action="store_true", dest="task2_new",
                        help="Process task2_New runs (new CellxGene datasets)")
    args = parser.parse_args()

    selected_tasks = []
    if args.task1:
        selected_tasks.append("1")
    if args.task1_new:
        selected_tasks.append("1_New")
    if args.task2:
        selected_tasks.append("2")
    if args.task2_new:
        selected_tasks.append("2_New")
    if not selected_tasks:
        selected_tasks = ["1", "1_New", "2", "2_New"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.abspath(os.path.join(script_dir, "..", "results"))
    csv_dir = os.path.join(script_dir, "paper_review")

    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    scan_results(results_dir)
    print(f"Discovered {len(DATASETS)} original datasets: {DATASETS}")
    print(f"task1_New datasets: {DATASETS_TASK1_NEW}")
    print(f"Discovered tasks: {TASKS}")
    print(f"Task1    - tests: {TASK1_TESTS}, vals: {TASK1_VALS}")
    print(f"Task1New - tests: {TASK1_NEW_TESTS}, vals: {TASK1_NEW_VALS}")
    print(f"Task2    - folds: {TASK2_FOLDS}, pcts: {TASK2_PCTS}, seeds: {TASK2_SEEDS}")

    completed_runs = find_completed_runs(results_dir)
    save_to_csv(completed_runs, csv_dir, selected_tasks)
    find_missing_runs(completed_runs, csv_dir, selected_tasks)


if __name__ == "__main__":
    main()
