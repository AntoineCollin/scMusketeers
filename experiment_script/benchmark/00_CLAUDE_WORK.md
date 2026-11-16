# Claude Code — Work Summary

## task1_New sbatch refactoring
- Removed the inner `for test_fold_j_nb in 0..4` loop from `task1_New_all_benchmark.sh` and `task1_New_scMusk_all_dataset.sh` — one sbatch job per (dataset, model, fold_i) instead of one per j value
- Updated `task1_New_allmodels_benchmark.sh` and `task1_New_scMusk_benchmark.sh` to accept optional j indices as trailing positional args (`${@:N}`) passed to `--test_fold_j_nb`
- Updated `task1_New_allmodels_label_transfer_between_batch.py` and `task1_New_scMusk_label_transfer_between_batch.py` to accept `--test_fold_j_nb` as `nargs='+'` (list of ints), with a per-j skip guard

## task1_New complete script
- Rewrote `task1_New_all_benchmark_complete.sh`: AWK groups missing runs by `(dataset, model, fold_i)` and collects missing j values per group
- Added inner `for j in $fold_js` loop: submits ONE sbatch per missing j value so they run in parallel
- Job names and log files include the j value: e.g. `t1n_complete_CellCards-Lung_0_j2`

## task2_New seed handling
- Updated `task2_New_allmodels_label_transfer_pct_split.py` and `task2_New_scMusk_label_transfer_pct_split.py` to accept `--random_seed_nb` as `nargs='+'`
- Updated `task2_New_allmodels_benchmark.sh` and `task2_New_scMusk_benchmark.sh` to pass optional seed indices as trailing args
- Rewrote `task2_New_all_benchmark_complete.sh`: AWK groups by `(dataset, model, fold, pct)`, converts seed values to indices (`seed_val - 30`)
- Added inner `for s in $seed_ids` loop: submits ONE sbatch per missing seed so they run in parallel
- Job names and log files include the seed index: e.g. `t2n_complete_TS-Blood_0_2_s3`

## Figure generation — csv_process separation
- Added `task_filter` parameter to `csv_process()` in `csv_processing.py` so only selected tasks are processed
- Updated `figure_generation.py` to call `csv_process` separately for task1 and for the task2_New group (task1_New + task2 + task2_New), driven by the active task flags

## Figure generation — new datasets and size grouping
- Added `SmallIntestine-All` and `SmallIntestine-20k` to `generate_figure_task1_New.py` and `generate_figure_task2_New.py`
- Replaced the `tabula_sapiens / others` dataset grouping with 3 size-based groups:
  - **small** (≤ 20k cells): TS-Neural, TS-Skin, SmallIntestine-20k
  - **medium** (20k–100k cells): TS-Liver, TS-BoneMarrow, PBMC-Lee, TS-Blood
  - **large** (> 100k cells): Ageing-Mouse-All, SmallIntestine-All, CellCards-Lung

## UCE / GPU diagnostics
- Diagnosed `ModuleNotFoundError: No module named 'pkg_resources'` in UCE env → fix: `pip install setuptools`
- Diagnosed `cudaErrorNoKernelImageForDevice` on gpu02 (V100, sm_70) → cause: PyTorch 2.7.1 dropped sm_70 support
- Fix: create `UCE_v100` conda env with PyTorch 2.4.0+cu121 (last version with solid V100 support)
- Workaround for `conda create --clone` failure with pip packages: export via `pip freeze`, filter out torch/nvidia/triton, recreate env manually

## SLURM job management
- Cancelled batches of `t2n_complete_*`, `t2n_allm`, and `t1n_complete_*` jobs (GPU queue overflow / wrong node / resubmission)
- Tip: use `scancel $(seq FIRST LAST)` for contiguous job ID ranges

## 00_benchmark_sbatch_review.py verification
- Confirmed script correctly identifies missing runs — folders listed as missing genuinely do not exist in results/
- Missing CSV is stale after jobs complete: re-run `00_benchmark_sbatch_review.py --task2-new` (or `--task1-new`) to refresh before submitting complete scripts

# Work day — 2026-06-09

## Raw accuracy metric added to the pipeline
- `csv_processing.py` `_read_metrics`: now also extracts `val_acc` (`evaluation/val/acc`) and `test_acc` (`evaluation/test/acc`) into the checkpoint, alongside the existing balanced-accuracy and entropy columns
- `generate_figure_task1_New.py` and `generate_figure_task2_New.py`: added a third "Accuracy" sub-figure (`test_acc`) → prefixes `Figure_task1_New_Accuracy` / `Figure_task2_New_Accuracy`; added `("test_acc", "Accuracy")` to the task2 `METRICS` list
- Pipeline now produces **3 metrics × 2 tasks × 3 size groups = 18 figures** (was 12); requires re-running `--process_csv` so the new `test_acc` column is populated
- Note: all plotted accuracies use the held-out **test** split (`test_acc`, `test_balanced_acc`); entropy uses the **full** embedding. `val_*` columns are stored but currently unused by the figure scripts (dead in pipeline, kept for diagnostics)

## Key finding — balanced accuracy is the right metric
- Raw accuracy is dominated by abundant cell types: PCA/UCE top raw accuracy (~0.86/0.85) but lag on balanced accuracy (0.65/0.63) — a ~0.21 gap = poor on rare types
- scMusketeers has the smallest acc-to-balanced-acc gap of any embedding method (~0.10): annotates rare and common types evenly. Confirms the choice of balanced accuracy for the benchmark

## NAR revision LaTeX document
- Created `analysis_notebooks/figure_review/benchmark_review_figures.tex` for the Nucleic Acids Research revision
- Contents: Overview + dataset characteristics table (from `data/cellxgene_datasets/List-datasets-2026.xlsx`), Analysis (5 paragraphs), then all 18 figure legends
- Figure paths prefixed with `task_1_New/` and `task_2_New/` (also keeps `\graphicspath`)
- Section order: **Overview → Analysis → Figure legends** (Analysis moved before the legends)
- Numbers in the prose come from the current (partial) checkpoints — re-verify after the remaining runs finish (~Wed) and figures are regenerated
- No LaTeX engine on the machine; document is standard and compiles elsewhere with `pdflatex`
