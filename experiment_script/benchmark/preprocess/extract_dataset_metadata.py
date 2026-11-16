#!/usr/bin/env python
"""Extract metadata for the new benchmark datasets.

Goes through every dataset used in the task1_New / task2_New benchmark and
reports, for each one:
    - number of cells          (adata.n_obs)
    - number of genes          (adata.n_vars)
    - number of batches        (unique values of batch_key, default "donor_id")
    - number of tissues        (unique values of "tissue")
    - number of cell types     (unique values of class_key, default "cell_type")

Dataset names and the class_key / batch_key conventions are taken from
    experiment_script/benchmark/task1_New/task1_New_all_benchmark.sh
and the name -> file mapping mirrors
    scmusketeers/workflow/dataset.py :: load_dataset

The AnnData objects are opened in backed mode so only `.obs` is read into
memory (the expression matrices, up to several GB, are never loaded).

Usage:
    python extract_dataset_metadata.py
    python extract_dataset_metadata.py --output my_metadata.csv
"""

import argparse
import logging
import os

import anndata as ad
import pandas as pd

logger = logging.getLogger(__name__)

# Repository root: this file is at <root>/experiment_script/benchmark/preprocess/
WORKING_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DATA_DIR = os.path.join(WORKING_DIR, "data", "cellxgene_datasets")

# Keys shared by all task1_New / task2_New datasets (see *_all_benchmark.sh)
CLASS_KEY = "cell_type"
BATCH_KEY = "donor_id"
TISSUE_KEY = "tissue"

# Datasets benchmarked in task1_New / task2_New (cellxgene_datasets/<name>.h5ad)
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


def _n_unique(obs, key):
    """Number of unique values of `key` in obs, or None if the column is absent."""
    if key not in obs.columns:
        logger.warning(f"    column '{key}' not found in obs")
        return None
    return int(obs[key].nunique())


def extract_metadata(datasets, data_dir, class_key, batch_key):
    rows = []
    for name in datasets:
        path = os.path.join(data_dir, f"{name}.h5ad")
        if not os.path.isfile(path):
            logger.warning(f"[skip] {name}: file not found ({path})")
            continue

        logger.info(f"Reading {name} ...")
        # backed='r' loads obs/var but leaves X on disk
        adata = ad.read_h5ad(path, backed="r")
        obs = adata.obs

        rows.append(
            {
                "dataset_name": name,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_batch": _n_unique(obs, batch_key),
                "n_tissue": _n_unique(obs, TISSUE_KEY),
                "n_cell_type": _n_unique(obs, class_key),
            }
        )
        adata.file.close()

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "new_datasets_metadata.csv"),
        help="Output CSV path",
    )
    parser.add_argument("--data_dir", default=DATA_DIR, help="Folder with the .h5ad files")
    parser.add_argument("--class_key", default=CLASS_KEY, help="obs column for cell type")
    parser.add_argument("--batch_key", default=BATCH_KEY, help="obs column for batch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    df = extract_metadata(NEW_DATASETS, args.data_dir, args.class_key, args.batch_key)
    if df.empty:
        logger.error("No dataset could be read.")
        return

    # Sort by cell count, mirroring the size grouping used in the figures
    df = df.sort_values("n_cells").reset_index(drop=True)

    df.to_csv(args.output, index=False)
    logger.info(f"\nSaved metadata for {len(df)} datasets to {args.output}\n")
    logger.info(df.to_string(index=False))


if __name__ == "__main__":
    main()
