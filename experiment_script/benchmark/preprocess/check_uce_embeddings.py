import scanpy as sc
import os

DATA_DIR = "/workspace/cell/scMusketeers/data/cellxgene_datasets"

datasets = {
    "Ageing-Mouse-All": "mouse",
    "CellCards-Lung": "human",
    "PBMC-Lee": "human",
    "TS-Blood": "human",
    "TS-BoneMarrow": "human",
    "TS-Liver": "human",
    "TS-Neural": "human",
    "TS-Skin": "human",
}

print(f"{'Dataset':<25} {'Status':<10} {'Cells orig':>10} {'Cells uce':>10} {'Genes orig':>10} {'Genes uce':>10} {'X_uce shape':>15}")
print("-" * 95)

for dataset, species in datasets.items():
    orig_path = f"{DATA_DIR}/{dataset}.h5ad"
    uce_path  = f"{DATA_DIR}/{dataset}_uce_input.h5ad"

    orig_exists = os.path.exists(orig_path)
    uce_exists  = os.path.exists(uce_path)

    if not orig_exists:
        print(f"{dataset:<25} {'MISSING orig':<10}")
        continue
    if not uce_exists:
        print(f"{dataset:<25} {'MISSING uce':<10}")
        continue

    orig = sc.read_h5ad(orig_path, backed='r')
    uce  = sc.read_h5ad(uce_path,  backed='r')

    cells_match = orig.n_obs == uce.n_obs
    has_xuce    = "X_uce" in uce.obsm

    issues = []
    if not cells_match:
        issues.append("CELL MISMATCH")
    if not has_xuce:
        issues.append("NO X_uce")

    status  = "OK" if not issues else " | ".join(issues)
    xuce_shape = str(uce.obsm["X_uce"].shape) if has_xuce else "N/A"

    print(f"{dataset:<25} {status:<10} {orig.n_obs:>10} {uce.n_obs:>10} {orig.n_vars:>10} {uce.n_vars:>10} {xuce_shape:>15}")
