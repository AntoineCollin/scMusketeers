import os
import logging

from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)
from scmusketeers.transfer.optimize_model import Workflow

logger = logging.getLogger("Sc-Musketeers")

def run_sc_musketeers(run_file):
    # Run transfer
    if run_file.process == PROCESS_TYPE[0]:
        adata_pred_path = os.path.join(
            run_file.out_dir, f"{run_file.out_name}.h5ad"
        )
        # Summarise what this run is going to do.
        logger.info("Cell type annotation transfer is starting")
        logger.info(f"  Reference dataset : {run_file.ref_path}")
        if run_file.query_path:
            logger.info(f"  Query dataset     : {run_file.query_path}")
        logger.info(
            f"  Cell type key '{run_file.class_key}', batch key "
            f"'{run_file.batch_key}'"
        )
        logger.info(
            f"  Predicting cells labelled '{run_file.unlabeled_category}'"
        )
        logger.info(f"  Output annotated dataset : {adata_pred_path}")

        # Transfer data
        workflow = Workflow(run_file=run_file)
        workflow.process_dataset()
        workflow.train_val_split()
        adata_pred, model, history, X_scMusk, query_pred = (
            workflow.make_experiment()
        )

        # Report where each result is stored inside the output AnnData.
        class_key = run_file.class_key
        logger.info(f"Saving annotated dataset to {adata_pred_path}")
        logger.info(
            f"  Predicted cell types stored in   adata.obs['{class_key}_scMusk']"
        )
        logger.info(
            f"  Class probabilities stored in    adata.obsm['{class_key}_scMusk_proba']"
        )
        logger.info(
            "  scMusketeers embedding stored in adata.obsm['X_scMusk']"
        )

        # anndata prints an INFO line for every column it converts to
        # categorical on write; silence that noise.
        logging.getLogger("anndata").setLevel(logging.WARNING)
        adata_pred.write_h5ad(adata_pred_path)
        logger.info("Cell type annotation transfer is done")

    # Run hyperparameters optimization
    elif run_file.process == PROCESS_TYPE[1]:
        # Imported lazily: the optim/benchmark workflow needs the optional
        # "workflow" dependency group (ax, scvi, celltypist, matplotlib, ...),
        # which the core transfer path does not require.
        from scmusketeers.workflow.run_workflow import run_workflow

        run_workflow(run_file)
    else:
        # No process
        print("Process not recognized")