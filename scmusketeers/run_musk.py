import os

from scmusketeers.arguments.neptune_log import (start_neptune_log,
                                                stop_neptune_log)
from scmusketeers.arguments.runfile import (PROCESS_TYPE, create_argparser,
                                            get_default_param, get_runfile)
from scmusketeers.transfer.experiment import MakeExperiment
from scmusketeers.transfer.optimize_model import Workflow
from scmusketeers.workflow.run_workflow import run_workflow



def run_sc_musketeers(run_file):

    if run_file.process == PROCESS_TYPE[0]:
        # Transfer data
        workflow = Workflow(run_file=run_file)
        start_neptune_log(workflow)
        workflow.process_dataset()
        workflow.train_val_split()
        adata_pred, model, history, X_scCER, query_pred = (
            workflow.make_experiment()
        )
        stop_neptune_log(workflow)
        print(query_pred)
        adata_pred_path = os.path.join(
            run_file.out_dir, f"{run_file.out_name}.h5ad"
        )
        print((f"Save adata_pred to {adata_pred_path}"))
        adata_pred.write_h5ad(adata_pred_path)

    elif run_file.process == PROCESS_TYPE[1]:
        # Hyperparameter optimization
        run_workflow(run_file)
    else:
        # No process
        print("Process not recognized")