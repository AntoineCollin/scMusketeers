import neptune
from neptune.utils import stringify_unsupported

try:
    from .optimize_hp import Workflow
except ImportError:
    from ..transfer.hyperparameters_chris import Workflow


def start_neptune_log(workflow: Workflow):
    if workflow.run_file.log_neptune:
        # self.run = neptune.init_run(
        #         project="becavin-lab/benchmark",
        #         api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiMmRkMWRjNS03ZGUwLTQ1MzQtYTViOS0yNTQ3MThlY2Q5NzUifQ==",
        # )
        workflow.run_neptune = neptune.init_run(
            project="sc-permut-packaging",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
        )
        workflow.run_neptune["parameters/model"] = "scPermut"
        for par, val in workflow.run_file.__dict__.items():
            if par in dir(workflow):
                workflow.run_neptune[f"parameters/{par}"] = (
                    stringify_unsupported(getattr(workflow, par))
                )
            # elif par in dir(workflow.ae_param):
            #    workflow.run_neptune[f"parameters/{par}"] = stringify_unsupported(getattr(workflow.ae_param, par))
        if (
            workflow.hp_params
        ):  # Overwrites the defaults arguments contained in the runfile
            for par, val in workflow.hp_params.items():
                workflow.run_neptune[f"parameters/{par}"] = (
                    stringify_unsupported(val)
                )


def add_custom_log(workflow: Workflow, name, value):
    workflow.run_neptune[f"parameters/{name}"] = stringify_unsupported(value)


def stop_neptune_log(workflow: Workflow):
    workflow.run_neptune.stop()
