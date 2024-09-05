import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from ax.metrics.branin import branin
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.managed_loop import optimize
from ax.service.utils.report_utils import exp_to_df
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import evaluate, load_mnist, train
from torch._tensor import Tensor
from torch.utils.data import DataLoader


def hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    return {
        "hartmann6": (hartmann6(x), 0.0),
        "l2norm": (np.sqrt((x**2).sum()), 0.0),
    }


class CNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(8 * 8 * 20, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
        x = x.view(-1, 8 * 8 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


def train_evaluate(
    parameterization, train_loader, valid_loader, run_neptune, device
):
    """
    Train the model and then compute an evaluation metric.

    In this tutorial, the CNN utils package is doing a lot of work
    under the hood:
        - `train` initializes the network, defines the loss function
        and optimizer, performs the training loop, and returns the
        trained model.
        - `evaluate` computes the accuracy of the model on the
        evaluation dataset and returns the metric.

    For your use case, you can define training and evaluation functions
    of your choosing.

    """
    dtype = torch.float
    net = CNN()
    net = train(
        net=net,
        train_loader=train_loader,
        parameters=parameterization,
        dtype=dtype,
        device=device,
    )
    run_neptune["training/tf_GPU_memory_step"].append(
        tf.config.experimental.get_memory_info("GPU:0")["current"] / 1e6
    )

    return evaluate(
        net=net,
        data_loader=valid_loader,
        dtype=dtype,
        device=device,
    )


def cnn_training(run_neptune, device):
    torch.manual_seed(42)

    BATCH_SIZE = 1024
    train_loader, valid_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)

    ax_client = AxClient()

    # Create an experiment with required arguments: name, parameters, and objective_name.
    ax_client.create_experiment(
        name="tune_cnn_on_mnist",  # The name of the experiment.
        parameters=[
            {
                "name": "lr",  # The name of the parameter.
                "type": "range",  # The type of the parameter ("range", "choice" or "fixed").
                "bounds": [1e-6, 0.4],  # The bounds for range parameters.
                # "values" The possible values for choice parameters .
                # "value" The fixed value for fixed parameters.
                "value_type": "float",  # Optional, the value type ("int", "float", "bool" or "str"). Defaults to inference from type of "bounds".
                "log_scale": True,  # Optional, whether to use a log scale for range parameters. Defaults to False.
                # "is_ordered" Optional, a flag for choice parameters.
            },
            {
                "name": "momentum",
                "type": "range",
                "bounds": [0.0, 1.0],
            },
        ],
        objectives={
            "accuracy": ObjectiveProperties(minimize=False)
        },  # The objective name and minimization setting.
        # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
        # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
    )

    # Attach the trial
    ax_client.attach_trial(parameters={"lr": 0.000026, "momentum": 0.58})

    # Get the parameters and run the trial
    baseline_parameters = ax_client.get_trial_parameters(trial_index=0)
    ax_client.complete_trial(
        trial_index=0,
        raw_data=train_evaluate(
            baseline_parameters,
            train_loader,
            valid_loader,
            run_neptune,
            device,
        ),
    )

    for i in range(50):
        parameters, trial_index = ax_client.get_next_trial()
        # Local evaluation here can be replaced with deployment to external system.
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=train_evaluate(
                parameters, train_loader, valid_loader, run_neptune, device
            ),
        )

    return ax_client


if __name__ == "__main__":

    import neptune

    run_neptune = neptune.init_run(
        project="becavin-lab/ax-test",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1Zjg5NGJkNC00ZmRkLTQ2NjctODhmYy0zZDAzYzM5ZTgxOTAifQ==",
    )  # your credentials
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    with tf.device("GPU"):
        print("yo")
        ax_client = cnn_training(run_neptune, device)
        best_parameters, values = ax_client.get_best_parameters()
        print(best_parameters)

    # loop_api()

    run_neptune.stop()
