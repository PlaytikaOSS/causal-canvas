"""Docstring for the iterate_structure_learn.py module.

This module accepts a yaml configuration file that follows
the practice of configuration file within
docs/config.yaml

The user needs to modify the configuration file
to allow different lasso, ridge parameters, hidden layers or neurons.
"""

import itertools
import subprocess

import yaml
from loguru import logger


def iterate_structure_learning():
    """
    This function demonstrates the process of performing causal discover using
    the Non linear NOTEARS-MLP by assuming GLM relationships between features.
    The script is designed to be adaptable to different use cases.

    It reads configuration parameters from the 'docs/config.yaml' file and iterates
    through different lasso, ridge and hidden layer arguments.
    For each combination, causal discovery is performed, and the results are saved
    in the respective output paths.
    """

    # Load configuration from the YAML file
    with open("docs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #  Define discretization options for different methods

    # Define the parameter values
    lasso_values = [0.1, 0.01, 0.001, 0.0001]
    ridge_values = [0.1, 0.01, 0.001, 0.0001]
    layer_configs = [
        [2],
        [10],
        [2, 2, 2],
        [10, 10],
    ]

    # Generate all combinations of parameters
    param_combinations = list(
        itertools.product(lasso_values, ridge_values, layer_configs)
    )

    # Iterate through discretisation methods and arguments
    for params in param_combinations:
        lasso, ridge, layers = params
        config["lasso_multiplier"] = lasso
        config["non_linear_args"]["ridge_multiplier"] = ridge
        config["non_linear_args"]["hidden_layer_units"] = layers
        logger.info(
            f"Preparing causal discovery with discretiser lasso={lasso}, ridge={ridge}, layers={layers}"
        )
        config_path = "docs/config_non_linear.yaml"
        with open(config_path, "w") as outfile:
            yaml.dump(config, outfile, default_flow_style=False)

            # Run the Causal Discovery script
            subprocess.call(
                f"python3 infer_bayesian_network.py -config_file {config_path}",
                shell=True,
            )


if __name__ == "__main__":
    iterate_structure_learning()
