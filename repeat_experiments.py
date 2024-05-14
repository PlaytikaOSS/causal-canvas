"""Docstring for the iterate_causal_inference.py module.

This module accepts a yaml configuration file that follows
the practice of configuration file within
docs/config.yaml

The user needs to modify the configuration file
to allow different discretisation (or inference) methods
as well as the output_directory_path
so that all the iterative model training results are saved single folder where each
sub-folder is matched to a single method and parameter.
"""

import glob
import subprocess

import yaml
from loguru import logger


def iterate_causal_inference():
    """
    This function demonstrates the process of performing causal inference using
    various methods, such as simple discretization, trees, and MDLP, on multiple
    segmented datasets. The script is designed to be adaptable to different use cases.

    It reads configuration parameters from the 'docs/config.yaml' file and iterates
    through different discretization methods and their corresponding arguments.
    For each combination, causal inference is performed, and the results are saved
    in specific output paths based on the dataset segments.
    """

    # Load configuration from the YAML file
    with open("docs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    #  Define discretization options for different methods
    discretiser = {"simple": [2, 3, 4], "tree": [2, 3, 4, 5], "mdlp": [5, 6, 10]}

    # Define the path to the segmented datasets
    segments_path = (
        "/mnt/VAST_SHARED/lausanne_ds/player360/clustering"
        "/bb/player_360_bb_20230125/data/player_360_bb_20230125_"
    )

    # Get paths to all segmented clusters
    all_cluster_paths = glob.glob(segments_path + "cluster*", recursive=True)
    # all_cluster_paths = ["/mnt/VAST_SHARED/lausanne_ds/player360/data/player_360_bb_20230125.parquet"]
    # all_cluster_paths = ["/mnt/VAST_SHARED/lausanne_ds/player360/data/player_360_20230125_non_zero_flags.parquet"]
    # all_cluster_paths = ["/mnt/VAST_SHARED/lausanne_ds/player360/data/player_360_20230125_non_zero_flags_only_vips.parquet"]

    output_path = config["output_path"]

    structures_path = "/mnt/VAST_SHARED/lausanne_ds/player360/experiment_results/bb/bb_bayesian_network/churn/14_features/"
    # structures_path = "/mnt/VAST_SHARED/lausanne_ds/player360/experiment_results/bb/bb_bayesian_network/sink/vips/discovered_graphs/20230824/causal_discovery/DAG_for_inference.dot"

    # Iterate through each cluster segment
    for i, cluster_path in enumerate(all_cluster_paths):
        config["structure_path"] = (
            structures_path
            + f"segments/segment_{i}/causal_discovery/DAG_for_inference.dot"
        )
        # config["structure_path"] = structures_path #+ f"all/causal_discovery/DAG_for_inference.dot"
        config["data_input_path"] = cluster_path
        output_path_seg = output_path + f"segments/segment_{i}/inference/mle/"
        # output_path_seg = output_path + f"BDeu_5K/"

        # Iterate through discretisation methods and arguments
        for k, v in discretiser.items():
            for value in v:
                config["discretiser"].update(
                    {"method": k, "argument": value, "proportion_threshold": 0.08}
                )
                output_sub_path = f"{k}_{str(value).replace('.', '_')}/"
                logger.info(
                    f"Preparing causal inference with discretiser method {k} and argument {value}"
                )
                config["output_path"] = output_path_seg + output_sub_path

                logger.info(config["output_path"])
                config_path = "docs/config_mle_seg.yaml"
                with open(config_path, "w") as outfile:
                    yaml.dump(config, outfile, default_flow_style=False)

                # Run the Bayesian network inference script
                subprocess.call(
                    f"python3 infer_bayesian_network.py -config_file {config_path}",
                    shell=True,
                )


if __name__ == "__main__":
    iterate_causal_inference()
