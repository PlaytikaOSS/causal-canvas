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
import sys

import yaml
from loguru import logger

# Default dataset locations for Playtika's internal environment. Override via
# CLI args when running outside that infra.
DEFAULT_SEGMENTS_PATH = (
    "/mnt/VAST_SHARED/lausanne_ds/player360/clustering"
    "/bb/player_360_bb_20230125/data/player_360_bb_20230125_"
)
DEFAULT_STRUCTURES_PATH = (
    "/mnt/VAST_SHARED/lausanne_ds/player360/experiment_results"
    "/bb/bb_bayesian_network/churn/14_features/"
)


def iterate_causal_inference(
    segments_path: str = DEFAULT_SEGMENTS_PATH,
    structures_path: str = DEFAULT_STRUCTURES_PATH,
):
    """
    Run causal inference on multiple segmented datasets across a grid of
    discretisation methods and arguments.

    Reads configuration parameters from ``docs/config.yaml`` and iterates
    through different discretisation methods. Results are saved in subfolders
    under ``config["output_path"]`` keyed by segment, method and parameter.

    Parameters
    ----------
    segments_path : str
        Filename prefix (not a directory) for the segment parquet files.
        The script globs ``{segments_path}cluster*`` to enumerate segments.
        Defaults to the Playtika-internal path.
    structures_path : str
        Root directory containing a ``segments/segment_{i}/causal_discovery/
        DAG_for_inference.dot`` tree. Defaults to the Playtika-internal path.
    """

    # Load configuration from the YAML file
    with open("docs/config.yaml") as f:
        config = yaml.safe_load(f)

    # Define discretization options for different methods
    discretiser = {"simple": [2, 3, 4], "tree": [2, 3, 4, 5], "mdlp": [5, 6, 10]}

    # Get paths to all segmented clusters
    all_cluster_paths = glob.glob(segments_path + "cluster*", recursive=True)

    output_path = config["output_path"]

    # Iterate through each cluster segment
    for i, cluster_path in enumerate(all_cluster_paths):
        config["structure_path"] = (
            structures_path
            + f"segments/segment_{i}/causal_discovery/DAG_for_inference.dot"
        )
        config["data_input_path"] = cluster_path
        output_path_seg = output_path + f"segments/segment_{i}/inference/mle/"

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
                subprocess.check_call(
                    [
                        sys.executable,
                        "infer_bayesian_network.py",
                        "-config_file",
                        config_path,
                    ]
                )


def main():
    import fire

    fire.Fire(iterate_causal_inference)


if __name__ == "__main__":
    main()
