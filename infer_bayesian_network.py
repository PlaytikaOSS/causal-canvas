from datetime import datetime

import fire
import pandas as pd
import yaml
from loguru import logger
from networkx.drawing import nx_pydot

from causal_canvas.bayesian_network_estimator import BayesianNetworkEstimator
from causal_canvas.preprocessor import Preprocessor
from causal_canvas.script_config import ScriptConfig
from causal_canvas.structure_learner import StructureLearner
from causal_canvas.utils import restructure_nx_object, save_structure


def infer_bayesian_network(config_file: str):
    """
    Parameters
    ----------
    config_file: str
        Path to the config file of the experiments that you want to do
    """

    logger.info("Config read")

    config = ScriptConfig.load_yaml(config_file)
    logger.info(config)

    logger.info("Reading data")

    if str(config.data_input_path).endswith("parquet"):
        data = pd.read_parquet(config.data_input_path)
    elif str(config.data_input_path).endswith("csv"):
        data = pd.read_csv(config.data_input_path)
    else:
        raise ValueError("Only CSV and Parquet files are supported.")

    today = datetime.now()
    output_path = config.output_path

    # Optionally add a date subfolder to output path
    if config.add_datetime_to_folder:
        output_path = output_path / f"{today.strftime('%Y%m%d_%H%M%S')}"

    # Create folder if it doesn't exists
    output_path.mkdir(parents=True, exist_ok=True)

    train_set = data

    # Filter train set dates if required
    if config.train_dates:
        data[config.date_column] = data[config.date_column].astype(str)
        train_set = data[
            data[config.date_column].between(
                left=str(config.train_dates.start), right=str(config.train_dates.end)
            )
        ]

    if config.drop_nans:
        train_set = train_set.dropna()

    logger.info("Reading data finished")
    logger.info("Splitting to validation and test sets")

    if not config.structure_path:
        preprocessor = Preprocessor(
            df=train_set,
            event_col=config.event_column,
            user_id_col=config.id_column,
            date_col=config.date_column,
            categorical_columns=config.categorical_features,
            drop_columns=config.drop_columns,
            features_select=config.features_select,
            sample_frac=config.sample_frac,
        )
        causal_discovery_path = output_path / "causal_discovery"
        causal_discovery_path.mkdir(exist_ok=True, parents=True)

        preprocessed_set = preprocessor.preprocess(artifacts_dir=causal_discovery_path)
        logger.info("Pre-processing finished")

        logger.info("Initiating Structure Learning")

        causal_discovery = StructureLearner(
            X=preprocessed_set,
            event_col=config.event_column,
            connections_type=config.connections_type,
            lasso_multiplier=config.lasso_multiplier,
            non_linear_args=config.non_linear_args,
            max_iter=config.max_iter,
            h_tol=config.h_tol,
            w_threshold=config.w_threshold,
            tabu_edges=config.tabu_edges,
            tabu_edge_features=config.tabu_edge_features,
            event_label=config.event_graph_label,
            event_color=config.event_color,
            higher_contribution_feature_color=config.higher_contribution_feature_color,
            invert_signs=config.invert_signs,
        )

        causal_discovery.discover_dag(path=causal_discovery_path)

        logger.info("Discovery completed")
        if causal_discovery.threshold_structural_model:
            discovered_graph = (
                causal_discovery.threshold_structural_model.get_largest_subgraph()
            )

            save_structure(
                discovered_graph,
                path=causal_discovery_path / "DAG_for_inference.dot",
            )
    else:
        discovered_graph = restructure_nx_object(
            nx_pydot.read_dot(config.structure_path)
        )

    if config.inference_method:
        logger.info("Using discovered graph as a Bayesian Network")
        bayesian_network_path = output_path / "bayesian_network"
        bayesian_network_path.mkdir(exist_ok=True)

        train_set_bn = train_set[config.features_select + [config.event_column]]

        numerical_features = [
            c for c in config.features_select if c not in config.categorical_features
        ]
        # Determine discretiser method
        # discretiser_method = getattr(config.discretiser, "method", None)
        # if discretiser_method not in ["simple", "tree", "mdlp"]:
        #     discretiser_method = "simple"  # or any other default value you prefer

        # Initialize BayesianNetworkEstimator with required fields
        # discretisation_cutoffs_target = (
        #     None
        #     if config.discretiser is None
        #     or "discretisation_cutoffs_target" not in config.discretiser
        #     else config.discretiser["discretisation_cutoffs_target"]
        # )
        feature_to_distribution_map = (
            None
            if config.non_linear_args is None
            or "feature_to_distribution_map" not in config.non_linear_args
            else config.non_linear_args["feature_to_distribution_map"]
        )
        bayesian_network = BayesianNetworkEstimator(
            structural_model=discovered_graph,
            train_set=train_set_bn,
            numerical_features=numerical_features,
            event_col=config.event_column,
            discretiser_method=config.discretiser.method,
            discretiser_argument=config.discretiser.argument,
            discretisation_cutoffs_target=config.discretiser.cutoffs_target,
            feature_to_distribution_map=feature_to_distribution_map,
            proportion_threshold=config.discretiser.proportion_threshold,
            max_categories=config.discretiser.max_categories,
            inference_method=config.inference_method,
            # split_dictionary={},  # Provide appropriate value for split_dictionary
            # model={},  # Provide appropriate value for model
        )

        logger.info("Fitting the bayesian network")
        bayesian_network.fit_bayesian_network(path=bayesian_network_path)

        if config.conditional_dependency_estimates:
            for node in config.conditional_dependency_estimates:
                logger.info(
                    f"Calculating conditional dependence probabilities for {node}"
                )
                bayesian_network.get_node_cpds(path=bayesian_network_path, node=node)

    logger.info("Writing config file")

    with (output_path / "config.yml").open("w") as fp:
        yaml.dump(dict(config), fp)


if __name__ == "__main__":
    fire.Fire(infer_bayesian_network)
