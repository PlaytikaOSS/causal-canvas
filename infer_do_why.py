import json
import pickle
from datetime import datetime

import fire
import pandas as pd
import yaml
from causalnex.inference import InferenceEngine
from loguru import logger

from causal_canvas.bayesian_network_estimator import BayesianNetworkEstimator
from causal_canvas.inference_utils import (
    compute_counterfactuals,
    compute_effect,
    compute_shift_of_probas,
    compute_uplift,
    convert_dict_for_json,
    get_all_combinations,
    map_conditionals_to_actuals,
    map_splits,
    plot_ATEs,
    plot_counterfactuals_or_shifts,
    plot_uplifts,
)
from causal_canvas.script_config import ScriptConfigInference


def infer(config_file: str):
    """
    Perform inference based on the provided configuration file.

    Parameters
    ----------
    config_file : str
        Path to the configuration file for the inference.
    """
    # Step 1: Read and load the configuration from the YAML file
    logger.info("Config read")

    config = ScriptConfigInference.load_yaml(config_file)
    logger.info(config)

    # Step 2: Load the Bayesian Network model for inference
    logger.info("Reading Bayesian Network for inference")
    with config.model_path.open("rb") as fp:
        model: BayesianNetworkEstimator = pickle.load(fp)

    if model.model is None:
        raise ValueError(
            f"Model stored in {config.model_path} has no .model attribute defined."
        )

    # Create an `InferenceEngine` to query marginals and make interventions
    ie = InferenceEngine(model.model)

    # Step 3: Create a directory for output files based on the current timestamp
    today = datetime.now()

    output_path = config.output_path / f"{today.strftime('%Y%m%d_%H%M%S')}"

    counterfactuals_dir = output_path / "counterfactuals"
    counterfactuals_dir.mkdir(parents=True, exist_ok=True)

    interventions_dir = output_path / "interventions"
    interventions_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: Save the computed marginals to a JSON file
    logger.info("Saving marginals")
    marginals = ie.query()
    marginals_mapped_dictionary = map_splits(model, marginals)

    with (output_path / "marginals.json").open("w") as outfile:
        json.dump(convert_dict_for_json(marginals_mapped_dictionary), outfile)

    # Step 5: Save conditional marginals to CSV files, if specified in the configuration
    logger.info("Saving conditional marginals")
    # Extracting conditional marginals
    conditional_marginal_combinations = get_all_combinations(
        inference_engine=ie, cond_marginals=config.conditionals, marginals=marginals
    )
    for key in conditional_marginal_combinations.keys():
        mapped_conditionals = map_conditionals_to_actuals(
            model=model,
            cond_marginals=conditional_marginal_combinations,
            target=key,
        )

        pd.DataFrame(mapped_conditionals[key]).T.to_csv(
            output_path / f"conditionals_{key}.csv"
        )

    # Step 6: Perform interventions and compute shifts, ATEs, and uplifts for each feature
    if config.interventions:
        # Fetch number of subjects used for inference
        N = model.train_set.shape[0]
        for intervention in config.interventions:
            logger.info(f"Calculating intervention strategy for {intervention.feature}")
            shifts = compute_shift_of_probas(
                inference_engine=ie,
                model=model,
                intervention=intervention,
                target=config.event_column,
            )
            # Fetch new
            plot_counterfactuals_or_shifts(
                cf=shifts,
                feature_name=intervention.feature,
                target_name=config.event_column,
                path=interventions_dir,
                counterfactuals=False,
            )

            # Fetch updated feature marginals
            marginals_updated = ie.query()
            marginals_updated_mapped_dictionary = map_splits(model, marginals_updated)

            ates = compute_effect(
                cf=shifts,
                intervention_marginals=marginals_updated_mapped_dictionary[
                    intervention.feature
                ],
                control_marginals=marginals_mapped_dictionary[intervention.feature],
                target_class=config.target_class,
                N=N,
                alpha=0.05,
            )

            plot_ATEs(
                cf=ates,
                feature_name=intervention.feature,
                target_name=config.event_column,
                path=interventions_dir,
                counterfactuals=False,
            )

            uplifts = compute_uplift(shifts, target_class=config.target_class)

            plot_uplifts(
                cf=uplifts,
                feature_name=intervention.feature,
                target_name=config.event_column,
                path=interventions_dir,
                counterfactuals=False,
            )

            shifts.to_csv(interventions_dir / f"shifts_{intervention.feature}.csv")
            ates.to_csv(interventions_dir / f"strategy_ATEs_{intervention.feature}.csv")
            uplifts.to_csv(
                interventions_dir / f"strategy_uplifts_{intervention.feature}.csv"
            )

            ie.reset_do(intervention.feature)
            ie.reset_do(config.event_column)

    # Step 7: Calculate and plot counterfactuals, ATEs, and uplifts for specified features
    N = model.train_set.shape[0]
    for feature in config.counterfactuals:
        logger.info(f"Calculating counterfactuals strategy for {feature}")
        cf = compute_counterfactuals(
            inference_engine=ie,
            model=model,
            feature=feature,
            target=config.event_column,
        )

        plot_counterfactuals_or_shifts(
            cf=cf,
            feature_name=feature,
            target_name=config.event_column,
            path=counterfactuals_dir,
            counterfactuals=True,
        )

        marginals_updated = ie.query(parallel=True)
        marginals_updated_mapped_dictionary = map_splits(model, marginals_updated)

        ates = compute_effect(
            cf=cf,
            intervention_marginals=marginals_updated_mapped_dictionary[feature],
            control_marginals=marginals_mapped_dictionary[feature],
            target_class=config.target_class,
            N=N,
            alpha=0.05,
        )

        plot_ATEs(
            cf=ates,
            feature_name=feature,
            target_name=config.event_column,
            path=counterfactuals_dir,
            counterfactuals=True,
        )

        uplifts = compute_uplift(cf, target_class=config.target_class)
        uplifts.to_csv(counterfactuals_dir / f"counterfactuals_uplifts_{feature}.csv")
        plot_uplifts(
            cf=uplifts,
            feature_name=feature,
            target_name=config.event_column,
            path=counterfactuals_dir,
            counterfactuals=True,
        )

        cf.to_csv(counterfactuals_dir / f"counterfactuals_{feature}.csv")
        ates.to_csv(counterfactuals_dir / f"counterfactuals_ATEs_{feature}.csv")
        uplifts.to_csv(counterfactuals_dir / f"counterfactuals_uplifts_{feature}.csv")

        ie.reset_do(feature)
        ie.reset_do(config.event_column)

    # Step 8: Write the final configuration to a YAML file
    logger.info("Writing config file")
    with (output_path / "config.yml").open("w") as outfile:
        yaml.dump(dict(config), outfile)


if __name__ == "__main__":
    fire.Fire(infer)
