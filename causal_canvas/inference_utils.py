from itertools import product
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import plotnine.ggplot as ggplot
import seaborn as sns
from causalnex.inference import InferenceEngine
from loguru import logger
from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
from plotnine import (
    aes,
    element_text,
    facet_grid,
    geom_errorbar,
    geom_line,
    geom_point,
    labs,
    position_dodge,
    theme,
    theme_bw,
)
from scipy.stats import norm

from causal_canvas.bayesian_network_estimator import BayesianNetworkEstimator
from causal_canvas.script_config import Conditional, Intervention


def map_splits(model, marginals):
    """
    Maps bins of features to translatable
    intervals.

    Parameters
    ----------
    model: BayesianNetwork class
    marginals: dict
        dictionary of marginals coming from InferenceEngine

    Returns
    -------
        dict
    """
    updated_dictionaries = dict()
    for feature in marginals.keys():
        if feature in model.split_dictionary.keys():
            mapping = model._get_single_feature_map(feature=feature)
            updated_dictionaries[feature] = change_keys(mapping, marginals[feature])
        else:
            updated_dictionaries[feature] = marginals[feature]
    return updated_dictionaries


def change_keys(mapping, dictionary):
    """
    Maps dictionary keys to specified mapping

    Parameters
    ----------
    mapping: dict
        dictionary with mappings
    dictionary: dict
        dictionary to change the keys

    Returns
    -------
        dict
    """
    return {mapping.get(key, key): value for key, value in dictionary.items()}


def convert_dict_for_json(dictionary):
    """
    Recursively converts a dictionary containing NumPy data types to a JSON-compatible format.

    This function traverses the input dictionary recursively and converts NumPy data types
    (such as np.float64, np.int64, and np.bool_) to their corresponding native Python types
    (float, int, and bool) to ensure compatibility with JSON serialization.

    Parameters
    ----------
    dictionary : dict
        The input dictionary to be converted.

    Returns
    -------
    dict
        A new dictionary with NumPy data types converted to native Python types.
    """
    converted_dict = {}
    for key, value in dictionary.items():
        if isinstance(key, (np.float64, np.int64, np.bool_, np.uint64)):
            key = float(key)
        if isinstance(value, (np.float64, np.int64, np.bool_, np.uint64)):
            value = float(value)
        elif isinstance(value, dict):
            value = convert_dict_for_json(value)
        converted_dict[key] = value
    return converted_dict


def query_marginals(inference_engine, conditions, target):
    """
    Queries marginal distributions, i.e. P(X=x) and
    conditional marginal distributions, i.e. P(Y=y|X=x, Z=z, ..)

    Parameters
    ----------
    inference_engine: InferenceEngine
        The inference engine used for querying conditional marginals.
    conditions: list
        list of conditional features
    target: str
        target name to extract the marginal

    Returns
    -------
        dict
    """
    return inference_engine.query(conditions, parallel=True)[target]


def get_all_combinations(
    inference_engine: InferenceEngine,
    cond_marginals: list[Conditional],
    marginals: Union[dict, list[dict]],
):
    """
    Calculate and retrieve conditional marginal distributions for various combinations of conditions.

    Parameters
    ----------
    inference_engine : InferenceEngine
        The inference engine used for querying conditional marginals.
    cond_marginals : list[Conditional]
        A list of Conditional.
    marginals : dict
        A dictionary containing marginal distributions for individual variables.

    Returns
    -------
    dict
        A nested dictionary containing calculated conditional marginal values for different combinations
        of conditions, organized by target variables and their corresponding condition sets.
    """
    # combinations = {
    #     key: {
    #         tuple(conditions_dict.items()): query_marginals(
    #             inference_engine, conditions_dict, key
    #         )
    #         for conditions in conditions_list
    #         for conditions_dict in (
    #             dict(zip(conditions, combination))
    #             for combination in product(
    #                 *(marginals[cond_key].keys() for cond_key in conditions)
    #             )
    #         )
    #     }
    #     for key, conditions_list in cond_marginals.items()
    # }

    # TODO: refactor and test this very complex piece of code. The original is above
    combinations = {
        c.feature: {
            tuple(conditions_dict.items()): query_marginals(
                inference_engine=inference_engine,
                conditions=conditions_dict,
                target=c.feature,
            )
            for conditions in c.conditional_features
            for conditions_dict in (
                dict(zip(conditions, combination))
                for combination in product(
                    *(marginals[cond_key].keys() for cond_key in conditions)
                )
            )
        }
        for c in cond_marginals
    }

    return combinations


def map_tupled_values_to_actual(mapping, data):
    """
    Map tupled values in a nested dictionary to their corresponding actual values based on a provided mapping.

    Parameters
    ----------
    mapping : dict
        A nested dictionary where keys represent original values and values represent their corresponding actual values.
    data : dict
        A nested dictionary containing data with tupled keys to be mapped.

    Returns
    -------
    dict
        A new nested dictionary with tupled keys replaced by their mapped actual values, based on the provided mapping.
    """

    def map_tuple_key(tuple_key):
        return tuple(
            (key, mapping.get(key, {}).get(value, value)) for key, value in tuple_key
        )

    return {
        key: {
            map_tuple_key(tuple_key): inner_inner_dict
            for tuple_key, inner_inner_dict in inner_dict.items()
        }
        for key, inner_dict in data.items()
    }


def map_conditionals_to_actuals(model, cond_marginals, target):
    """
    Map conditional feature values to their corresponding actual values based on a provided model's mapping.

    Parameters
    ----------
    model : BayesianNetwork Class
        The model that was trained and contains feature mappings.
    cond_marginals : dict
        A dictionary where keys represent target variables and values are lists of condition sets.
    target : str
        The target variable for which conditional marginals are calculated.

    Returns
    -------
    dict
        A new dictionary with conditional marginals, where conditional feature values are replaced
        by their corresponding actual values according to the provided model's mapping.
    """

    features_for_cond = [item for sublist in cond_marginals[target] for item in sublist]

    mapping_dict = dict()
    for feat in features_for_cond:
        if feat in model.split_dictionary.keys():
            mapping_dict[feat] = model._get_single_feature_map(feature=feat)

    if target in model.split_dictionary.keys():
        mapping_dict[target] = model._get_single_feature_map(feature=target)

    # Map the values in the given dictionary to the actual mapping
    return map_tupled_values_to_actual(mapping_dict, cond_marginals)


def compute_counterfactuals(inference_engine, model, feature, target):
    """
    Compute counterfactual distributions by intervening on a specified feature and observing the changes
    in the target distribution.

    Parameters
    ----------
    inference_engine : InferenceEngine
        The inference engine used for querying distributions.
    model : BayesianNetwork class
        The model class containing feature mappings.
    feature : str
        The feature on which to intervene.
    target : str
        The target variable for which counterfactual distributions are computed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing counterfactual distributions for the target variable after intervening on the feature.
    """

    dfs = []
    no_intervention = pd.Series(inference_engine.query()[target]).rename("control")
    dfs.append(no_intervention)
    levels = inference_engine.query(parallel=True)[feature]
    if feature in model.split_dictionary.keys():
        feature_mapping = model._get_single_feature_map(feature=feature)
        feature_mapping["control"] = "control"

    for key, _ in levels.items():
        new_level_vals = {k: 1 if k == key else 0 for k in levels}
        inference_engine.do_intervention(feature, new_level_vals)
        dfs.append(pd.Series(inference_engine.query()[target]))

    cf = pd.concat(dfs, axis=1)
    if feature in model.split_dictionary.keys():
        cf.rename(columns=feature_mapping, inplace=True)

    if target in model.split_dictionary.keys():
        cf.rename(model._get_single_feature_map(feature=target), inplace=True)
    return cf


def compute_shift_of_probas(
    inference_engine: InferenceEngine,
    model: BayesianNetworkEstimator,
    intervention: Intervention,
    target: str,
):
    """
    Compute the shift in probabilities of a target variable by intervening on
    a feature with specified probability shifts.

    Parameters
    ----------
    inference_engine : InferenceEngine
        The inference engine used for querying distributions.
    model : BayesianNetworkEstimator
    intervention : Intervention
        A class defining the feature on which to intervene, with a dictionary where keys
        represent labels for the probability shifts and values are dictionaries of
        probability shifts.
    target : str
        The target variable for which probability shifts are computed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing shifted probability distributions for the target variable after interventions.
    """

    dfs = []
    no_intervention = pd.Series(inference_engine.query()[target]).rename("control")
    dfs.append(no_intervention)
    labels = ["low", "high"]
    for label in labels:
        logger.info(f"{intervention.feature}, {label}, {getattr(intervention, label)}")
        inference_engine.do_intervention(
            intervention.feature, getattr(intervention, label).dict()
        )
        dfs.append(pd.Series(inference_engine.query(parallel=True)[target]))

    shifts = pd.concat(dfs, axis=1)
    shifts.columns = ["control"] + [
        f"{intervention.feature}_{label}" for label in list(labels)
    ]
    if target in model.split_dictionary.keys():
        shifts.rename(model._get_single_feature_map(feature=target), inplace=True)
    return shifts


def plot_counterfactuals_or_shifts(
    cf: pd.DataFrame,
    feature_name: str,
    target_name: str,
    path: Path,
    counterfactuals: bool = False,
):
    """
    Plot counterfactual or shift distributions of a target variable given different levels of a feature.

    Parameters
    ----------
    cf : pandas.DataFrame
        A DataFrame containing counterfactual or shift distributions of the target variable.
    feature_name : str
        The name of the feature being intervened on.
    target_name : str
        The name of the target variable.
    path: Path
        The name of path to save the plot
    counterfactuals : bool, optional
        If True, the plot title includes 'by counterfactuals of'. If False, the plot title includes
        'intervention strategies of'.

    """

    plt.figure(figsize=(12, 5), dpi=120)
    for col in cf.columns:
        plt.plot(cf.index, cf[col], "o--", label=f"{feature_name} = {col}")

    plt.legend()
    if counterfactuals:
        plt.title(f"Distribution of {target_name} by counterfactuals of {feature_name}")
        path_save = "counterfactuals"
    else:
        plt.title(
            f"Distribution of {target_name} given interventional strategy of {feature_name}"
        )
        path_save = "intervention"
    plt.xlabel(target_name)
    plt.ylabel("Probability")

    plt.savefig(path / f"{path_save}_plot_{feature_name}.png", bbox_inches="tight")
    plt.clf()


def compute_effect(
    cf, intervention_marginals, control_marginals, target_class, N, alpha=0.05
):
    """
    Calculate the Average Treatment Effect (ATE) for each level of a feature distribution.

    ATE estimands considered are:
    - Risk Difference (RD): Difference in probabilities of a positive outcome
    between treatment and control or actual (control) and counterfactual.
    - Relative Risk (RR): Ratio of probabilities of a positive outcome
    between treatment and control or actual and counterfactual.
    - Odds Ratio (OR): Ratio of odds of a positive outcome between
    treatment and control or actual (control) and counterfactual.

    Parameters
    ----------
    cf : pandas.DataFrame
        A DataFrame containing counterfactual distributions for different levels.
    intervention_marginals : dict
        A dictionary containing the marginal probabilities for the intervention scenario.
    control_marginals : dict
        A dictionary containing the marginal probabilities for the control scenario.
    target_class : str
        The target class considered as the positive outcome.
    N : int
        Total number of samples.
    alpha : float, optional
        Significance level for confidence intervals, by default 0.05.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with calculated effects for each level and ATE estimand.
    """
    list_dfs = []
    control_counts = (
        pd.DataFrame.from_dict(control_marginals, orient="index").rename(
            columns={0: "counts"}
        )
        * N
    )
    intervention_counts = (
        pd.DataFrame.from_dict(intervention_marginals, orient="index").rename(
            columns={0: "counts"}
        )
        * N
    )
    # in case we use counterfactual we need to replace 0 for the sake of calculations
    intervention_counts = intervention_counts.replace(0, 0.1)

    for method in ["RD", "RR", "OR"]:
        effects = {}  # Dictionary to store effects for each level

        for level in cf.columns:
            indx = cf.index == target_class
            if method != "OR":
                expected_no = (indx * cf["control"]).sum()
                expected_yes = (indx * (cf[level])).sum()
                if method == "RD":
                    mean = expected_yes - expected_no
                    se = np.sqrt((cf["control"].prod() / N) + (cf[level].prod() / N))
                    lower = mean + norm.ppf(alpha / 2) * se
                    upper = mean + norm.ppf(1 - alpha / 2) * se
                else:
                    error = 1 / (expected_no * N) + 1 / (expected_yes * N) - 2 / N
                    mean = expected_yes / expected_no
                    lower = np.exp(np.log(mean) + norm.ppf(alpha / 2) * np.sqrt(error))
                    upper = np.exp(
                        np.log(mean) + norm.ppf(1 - alpha / 2) * np.sqrt(error)
                    )

            else:
                expected_no = (indx * cf["control"]).sum() / (
                    1 - (indx * cf["control"]).sum()
                )
                expected_yes = (indx * (cf[level])).sum() / (
                    1 - (indx * (cf[level])).sum()
                )
                mean = expected_yes / expected_no
                error = (1 / control_counts).sum() + (1 / intervention_counts).sum()
                lower = np.exp(np.log(mean) + norm.ppf(alpha / 2) * np.sqrt(error[0]))
                upper = np.exp(
                    np.log(mean) + norm.ppf(1 - alpha / 2) * np.sqrt(error[0])
                )

            effects[level] = {
                "Estimand": method,
                "ATE": mean,
                f"{100 * alpha / 2}%CI": lower,
                f"{100 * (1 - 0.5 * alpha)}%CI": upper,
            }

        list_dfs.append(pd.DataFrame.from_dict(effects, orient="columns").T)

    # Concatenate the effects for each method into a single DataFrame
    dfs = pd.concat(list_dfs).reset_index().rename(columns={"index": "bin"})
    dfs["ATE"] = dfs["ATE"].astype(float)
    dfs["2.5%CI"] = dfs["2.5%CI"].astype(float)
    dfs["97.5%CI"] = dfs["97.5%CI"].astype(float)
    return dfs


def create_custom_categorical_order(order_list):
    return CategoricalDtype(categories=order_list, ordered=True)


def plot_ATEs(cf, feature_name, target_name, path, counterfactuals=False):
    """
    Plot the Average Treatment Effects (ATEs) for different counterfactual bins.

    Parameters
    ----------
    cf : pandas.DataFrame
        A DataFrame containing counterfactual distributions and ATE values.
    feature_name : str
        The name of the feature being intervened on.
    target_name : str
        The name of the target variable.
    path : str, optional
        The path to save the plot image file. If None, the plot will be displayed but not saved.
    counterfactuals : bool, optional
        If True, the plot title includes 'on counterfactuals'. If False, the plot title includes
        'Interventional Strategy of'.
    """
    if counterfactuals:
        title = f"Average treatment effect of {target_name} on counterfactuals"
        x_lab = f"Counterfactual Bins of {feature_name}"
        path_save = "counterfactuals"
        cf = cf[(cf.bin != "control") & (cf.Estimand != "OR")]
        ordered_bins = create_custom_categorical_order(cf.bin.unique())
        cf["bin"] = cf.bin.astype(ordered_bins)
    else:
        title = f"Average treatment effect of {target_name} on interventions"
        x_lab = f"Interventional Strategy of {feature_name}"
        path_save = "intervention"
        cf = cf[cf.bin != "control"]

    fig = (
        ggplot(cf, aes(x="bin", y="ATE", colour="bin"))
        + geom_line()
        + geom_point()
        + facet_grid("~Estimand")
        + theme_bw()
        + geom_errorbar(
            aes(x="bin", ymin="2.5%CI", ymax="97.5%CI"),
            size=0.5,
            width=0.25,
            position=position_dodge(0.9),
        )
        + facet_grid("Estimand~", scales="free")
        + labs(title=title, x=x_lab)
        + theme(axis_text_x=element_text(angle=90, hjust=1))
    ).draw(show=True)
    if path:
        fig.savefig(
            f"{path}/{path_save}_ate_plot_{feature_name}.png",
            bbox_inches="tight",
            dpi=150,
        )
    fig.clf()


def compute_uplift(cf, target_class):
    """
    Calculate the uplift for each level of feature X or a counterfactual distribution.

    For level: Uplift = Average Outcome (Y) for Specific X Level - Overall Average Outcome (Y)
    For Counterfactual: Uplift = Observed Outcome (Y) – Counterfactual Outcome (Y)

    Parameters
    ----------
    cf : pandas.DataFrame
        A DataFrame containing counterfactual distributions for different levels.
    target_class : str
        The class label considered as the target class.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with calculated uplifts for each level.
    """
    uplifts = {}  # Dictionary to store uplifts for each level

    # Iterate over each level in the DataFrame
    for level in cf.columns[1:]:
        # Calculate uplift for the current level and target class
        uplift = cf["control"][target_class] - cf[level][target_class]
        uplifts[level] = uplift

    # Create a DataFrame from the uplifts dictionary
    uplifts_df = (
        pd.DataFrame.from_dict(uplifts, orient="index")
        .reset_index()
        .rename(columns={"index": "bin", 0: "uplift"})
    )

    return uplifts_df


def plot_uplifts(cf, feature_name, target_name, path, counterfactuals=False):
    """
    Plot the uplift in probabilities for a target variable given different levels of a feature.

    Parameters
    ----------
    cf : pandas.DataFrame
        A DataFrame containing uplift values for different levels of the feature.
    feature_name : str
        The name of the feature being intervened on.
    target_name : str
        The name of the target variable.
    path : str, optional
        The path to save the plot image file. If None, the plot will be displayed but not saved.
    counterfactuals : bool, optional
        If True, the plot title and labels indicate 'by counterfactuals'.
        If False, they indicate 'by interventional strategy'.
    """

    sns.barplot(
        data=cf,
        x="uplift",
        y="bin",
        linewidth=3,
        edgecolor=".5",
        orient="h",
        palette="Set2_r",
    )

    if counterfactuals:
        plt.title(f"Uplift  of {target_name} by counterfactuals")
        plt.ylabel(f"Counterfactual Bins of {feature_name}")
        path_save = "counterfactuals"
    else:
        plt.title(f"Uplift of {target_name} by interventional strategy")
        plt.ylabel(f"Interventional Strategy of {feature_name}")
        path_save = "intervention"
    plt.show()

    if path:
        plt.savefig(
            f"{path}/{path_save}_uplift_plot_{feature_name}.png", bbox_inches="tight"
        )
    plt.clf()
