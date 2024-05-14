import json
import multiprocessing as mp
import os
import pickle
from multiprocessing import Pool
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pydantic import confloat, conint
from sklearn.metrics import classification_report


def plot_style(figsize=(12, 6), labelsize=20, titlesize=24, ticklabelsize=14, **kwargs):
    basic_style = {
        "figure.figsize": figsize,
        "axes.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "xtick.labelsize": ticklabelsize,
        "ytick.labelsize": ticklabelsize,
        "axes.spines.top": False,
        "axes.spines.bottom": True,
        "axes.spines.right": False,
        "axes.spines.left": False,
    }
    basic_style.update(kwargs)
    return plt.rc_context(rc=basic_style)


def round_interval(interval_str: str) -> str:
    """
    This function accepts a string interval, i.e. (4.7654, 8.767] or
    and returns the bounds of the interval separately as float numbers
    rounded to 3 decimal digits.

    Parameters
    ----------
    interval_str: str
        interval as a string

    Returns
    -------
        float, float
    """
    bounds = interval_str.strip("()[]").split(", ")
    rounded_bounds = [f"{float(bound):.3f}" for bound in bounds]
    return f"[{rounded_bounds[0]}, {rounded_bounds[1]})"


def plot_conditionals(
    cpds_path: str,
    target_column_names: str,
    target_class: str,
    threshold: confloat(ge=0, le=1),
    inference_method: tuple,
    target: str,
    path: str = None,
    sub_path: str = "",
):
    """
    Plot conditional probability distributions (CPDs) for specific features.

    This function reads CPDs estimated by a Bayesian network from a specified path and plots
    the CPDs of the features for both target and non-target classes. The CPDs are visualized as
    scatter plots, with the x-axis representing different intervals or values of the feature,
    and the y-axis representing the probability P(target|feature=i, X).

    Parameters
    ----------
    cpds_path : str
        Path to the CPDs estimated by a Bayesian network.
    threshold : float
        Threshold value for the target probability (P(target) > threshold).
    inference_method : tuple
        Inference method used (e.g., ("mle", "tree_2")).
    target : str
        Name of the target variable for labeling.
    path : str, optional
        Path to save the plot (if provided).

    Note
    ----
    - The input CPDs DataFrame is expected to have columns: 'True', 'False', and feature names.
    - The 'True' column represents the conditional probabilities for the target class.
    - The 'False' column represents the conditional probabilities for the non-target class.
    - The function uses matplotlib for plotting.
    """
    # Produce CPD plots
    cpds = pd.read_csv(cpds_path).drop("", axis=1)
    feats = [c for c in cpds.columns if c not in target_column_names]

    n_rows = 2
    n_cols = int(len(feats) / 2 if len(feats) % 2 == 0 else round(len(feats) / 2) + 1)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 15), sharey=True)

    for col, ax in zip(feats, axs.ravel()):
        if cpds[col].dtype != "int64":
            cpds["order_val"] = (
                cpds[col]
                .astype(str)
                .str.extract(r"(-?inf|\d+\.\d+|\d+)")
                .replace({"-inf": -1})
                .astype(float)
            )
            # Apply the rounding function to the DataFrame column
            cpds["rounded_intervals"] = cpds[col].apply(round_interval)
        else:
            cpds["order_val"] = cpds[col].astype(int)

        cpds = cpds.sort_values("order_val")

        event = cpds[cpds[target_class] > threshold].sort_values("order_val")
        non_event = cpds[cpds[target_class] < threshold].sort_values("order_val")

        with plot_style():
            ax.scatter(
                event["order_val"].astype(str),
                event[target_class],
                color="red",
                linewidth=2,
            )
            ax.scatter(
                non_event["order_val"].astype(str),
                non_event[target_class],
                color="green",
                linewidth=2,
            )
            ax.set_ylim((0, 1))
            ax.grid(True)
            if cpds[col].dtype != "int64":
                ax.set_xticks(
                    ticks=cpds["order_val"].astype(str).unique(),
                    labels=cpds["rounded_intervals"].unique(),
                    rotation=90,
                )
            ax.set_ylabel(f"P({target}|{col}=i, X)")
            ax.axhline(threshold, color="black")
            ax.set_title(col, {"fontsize": 10})
    fig.suptitle(
        f"CPDs of one feature | model {inference_method[0]} with {inference_method[1]}"
    )
    fig.tight_layout()
    fig.show()
    if path:
        fig.savefig(f"{path}/cpds_{sub_path}.png", bbox_inches="tight")
        fig.clf()


def map_splits(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts a Bayesian network model class and a dataframe
    and maps the appropriate numerical features bins to
    their accosiate intervals as decided by the discretiser

    Parameters
    ----------
    model: BayesianNetworkEstimator class
    df: pd.DataFrame

    Returns
    -------
        pd.DataFrame
    """
    for col in model.split_dictionary.keys():
        if col in df.columns:
            mapping = model._get_single_feature_map(feature=col)
            df[col] = df[col].map(mapping)
    return df


def extract_all_models(
    all_models_path: str,
    model_path: str,
    inference_methods: List,
    models_combinations: Dict,
) -> [List, List]:
    models_list = []
    model_names_list = []

    for method in inference_methods:
        path = f"{all_models_path}{method}/"
        for k, v in models_combinations.items():
            for val in v:
                try:
                    file = open(path + f"{k}_{val}" + model_path, "rb")
                    model_names_list.append((method, f"{k}_{val}"))
                    models_list.append(pickle.load(file))
                    pass
                except FileNotFoundError:
                    logger.info(
                        f"Bayesian Network file does not exist for {k}, {val}. Skipping this iteration."
                    )
                    continue  # Skip this iteration and move to the next model
    return models_list, model_names_list


def predict_samples(
    df: pd.DataFrame,
    model: Any,
    feats: List[str],
    target_name: str,
    target_class: str,
    target_column_names_cpds: list,
    discretisation_method: str,
    discretisation_path: str,
) -> [pd.DataFrame, pd.DataFrame]:
    """
    Run Bayesian network inference on a subset of data and extract user-level predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Subsample of a larger DataFrame.
    model : BayesianNetworkEstimator
        Bayesian network model.
    feats : list
        List of feature names to be used.
    target_name : str
        Name of the target variable.
    discretisation_method : str
        Discretisation method used. It can be between `simple`, `tree` or `mdlp`.
    discretisation_path : str
        Path to discretisation model if method
        is `tree` or `mdlp` or discretisation quantile splits if method is `simple`.

    Returns
    -------
    transformed_feats : pd.DataFrame
        Transformed and interpretable intervals of numerical features.
    predictions : pd.DataFrame
        User-level predictions.

    """

    disc = (
        feats + [target_name] if target_name in model.split_dictionary.keys() else feats
    )

    if discretisation_method in ["tree", "mdlp"] and not model.proportion_threshold:
        new_df = model.discretiser.transform(df[feats])
        splitter_path = discretisation_path
    elif discretisation_method in ["tree", "mdlp"] and model.proportion_threshold:
        new_df = model._known_splits_discretise(
            x=df[disc], splits=model.condenced_thresholds
        )
        splitter_path = None
    else:
        with open(discretisation_path, "r") as f:
            splits = json.load(f)
        splitter_path = discretisation_path
        new_df = model._known_splits_discretise(x=df[disc], splits=splits)

    transformed_feats = map_splits(model, new_df)
    predictions = model.predict_proba(
        X=df[feats + [target_name]],
        node=target_name,
        spliter_path=splitter_path,
        method=discretisation_method,
    )
    predictions.columns = target_column_names_cpds
    predictions[f"{target_name}_pred"] = predictions[f"{target_class}"] > 0.5
    predictions[f"{target_name}_real"] = np.where(
        target_name in model.split_dictionary.keys(),
        new_df[f"{target_name}"] == target_class,
        df[f"{target_name}"] == target_class,
    )

    predictions[f"{target_name}"] = df[target_name].values
    return transformed_feats, predictions


def process_predictions(
    df: pd.DataFrame,
    model: Any,
    feats: List[str],
    target_name: str,
    target_class: str,
    target_column_names_cpds: list,
    discretisation_method: str,
    discretisation_path: str,
) -> pd.DataFrame:
    """
    Combine user-level predictions and transformed feature splits.

    Parameters
    ----------
    df : pd.DataFrame
        Subsample of a larger DataFrame.
    model : BayesianNetworkEstimator
        Bayesian network model.
    feats : list
        List of features to be mapped to splits.
    target_name : str
        Name of the target variable.
    discretisation_method : str
        Discretisation method used. It can be between `simple`, `tree` or `mdlp`.
    discretisation_path : str
        Path to discretisation model if method
        is `tree` or `mdlp` or discretisation quantile splits if method is `simple`.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing transformed features and predictions.
    """
    transformed_feats, predictions = predict_samples(
        df,
        model,
        feats,
        target_name,
        target_class,
        target_column_names_cpds,
        discretisation_method,
        discretisation_path,
    )
    return pd.concat([transformed_feats, predictions], axis=1)


def extract_predictions(
    df: pd.DataFrame,
    chunk_size: conint(ge=1),
    model: Any,
    features: List[str],
    model_name: str,
    folder_name: str,
    target_name: str,
    target_class: str,
    target_column_names_cpds: list,
    discretisation_method: str,
    discretisation_path: str,
    multiprocessing=False,
):
    """
    Calculate user-level predictions from a Bayesian Network of a large dataset and mapped features.
    Save the predictions into a specified folder using multiprocessing if enabled.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features for calculating predictions.
    chunk_size : int
        Size of each processing chunk.
    model : BayesianNetworkEstimator
        Bayesian network model.
    features : list
        List of features to be used for prediction.
    model_name : list
        Model identifier (e.g., ['tree', 2] for decision tree with 2 splits).
    folder_name : str
        Name of the main folder to save predictions.
    target_name : str
        Name of the target variable.
    discretisation_method : str
        Discretisation method used.
    discretisation_path : str
        Path to discretisation data.
    multiprocessing : bool, optional
        Enable multiprocessing if True, by default False.

    """
    if multiprocessing:
        chunks = np.array_split(df, int(np.ceil(df.shape[0] / chunk_size)))
        pool = Pool(mp.cpu_count())
        async_results = []
        os.makedirs(f"{folder_name}/{model_name[0]}/", exist_ok=True)
        for chunk in chunks:
            async_result = pool.apply_async(
                process_predictions,
                (
                    chunk,
                    model,
                    features,
                    target_name,
                    target_class,
                    target_column_names_cpds,
                    discretisation_method,
                    discretisation_path,
                ),
            )
            async_results.append(async_result)

        results = [async_result.get() for async_result in async_results]
        concatenated_results = pd.concat(results)
        # Save predictions by model index

        concatenated_results.to_csv(
            f"{folder_name}/{model_name[0]}/predictions_model_{model_name[1]}.csv"
        )

        pool.close()
        pool.join()
        return concatenated_results
    else:
        results = process_predictions(
            df,
            model,
            features,
            target_name,
            target_class,
            target_column_names_cpds,
            discretisation_method,
            discretisation_path,
        )
        results.to_csv(
            f"{folder_name}/{model_name[0]}/predictions_model_{model_name[1]}.csv"
        )
        return results


def get_best_threshold(
    predictions: pd.DataFrame, target_name: str, target_class: str, score: str
) -> [Dict, Dict, float]:
    """
    Calculates threshold with best f1 score for a
    set of predictions.

    Parameters
    ----------
    predictions: pd.DataFrame
        Dataset with predictions
    target_name: str
        target name used
    score: str
        score to be calculated. It can be f1-score, precision or recall.

    Returns
    -------
        dict, dict, float
    f1_score_by_threshold, max_value, score_050
    """
    score_by_threshold = dict()
    max_value = dict()
    for threshold in np.arange(0, 1.05, 0.05):
        predictions[f"{target_name}_pred"] = predictions[f"{target_class}"] > threshold

        score_by_threshold[threshold] = classification_report(
            predictions[f"{target_name}_real"].astype(int),
            predictions[f"{target_name}_pred"].astype(int),
            output_dict=True,
        )["1"][score]
        if threshold == 0.5:
            score_050 = score_by_threshold[threshold]

    max_value["best_threshold"] = max(score_by_threshold, key=score_by_threshold.get)
    max_value["value"] = score_by_threshold[max_value["best_threshold"]]

    return score_by_threshold, max_value, score_050


def calculate_score(
    predictions: pd.DataFrame,
    threshold: confloat(ge=0, le=1),
    target_name: str,
    target_class: str,
    score: str,
) -> Dict:
    """
    Calculates a score of minority class given a threshold.

    Parameters
    ----------
    predictions: pd.DataFrame
        Dataset with predictions
    threshold: float
        Threshold to classify users and calculate f1-score, defined between 0 and 1.
    target_name: str
        target name used
    score: str
        score to be calculated. It can be f1-score, precision or recall.

    Returns
    -------
        dict
    """
    score_dict = dict()
    score_dict["threshold"] = threshold
    predictions[f"{target_name}_pred"] = predictions[target_class] > threshold

    score_dict[score] = classification_report(
        predictions[f"{target_name}_real"].astype(int),
        predictions[f"{target_name}_pred"].astype(int),
        output_dict=True,
    )["1"][score]
    return score_dict


def get_bootstrapped_scores(
    predictions, target_name, target_class, threshold, boot_iterations, alpha
):
    """
    Compute bootstrapped precision, recall, and F1-score for a binary classification model.

    This function performs bootstrapped sampling on the predictions of a binary classification model and calculates
    precision, recall, and F1-score for the minority class using different quantiles of the bootstrapped distribution.

    Parameters
    ----------
    predictions : pd.DataFrame
        DataFrame containing prediction results and the actual target values.
    target_name : str
        Name of the target column in the DataFrame.
    threshold : float
        Classification threshold to determine the predicted class.
    boot_iterations : int
        Number of bootstrapped iterations to perform.
    alpha : float
        Significance level used for computing confidence intervals.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing precision, recall, and F1-score calculated using bootstrapped samples.
        The DataFrame has quantile labels ("alpha/2%", "50%", "(1 - alpha/2)%") and columns
        ("quantile", "precision", "recall", "f1-score").
    """
    scores = np.zeros((boot_iterations, 3))

    predictions[f"{target_name}_pred"] = predictions[target_class] > threshold

    for i in range(boot_iterations):
        boot_preds = predictions.sample(predictions.shape[0], replace=True)

        scores_minority = classification_report(
            predictions[f"{target_name}_real"].astype(int),
            boot_preds[f"{target_name}_pred"].astype(int),
            output_dict=True,
        )["1"]

        scores[i, :] = (
            scores_minority["precision"],
            scores_minority["recall"],
            scores_minority["f1-score"],
        )
    scores = pd.DataFrame(np.quantile(scores, [alpha, 0.5, 1 - alpha / 2], axis=0))
    scores.columns = ["precision", "recall", "f1-score"]
    scores.index = [f"{(alpha / 2) * 100}%", "50%", f"{(1 - alpha / 2) * 100}%"]
    return scores.reset_index().rename(columns={"index": "quantile"})


def plot_bootstrapped_scores(scores, path):
    """
    Create a plot illustrating bootstrapped precision, recall, and F1-score for different quantiles.

    This function generates a plot that visualizes bootstrapped precision, recall, and F1-score for different quantiles
    of the distribution. The input DataFrame should contain quantile labels ("alpha/2%", "50%", "(1 - alpha/2)%") as indices
    and columns ("quantile", "precision", "recall", "f1-score").

    Parameters
    ----------
    scores : pd.DataFrame
        DataFrame containing bootstrapped precision, recall, and F1-score for different quantiles.
    path : str
        Path to save the generated plot.
    """
    pivoted_df = scores.pivot(index="model", columns="quantile").rename_axis(
        index="model", columns=["score", "quantile"]
    )
    for score in ["precision", "recall", "f1-score"]:
        score_df = pivoted_df[score].reset_index()
        with plot_style():
            for lower, mean, upper, y in zip(
                score_df["2.5%"],
                score_df["50%"],
                score_df["97.5%"],
                range(len(score_df)),
            ):
                plt.plot((lower, mean, upper), (y, y, y), "ro-", color="tab:blue")
            plt.yticks(range(len(score_df)), list(score_df["model"]))
            plt.title(f"Bootstrapped summaries of {score} for all models")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{path}/bootstrap_scores.png", bbox_inches="tight", dpi=100)
        plt.clf()


def plot_prediction_time(times, title, path):
    """
    Create a bar plot to visualize prediction times for different models.

    This function generates a bar plot to visualize the time taken to calculate predictions for different models.
    The input dictionary should have model names as keys and corresponding time values (in seconds) as values.

    Parameters
    ----------
    times : dict
        Dictionary containing model names as keys and prediction time (in seconds) as values.
    title : str
        Title for the generated plot.
    path : str
        Path to save the generated plot.
    """
    (
        pd.DataFrame.from_dict(times, orient="index")
        .reset_index()
        .rename(columns={0: "time_seconds", "index": "model"})
        .set_index("model")
        .sort_values("time_seconds")
        .plot(kind="bar", legend=False)
    )
    ax = plt.gca()
    ax.set_ylabel("Time (s)", color="tab:red")
    plt.title(f"Time in (s) of calculating predictions - {title}")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path}/prediction_times.png", bbox_inches="tight")
    plt.clf()
