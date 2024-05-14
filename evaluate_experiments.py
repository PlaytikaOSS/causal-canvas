import json
import os
import time

import fire
import pandas as pd
import yaml
from loguru import logger

from causal_canvas.evaluation_utils import (
    extract_all_models,
    extract_predictions,
    get_best_threshold,
    get_bootstrapped_scores,
    plot_bootstrapped_scores,
    plot_conditionals,
    plot_prediction_time,
)
from causal_canvas.script_config import ScriptConfigEvaluation


def predict_and_evaluate_models(config_file: str):
    logger.info("Config read")

    config = ScriptConfigEvaluation.load_yaml(config_file)
    logger.info(config)

    model_path = "/bayesian_network/model_fit/bayesian_network.pkl"
    cpds_path = (
        f"/bayesian_network/dependency_estimates/{config.event_column['name']}.csv"
    )
    discretiser_path = "/bayesian_network/model_fit/discretiser.pkl"
    q_splits_path = "/bayesian_network/model_fit/quantile_splits.json"

    for i, train_data_path in enumerate(config.data_train_input_path):
        logger.info(f"Reading data in {train_data_path}")

        if str(train_data_path).endswith("parquet"):
            data = pd.read_parquet(train_data_path)
        elif str(train_data_path).endswith("csv"):
            data = pd.read_csv(train_data_path)

        if config.event_column["target_type"] != "cont":
            data[config.event_column["name"]] = data[
                config.event_column["name"]
            ].astype(int)
            if data[config.date_column].dtype != "str":
                data[config.date_column] = data[config.date_column].astype(str)

        if config.test_dates and config.optim_test_dates:
            optim_test_set = data[
                data[config.date_column].between(*config.optim_test_dates)
            ]
            test_set = data[data[config.date_column].between(*config.test_dates)]
        else:
            logger.info(f"Reading data in {config.data_test_input_path[i]}")
            if str(config.data_test_input_path[i]).endswith("parquet"):
                data_for_optim_test = pd.read_parquet(config.data_test_input_path[i])
            elif str(config.data_test_input_path[i]).endswith("csv"):
                data_for_optim_test = pd.read_csv(config.data_test_input_path[i])
            optim_test_set = data_for_optim_test.sample(frac=0.5, random_state=11)
            test_set = data_for_optim_test[
                ~data_for_optim_test.index.isin(optim_test_set.index)
            ]

        all_models_path = (
            config.all_models_path
            if len(config.data_train_input_path) == 1
            else config.all_models_path + f"segment_{i}/inference/"
        )

        models_list, model_names_list = extract_all_models(
            all_models_path=all_models_path,
            model_path=model_path,
            inference_methods=config.inference_methods,
            models_combinations=config.models_combinations,
        )
        logger.info(f"Models detected for dataset:{model_names_list}")
        output_path = (
            config.output_path
            if len(config.data_train_input_path) == 1
            else config.output_path + f"segment_{i}/"
        )

        all_predictions = dict()
        scores = dict()
        max_scores = dict()
        scores_050_thresh = dict()
        times_to_predict = dict()
        times_to_predict_test = dict()
        test_predictions = dict()
        bootstrapped_scores = pd.DataFrame(
            columns=["quantile", "precision", "recall", "f1-score", "model"]
        )
        bootstrapped_scores_test = pd.DataFrame(
            columns=["quantile", "precision", "recall", "f1-score", "model"]
        )

        for name, model in zip(model_names_list, models_list):
            reading_path = all_models_path + f"{name[0]}/{name[1]}"

            os.makedirs(output_path + f"test_set_predictions/{name[0]}", exist_ok=True)
            os.makedirs(output_path + f"optim_set_predictions/{name[0]}", exist_ok=True)
            os.makedirs(output_path + "cpd_plots/", exist_ok=True)

            logger.info("Plotting CPDs")
            plot_conditionals(
                cpds_path=reading_path + cpds_path,
                target_column_names=config.event_column["target_column_names_cpds"],
                target_class=config.event_column["target_class"],
                threshold=0.5,
                inference_method=name,
                target=config.event_column["name"].upper(),
                path=output_path + "cpd_plots/",
                sub_path=f"{name[0]}_{name[1]}",
            )

            logger.info("Predicting set for optimisation")
            start_time = time.time()
            optim_test_set_predictions = extract_predictions(
                df=optim_test_set,
                chunk_size=1000,
                model=model,
                features=config.features,
                model_name=name,
                target_name=config.event_column["name"],
                target_column_names_cpds=config.event_column[
                    "target_column_names_cpds"
                ],
                target_class=config.event_column["target_class"],
                discretisation_method=name[1].split("_")[0],
                discretisation_path=reading_path + q_splits_path
                if name[1].split("_")[0] == "simple"
                else reading_path + discretiser_path,
                multiprocessing=config.use_multiprocessing,
                folder_name=output_path + "optim_set_predictions",
            )
            end_time = time.time()
            times_to_predict[f"{name[0]}_{name[1]}"] = end_time - start_time

            all_predictions[name] = optim_test_set_predictions

            logger.info(
                "Calculating bootstrapped scores for validation set before optimisation"
            )

            boot_scores = get_bootstrapped_scores(
                predictions=all_predictions[name],
                target_name=config.event_column["name"],
                target_class=config.event_column["target_class"],
                threshold=0.5,
                boot_iterations=config.boot_iterations,
                alpha=config.alpha,
            )
            boot_scores["model"] = f"{name[0]}_{name[1]}"
            bootstrapped_scores = pd.concat([bootstrapped_scores, boot_scores], axis=0)

            logger.info("Calculating optimised threshold")

            results = get_best_threshold(
                predictions=all_predictions[name],
                target_name=config.event_column["name"],
                target_class=config.event_column["target_class"],
                score=config.score_to_optimise,
            )

            scores[str(name)] = results[0]
            max_scores[str(name)] = results[1]
            scores_050_thresh[str(name)] = results[2]

            logger.info("Calculating test set predictions")

            start_time = time.time()

            # Predict test set based on optimised threshold
            test_predictions[name] = extract_predictions(
                df=test_set,
                chunk_size=1000,
                model=model,
                features=config.features,
                model_name=name,
                target_name=config.event_column["name"],
                target_class=config.event_column["target_class"],
                target_column_names_cpds=config.event_column[
                    "target_column_names_cpds"
                ],
                discretisation_method=name[1].split("_")[0],
                discretisation_path=reading_path + q_splits_path
                if name[1].split("_")[0] == "simple"
                else reading_path + discretiser_path,
                multiprocessing=config.use_multiprocessing,
                folder_name=output_path + "test_set_predictions",
            )

            end_time = time.time()
            times_to_predict_test[f"{name[0]}_{name[1]}"] = end_time - start_time

            logger.info(
                "Calculating bootstrapped scores for test set with optimised threshold"
            )

            # Use optimised threshold to extract bootstrap performances
            boot_scores = get_bootstrapped_scores(
                predictions=all_predictions[name],
                target_name=config.event_column["name"],
                target_class=config.event_column["target_class"],
                threshold=max_scores[str(name)]["best_threshold"],
                boot_iterations=config.boot_iterations,
                alpha=config.alpha,
            )
            boot_scores["model"] = f"{name[0]}_{name[1]}"
            bootstrapped_scores_test = pd.concat(
                [bootstrapped_scores_test, boot_scores], axis=0
            )

        all_scores_dict = {"best_scores": max_scores, "scores": scores}

        out_file = open(output_path + "thresholds.json", "w")

        json.dump(all_scores_dict, out_file)

        out_file.close()

        plot_bootstrapped_scores(
            bootstrapped_scores, output_path + "optim_set_predictions/"
        )
        plot_bootstrapped_scores(
            bootstrapped_scores_test, output_path + "test_set_predictions/"
        )
        if config.test_dates:
            plot_prediction_time(
                times=times_to_predict,
                title="optim set",
                path=output_path + "optim_set_predictions/",
            )
            plot_prediction_time(
                times=times_to_predict_test,
                title="test set",
                path=output_path + "test_set_predictions/",
            )

        bootstrapped_scores.to_csv(
            output_path + "optim_set_predictions/bootstrapped_scores.csv"
        )
        bootstrapped_scores_test.to_csv(
            output_path + "test_set_predictions/bootstrapped_scores.csv"
        )

    logger.info("Writing config file")
    with open(config.output_path + "config.yml", "w") as outfile:
        yaml.dump(dict(config), outfile)


if __name__ == "__main__":
    fire.Fire(predict_and_evaluate_models)
