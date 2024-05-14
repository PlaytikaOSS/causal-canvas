import json
import pickle
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from causalnex.discretiser import Discretiser
from causalnex.discretiser.abstract_discretiser import (
    AbstractSupervisedDiscretiserMethod,
)
from causalnex.discretiser.discretiser_strategy import (
    DecisionTreeSupervisedDiscretiserMethod,
    MDLPSupervisedDiscretiserMethod,
)
from causalnex.network import BayesianNetwork
from loguru import logger
from pydantic import BaseModel, Field, confloat, conint, validator

DISCRETISER_METHODS = Literal["simple", "tree", "mdlp"]


class InferenceMethod(BaseModel):
    method: Literal["MaximumLikelihoodEstimator", "BayesianEstimator"]
    bayes_prior: Literal["K2", "BDeu"]
    equivalent_sample_size: Optional[int] = None


class BayesianNetworkEstimator(BaseModel):
    """
    This class instatiates and fits a Bayesian network on a
    predefined network structure.
    Bayesian Networks in CausalNex support only discrete distributions.
    Any continuous features, or features with a large number of categories,
    should be discretised prior to fitting the Bayesian Network.
    Models containing variables with many possible values will typically be badly fit,
    and exhibit poor performance.

    There are several discretisation methods that can be selected by the user. The first one
    is simple quantile split discretisation. The second one is via Decision
    Trees where the cutting points on the Decision Tree become
    the chosen discretisation thresholds and the final approach is via the
    MDLP algorithm (Fayyad and Irani, 1993) where dynamic split strategy based
    on binning the number of candidate splits is implemented to increase efficiency
    (Chickering, Meek and Rounthwaite, 2001).

    After discretisation, the conditional probabilities are estimated
    by either Maximum likelihood estimation, or Bayesian estimation with
    K2 prior, i.e, dirichlet where all pseudo_counts are 1,
    or BDeu prior with a dirichlet and using uniform ‘pseudo_counts’ of
    equivalent_sample_size / (node_cardinality * np.prod(parents_cardinalities)).

    The class can be used for predictions as well.

    Parameters
    ----------
    structural_model: causalnex.structure.structuremodel.StructureModel
        discovered graph from causalnex library in the form of a DAG
    train_set: pd.DataFrame
        train set to fit the model on
    numerical_features: list
        numerical features to be discretised
    discretiser_method: str
        `simple` for discretising each numerical feature with respect to its quantiles.
        discretiser_argument can be set to choose the number of quantile splits. Default is 5.
        `tree` for a decision tree for each numerical feature where the cutting points on the
        Decision Tree becomes the chosen discretisation threshold. discretiser_argument can be
        tuned to decide on the tree depth. Default is 5.
        `mdlp` which is a dynamic splitting strategy (Fayyad and Irani, 1993).
    inference_method: dict
        dictionary with method and arguments for fitting the model.
        - method: str
            `MaximumLikelihoodEstimator` or `BayesianEstimator`
        - bayes_prior: str
            K2 or BDeu
        - equivalent_sample_size: int
            used only with BDeu to tune the prior quantity
            equivalent_sample_size / (node_cardinality * np.prod(parents_cardinalities))
    discretiser_argument: int
        for `simple` discretiser reflects the number of quantile splits for the
        features. For `tree` discretiser reflects the maximum tree depth
    """

    structural_model: Any
    train_set: pd.DataFrame
    numerical_features: List[str]
    event_col: str
    discretiser_method: DISCRETISER_METHODS
    inference_method: InferenceMethod
    discretiser_argument: conint(gt=0)
    proportion_threshold: Optional[confloat(gt=0, le=1)]
    split_dictionary: dict = Field(default_factory=dict)
    model: Optional[BayesianNetwork] = None
    discretiser: Optional[AbstractSupervisedDiscretiserMethod] = None
    condenced_thresholds: dict = Field(default_factory=dict)
    discretisation_cutoffs_target: List
    feature_to_distribution_map: Optional[dict] = None
    cat_encoding_mappings: Optional[dict] = None
    max_categories: Optional[int]

    @validator("discretisation_cutoffs_target", pre=True)
    def set_discretisation_cutoffs_target_default(cls, v):
        return [] if v is None else v

    class Config:
        arbitrary_types_allowed = True

    def discretise_data(self, path: Path):
        """
        Process that discretises features
        based on the desired method.

        Parameters
        ----------
        path: str
            if specified, path to save the mapped dictionary of
            splits and the discretiser.

        Returns
        -------
            pd.DataFrame
        transformed (discretised) dataframe
        """
        logger.info(f"Discretising data with {self.discretiser_method}")
        numerical_features = [
            c for c in self.structural_model.nodes if c in self.numerical_features
        ]
        categorical_features = [
            c
            for c in self.structural_model.nodes
            if c not in numerical_features and c not in [self.event_col]
        ]

        if self.discretiser_method == "simple":
            X_train_transformed = self._simple_discretise(
                features=numerical_features, path=path
            )
        elif self.discretiser_method == "tree":
            X_train_transformed = self._tree_discretise(
                x=self.train_set, features=numerical_features, path=path
            )
        else:
            X_train_transformed = self._mdlp_discretise(
                x=self.train_set[numerical_features + [self.event_col]], path=path
            ).drop(self.event_col, axis=1)

        if len(self.discretisation_cutoffs_target) > 0:
            self.train_set = self._known_splits_discretise(
                x=self.train_set,
                splits={self.event_col: self.discretisation_cutoffs_target},
            )
            self.split_dictionary[self.event_col] = self.discretisation_cutoffs_target
            self.condenced_thresholds[self.event_col] = (
                self.discretisation_cutoffs_target
            )

        if self.feature_to_distribution_map and "cat" in list(
            self.feature_to_distribution_map.values()
        ):
            multinomial_feats = [
                feat
                for feat, dist in self.feature_to_distribution_map.items()
                if dist == "cat"
            ]

            X_train_transformed_cats = self.preprocess_categorical_features(
                self.train_set, multinomial_feats
            ).drop(numerical_features + [self.event_col], axis=1)
        else:
            X_train_transformed_cats = self.train_set[categorical_features]

        return pd.concat(
            [
                X_train_transformed.reset_index(drop=True),
                X_train_transformed_cats.reset_index(drop=True),
                self.train_set[self.event_col].reset_index(drop=True),
            ],
            axis=1,
        )

    def _simple_discretise(self, features, path: Path):
        """
        Simple discretiser that splits numerical features
        in discretiser_argument quantiles and
        transforms the mapping.

        Parameters
        ----------
        features: list
            numerical features to be transformed
        path: str
            if specified, numerical splits are saved

        Returns
        -------
            pd.DataFrame
        transformed (discretised) dataframe
        """
        model_fit_path = path / "model_fit"
        model_fit_path.mkdir(exist_ok=True)

        x_new = pd.DataFrame(columns=features)

        logger.info(f"Splitting data in {self.discretiser_argument} quantiles")
        for feat in features:
            disc = Discretiser(method="quantile", num_buckets=self.discretiser_argument)
            x_new[feat] = disc.fit_transform(self.train_set[feat].values)
            self.split_dictionary.update({feat: list(disc.numeric_split_points)})
        logger.info(self.split_dictionary)

        with (model_fit_path / "quantile_splits.json").open("w") as outfile:
            json.dump(self.split_dictionary, outfile)
        return x_new

    def _tree_discretise(self, x, features, path: Path):
        """
        Tree discretiser. Applies a decision tree for each numerical feature wrt the target
        and splits are based on the cutting points of the Decision Tree.

        Parameters
        ----------
        x: pd.DataFrame
            dataframe with data to be transformed
        features: list
            numerical features to be transformed
        path: str
            if specified, model class and disctionary
            with splits are saved.

        Returns
        -------
            pd.DataFrame
        transformed (discretised) dataframe
        """

        logger.info(f"Decision tree with in {self.discretiser_argument} max_depth")

        dt = DecisionTreeSupervisedDiscretiserMethod(
            mode="multi",
            split_unselected_feat=True,
            tree_params={
                "max_depth": self.discretiser_argument,
                "random_state": 11,
                "class_weight": "balanced",
            },
        )

        self.discretiser = dt.fit(
            feat_names=features,
            dataframe=x[features + [self.event_col]],
            target=self.event_col,
            target_continuous=False,
        )

        x_new = pd.DataFrame(self.discretiser.transform(x[features]))

        model_fit_path = path / "model_fit"
        model_fit_path.mkdir(exist_ok=True)

        if self.proportion_threshold:
            x_new = self.get_final_discretisation(x_old=x[features], x_new=x_new)
            with (model_fit_path / "condenced_splits.json").open("w") as outfile:
                json.dump(self.condenced_thresholds, outfile)

        for k, v in self.discretiser.map_thresholds.items():
            self.split_dictionary.update({k: list(v)})

        with (model_fit_path / "discretiser.pkl").open("wb") as fp:
            pickle.dump(self.discretiser, fp)
        with (model_fit_path / "tree_splits.json").open("w") as outfile:
            json.dump(self.split_dictionary, outfile)

        return x_new

    def _mdlp_discretise(self, x, path: Path):
        """
        MDLP algorithm discretiser (Fayyad and Irani, 1993) where dynamic split strategy based
        on binning the number of candidate splits is implemented to increase efficiency
        (Chickering, Meek and Rounthwaite, 2001).

        Parameters
        ----------
        x: pd.DataFrame
            dataframe with data to be transformed
        features: list
            numerical features to be transformed
        path: Path
            if specified, model class and dictionary
            with splits are saved.

        Returns
        -------
            pd.DataFrame
        transformed (discretised) dataframe
        """

        dt = MDLPSupervisedDiscretiserMethod(
            {
                "min_depth": 0,
                "random_state": 11,
                "min_split": self.discretiser_argument,
                "dtype": int,
            },
        )
        features = [f for f in x.columns if f not in [self.event_col]]

        self.discretiser = dt.fit(
            feat_names=features,
            dataframe=x,
            target=self.event_col,
            target_continuous=False,
        )

        # Fix bug where mdlp does not return any cut points for 'unimportant' features
        for feature in features:
            if len(self.discretiser.map_thresholds[feature]) == 0:
                self.discretiser.map_thresholds[feature] = np.array(
                    [np.median(x[features])]
                )

        x_new = pd.DataFrame(self.discretiser.transform(x))

        # save to disk
        model_fit_path = path / "model_fit"
        model_fit_path.mkdir(exist_ok=True)

        if self.proportion_threshold:
            x_new = self.get_final_discretisation(x_old=x, x_new=x_new)
            with (model_fit_path / "condenced_splits.json").open("w") as outfile:
                json.dump(self.condenced_thresholds, outfile)

        for k, v in self.discretiser.map_thresholds.items():
            self.split_dictionary.update({k: list(v)})

        with (model_fit_path / "discretiser.pkl").open("wb") as file_pi:
            pickle.dump(self.discretiser, file_pi)
        with (model_fit_path / "mdlp_splits.json").open("w") as outfile:
            json.dump(self.split_dictionary, outfile)

        return x_new

    def _condense_categories(self, x_new, feature):
        """
        Condense categories in a given feature of the input data frame based on a proportion threshold.

        Parameters
        ----------
        x_new: pd.DataFrame
            The input data frame.
        feature: str
            The name of the feature to process.

        Returns
        -------
            dict
            A mapping of original categories to condensed categories.
        """
        ratios = x_new[feature].value_counts().sort_index() / x_new.shape[0]
        ratios = ratios.reset_index().rename(
            columns={"index": "category", feature: "proportion"}
        )
        category_mapping = {}

        # Combine categories while ensuring no category has less than 5% counts
        new_category = 0
        count_so_far = 0
        for index, row in ratios.iterrows():
            count_so_far += row["proportion"]
            if count_so_far >= self.proportion_threshold:
                category_mapping[row["category"]] = new_category
                new_category += 1
                count_so_far = 0
            else:
                category_mapping[row["category"]] = new_category

        # If the last category doesn't meet the 5% threshold, merge it with the previous category
        if count_so_far < self.proportion_threshold:
            category_mapping[ratios["category"].iloc[-1]] = new_category - 1

        if self.max_categories:
            # Check if the current number of categories exceeds 4
            while new_category + 1 > self.max_categories:
                # Merge categories in ascending order
                unique_mappings = sorted(set(category_mapping.values()))
                merged_category_number = max(unique_mappings) - 1

                for unique_mapping in unique_mappings:
                    if unique_mapping > self.max_categories - 1:
                        for category, mapped_category in category_mapping.items():
                            if mapped_category == unique_mapping:
                                category_mapping[category] = merged_category_number

                # Update the maximum category value
                new_category = new_category - 1

        return category_mapping

    def _replace_thresholds(self, category_mapping, feature):
        """
        Replace original category thresholds with merged values based on category mapping.

        Parameters
        ----------
        category_mapping: dict
            A mapping of original categories to condensed categories.
        feature: str
            The name of the feature to process.

        Returns
        -------
            np.ndarray: An array containsing the merged threshold values.
        """
        # Create a dictionary to track the max key for each value
        max_keys = {}
        for key, value in category_mapping.items():
            if value not in max_keys or key > max_keys[value]:
                max_keys[value] = key

        merged_to_values_mapping = []
        for key, value in max_keys.items():
            value = (
                int(value)
                if value < len(self.discretiser.map_thresholds[feature])
                else int(value) - 1
            )
            merged_to_values_mapping.append(
                self.discretiser.map_thresholds[feature][int(value)]
            )

        return np.sort(list(set(merged_to_values_mapping)))

    def get_final_discretisation(self, x_old, x_new):
        """
        Generate a final discretisation of the input data frame using condensed category thresholds.

        Parameters
        ----------
        x_old: pd.DataFrame
            The original input data frame.
        x_new (DataFrame): The modified data frame after DT or MDLP discretisation.

        Returns
        -------
            pd.DataFrame
            The final discretised data frame with condensed categories.
        """
        self.condenced_thresholds = dict()
        for feature in self.discretiser.map_thresholds.keys():
            if len(self.discretiser.map_thresholds[feature]) >= 3:
                category_mapping = self._condense_categories(x_new, feature)
                self.condenced_thresholds[feature] = self._replace_thresholds(
                    category_mapping, feature
                )
            else:
                self.condenced_thresholds[feature] = self.discretiser.map_thresholds[
                    feature
                ]

        for feature in self.discretiser.map_thresholds.keys():
            self.condenced_thresholds[feature] = list(
                self.condenced_thresholds[feature]
            )
        x_new = self._known_splits_discretise(x_old, self.condenced_thresholds)
        return x_new

    def _known_splits_discretise(self, x, splits):
        """
        Discretiser based on known splits.
        This method is useful for already trained
        simple discretiser, to reconstruct
        the mapping on a new dataset.

        Parameters
        ----------
        x: pd.DataFrame
            dataframe with data to be transformed
        splits: list
            numerical splits as
            extracted from simple discretisation

        Returns
        -------
            pd.DataFrame
        transformed (discretised) dataframe

        """
        x_new = pd.DataFrame(columns=splits.keys())
        for feat, split in splits.items():
            disc = Discretiser(method="fixed", numeric_split_points=split)
            x_new[feat] = disc.transform(x[feat].values)
        remaining_columns = [c for c in x.columns if c not in splits.keys()]
        x_new = pd.concat(
            [x_new.reset_index(drop=True), x[remaining_columns].reset_index(drop=True)],
            axis=1,
        )
        return x_new

    def _get_single_feature_map(self, feature: str):
        """
        Accepts a feature name and returns
        a dictionary with well-defined mappings
        of boundaries based on their splits.

        Parameters
        ----------
        feature: str
            numerical feature's name associated with the discretisation

        Returns
        -------
            dict
            mapping of values to meaningful intervals
        """
        mapping_dict = (
            self.condenced_thresholds
            if self.condenced_thresholds
            else self.split_dictionary
        )

        mapping = {0: f"[-inf, {mapping_dict[feature][0]})"}
        for i in range(len(mapping_dict[feature]) - 1):
            mapping.update(
                {i + 1: f"[{mapping_dict[feature][i]}, {mapping_dict[feature][i + 1]})"}
            )
        mapping.update(
            {len(mapping_dict[feature]): f"[{mapping_dict[feature][-1]}, inf)"}
        )
        return mapping

    def get_node_cpds(self, node, path: Path):
        """
        Returns the predicted conditional dependence
        probabilities for each segment of a node
        with respect to the segments of its
        parents.

        Parameters
        ----------
        node: str
            name of node to extract the dependencies
        path: path
            path to save the conditional probabilities

        Returns
        -------
            pd.DataFrame
            dataframe with conditional probabilities
        """
        if self.model:
            dependencies = self.model.cpds[node]
            values_in_node = list(dependencies.index.values)
            dependencies = dependencies.T.reset_index()
            causal_associated_cols = [
                c for c in dependencies.columns if c not in values_in_node
            ]
            if self.split_dictionary:
                for col in causal_associated_cols:
                    if col in self.split_dictionary.keys():
                        mapping = self._get_single_feature_map(feature=col)
                        dependencies[col] = dependencies[col].map(mapping)
                if node in self.split_dictionary.keys():
                    dependencies = dependencies.rename(
                        columns=self._get_single_feature_map(feature=node)
                    )
            if self.discretisation_cutoffs_target:
                dependencies = dependencies.rename(
                    columns={
                        i: f"up to {cut}"
                        for i, cut in enumerate(self.discretisation_cutoffs_target)
                    }
                ).rename(columns={len(self.discretisation_cutoffs_target): "up to inf"})

            dep_estimates_path = path / "dependency_estimates"
            dep_estimates_path.mkdir(exist_ok=True)
            dependencies.to_csv(dep_estimates_path / f"{node}.csv")
            return dependencies

    def fit_bayesian_network(self, path: Path):
        """
        Fits the probabilistic model of a Bayesian
        Network and learns conditional probability distributions
        for all nodes in the Bayesian Network, conditioned on their incoming edges (parents),
        with either Maximum likelihood estimation or with Bayesian estimation.


        Parameters
        ----------
        path: str
            path where the model should be saved

        """

        model_fit_path = path / "model_fit"
        model_fit_path.mkdir(exist_ok=True)

        X = self.discretise_data(path=path)
        logger.info("Discretiser finished")
        logger.info("Fitting Bayesian Network on discovered graph")
        self.model = BayesianNetwork(self.structural_model)
        logger.info(f"Estimating CPDs with {self.inference_method}")
        self.model.fit_node_states_and_cpds(
            X,
            method=self.inference_method.method,
            bayes_prior=self.inference_method.bayes_prior,
            equivalent_sample_size=self.inference_method.equivalent_sample_size,
        )

        with (model_fit_path / "bayesian_network.pkl").open("wb") as fp:
            pickle.dump(self, fp)

    def _tranform_test_data(self, X, method, spliter_path: Optional[Path] = None):
        """
        Transforms test data to follow the same splitting
        rules as the train data.

        Parameters
        ----------
        X: pd.DataFrame
            dataset with test data
        method:
            method that was used to discretise data.
            It can be `simple`, `tree`, `mdlp`.
        spliter_path: str
            path to either the dictionary of splits
            for the simple method or for the
            discretiser model.

        Returns
        -------
            pd.DataFrame
        transformed (discretised) dataframe
        """
        if method == "simple":
            if spliter_path is None:
                raise AttributeError("spliter_path cannot be None with `simple` method")
            else:
                with open(spliter_path, "r") as f:
                    splits = json.load(f)
                x_new = self._known_splits_discretise(x=X, splits=splits)
        elif method in ["tree", "mdlp"]:
            if spliter_path:
                with open(spliter_path, "rb") as f:
                    discretiser = pickle.load(f)
                x_new = discretiser.transform(X)
            elif self.proportion_threshold:
                x_new = self._known_splits_discretise(
                    x=X, splits=self.condenced_thresholds
                )
            else:
                x_new = self.discretiser.transform(X)
        else:
            raise ValueError(f"{method} does not exist")

        if not self.feature_to_distribution_map and "cat" not in list(
            self.feature_to_distribution_map.values()
        ):
            return x_new
        else:
            multinomial_feats = [
                feat
                for feat, dist in self.feature_to_distribution_map.items()
                if dist == "cat"
            ]

            X_transformed_cats = self.preprocess_categorical_features_test(x_new)
            return pd.concat(
                [
                    x_new.drop(multinomial_feats, axis=1).reset_index(drop=True),
                    X_transformed_cats.reset_index(drop=True),
                ],
                axis=1,
            )

    def preprocess_categorical_features(self, X, categorical_features):
        """
        Preprocesses all categorical features by one-hot encoding and expanding columns.

        Args:
            X: The DataFrame containing the features to be processed.

        Returns:
            Preprocessed DataFrame X.
        """
        X_processed = (
            X.copy()
        )  # Make a copy to prevent overwriting the original DataFrame
        self.cat_encoding_mappings = {}
        for feat in categorical_features:
            # Perform one-hot encoding
            encoded = pd.get_dummies(X_processed[feat], prefix=feat, drop_first=False)

            # Store the mapping between original values and encoded columns
            encoding_map = {
                original_value: f"{feat}_{original_value}"
                for original_value in X_processed[feat].unique()
            }
            self.cat_encoding_mappings[feat] = encoding_map

            X_processed.drop(feat, axis=1, inplace=True)

            column_names = [
                f"{feat}{i if i > 0 else ''}" for i in range(len(encoded.columns))
            ]
            encoded.columns = column_names

            # Concatenate encoded columns to original DataFrame
            X_processed = pd.concat([X_processed, encoded], axis=1)
        return X_processed

    def preprocess_categorical_features_test(self, X_test):
        """
        Preprocesses the test set categorical features using the encoding mappings obtained from the train set.

        Args:
            X_test: The DataFrame containing the features of the test set.
            encoding_mappings: A dictionary containing the encoding mappings for each categorical feature.

        Returns:
            Preprocessed DataFrame X_test.
        """
        X_test_processed = X_test[self.cat_encoding_mappings.keys()].copy()

        for feat, encoding_map in self.cat_encoding_mappings.items():
            # Perform one-hot encoding on test set using the encoding mapping
            encoded_test = pd.DataFrame(
                columns=encoding_map.values(), index=X_test_processed.index
            )
            for original_value, encoded_column in encoding_map.items():
                encoded_test[encoded_column] = (
                    X_test_processed[feat] == original_value
                ).astype(int)

            # Add missing columns in test set based on encoding mapping
            encoded_columns_set = set(encoded_test.columns)
            missing_columns = set(encoding_map.values()) - encoded_columns_set
            for col in missing_columns:
                encoded_test[col] = 0

            # Reorder columns to match the train set
            encoded_test = encoded_test[list(encoding_map.values())]
            # Rename columns to match the train set
            column_names = [
                f"{feat}{i if i > 0 else ''}" for i in range(len(encoded_test.columns))
            ]
            encoded_test.columns = column_names

            # Concatenate encoded columns to original DataFrame
            X_test_processed.drop(feat, axis=1, inplace=True)
            X_test_processed = pd.concat([X_test_processed, encoded_test], axis=1)

        return X_test_processed

    def predict(self, X, node, method, spliter_path):
        """
        Transforms a test set and predicts
        conditional dependence probabilities for a specific node
        data based on the discretisation method that was used and
        the probabilistic model that was trained.


        Parameters
        ----------
        X: pd.DataFrame
            dataset with test data
        node: str
            name of node to be predicted
        method: str
            method that was used for discretisation.
            It can be `simple`, `tree`, `mdlp`.
        spliter_path: path
            path to either the dictionary of splits
            for the simple method or for the
            discretiser model.

        Returns
        -------
            pd.DataFrame
            dataframe with predictions
        """
        if self.model:
            transformed_data = self._tranform_test_data(X, method, spliter_path)
            return self.model.predict(transformed_data, node)

    def predict_proba(self, X, node, method, spliter_path):
        if self.model:
            transformed_data = self._tranform_test_data(X, method, spliter_path)
            return self.model.predict_probability(transformed_data, node)
