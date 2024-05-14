import pickle
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger
from pydantic import BaseModel, confloat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# class Preprocessor(BaseModel):
#     """
#     This class prepares a dataset for Structural Learning
#     for a Causal graph. A validation set
#     is selected according to dates and categorical features
#     are encoded to continuous values in order for
#     the NOTEARS algorithm to work.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         pandas.DataFrame of features and target.
#     validation_date_range: list
#         list of validation date range for the data to be filtered
#     event_col: str
#         string indicating event column of interest
#     user_id_col: str
#         user id column name
#     date_col: str
#         date column name
#     drop_columns: list
#         list of features/targets to be dropped.
#     categorical_columns: list
#         categorical columns (and their derivatives) to be encoded
#     features_select: list
#         feature columns to be selected
#     sample_frac: float
#         float indicating whether a sample instead of the full
#         validation set should be used for discovery.
#     """

#     df: pd.DataFrame
#     event_col: str
#     user_id_col: str
#     drop_columns: List[str]
#     date_col: Optional[str]
#     categorical_columns: Optional[List[str]]
#     features_select: Optional[List[str]]
#     label_encoder: Optional[sklearn.preprocessing.LabelEncoder]
#     min_max_scaler: Optional[sklearn.preprocessing.MinMaxScaler]
#     sample_frac: Optional[confloat(gt=0, le=1)] = 1

#     class Config:
#         arbitrary_types_allowed = True

#     def preprocess(self, path=None):
#         """
#         Preprocess data for Structure learning with NOTEARS algorithm.

#         Parameters
#         ----------
#         path: str
#             folder path to save label encoder

#         Returns
#         -------
#             pd.DataFrame
#             preprocessed dataset
#         """
#         logger.info("Splitting validation set days")
#         columns_to_drop = []

#         if self.user_id_col is not None:
#             columns_to_drop.append(self.user_id_col)
#         if self.date_col is not None:
#             columns_to_drop.append(self.date_col)

#         validation_set = self.df.drop(columns_to_drop, axis=1).reset_index(drop=True)
#         validation_set[self.event_col] = validation_set[self.event_col].astype(int)

#         if self.drop_columns:
#             validation_set = validation_set.drop(
#                 self.drop_columns, axis=1, errors="ignore"
#             )

#         numerical_cols = list(validation_set.drop(self.event_col, axis=1).columns)

#         if self.categorical_columns:
#             numerical_cols = [
#                 c for c in numerical_cols if c not in self.categorical_columns
#             ]

#         logger.info(f"Pre-process numerical features: {numerical_cols}")
#         self.min_max_scaler = MinMaxScaler()

#         processed_data = pd.concat(
#             [
#                 pd.DataFrame(
#                     self.min_max_scaler.fit_transform(validation_set[numerical_cols])
#                 ),
#                 validation_set[self.event_col],
#             ],
#             axis=1,
#         )

#         processed_data.columns = validation_set[
#             numerical_cols + [self.event_col]
#         ].columns

#         if self.categorical_columns:
#             categorical_cols = []

#             for col in validation_set.columns:
#                 for feat in self.categorical_columns:
#                     if col.startswith(feat + "_lag"):
#                         categorical_cols.append(col)
#                         break

#             logger.info("Pre-process categorical features to numerical values")

#             self.label_encoder = LabelEncoder()

#             for col in self.categorical_columns + categorical_cols:
#                 validation_set[col] = self.label_encoder.fit_transform(
#                     validation_set[col]
#                 )

#             X_cat = validation_set[self.categorical_columns + categorical_cols]
#             processed_data = pd.concat([processed_data, X_cat], axis=1)

#         if self.features_select:
#             logger.info("Selecting features specified by user")
#             cols_to_select = self.features_select + [self.event_col]
#             processed_data = processed_data[cols_to_select]
#             for feat in self.features_select:
#                 processed_data[feat] = processed_data[feat].astype(float)
#         if path:
#             self.save_preprocessors(path)

#         return processed_data.sample(frac=self.sample_frac)


class Preprocessor(BaseModel):
    """
    This class prepares a dataset for Structural Learning
    for a Causal graph. A validation set
    is selected according to dates and categorical features
    are encoded to continuous values in order for
    the NOTEARS algorithm to work.
    """

    df: pd.DataFrame
    event_col: str
    date_col: str
    user_id_col: Optional[str]
    drop_columns: List[str]
    categorical_columns: Optional[List[str]]
    features_select: Optional[List[str]]
    sample_frac: Optional[confloat(gt=0, le=1)] = 1

    class Config:
        arbitrary_types_allowed = True

    def preprocess(self, artifacts_dir: Optional[Path] = None):
        logger.info("Preprocessing data")
        # Drop specified columns
        columns_to_drop = [self.user_id_col, self.date_col]
        validation_set = self.df.drop(
            [col for col in columns_to_drop if col], axis=1
        ).reset_index(drop=True)
        validation_set[self.event_col] = validation_set[self.event_col].astype(int)

        # Drop additional specified columns
        if self.drop_columns:
            validation_set = validation_set.drop(
                self.drop_columns, axis=1, errors="ignore"
            )

        # Handle numerical columns
        numerical_cols = [
            col
            for col in validation_set.columns
            if col != self.event_col and col not in (self.categorical_columns or [])
        ]

        # Initialize and apply MinMaxScaler to numerical columns
        min_max_scaler = MinMaxScaler()
        if numerical_cols:
            validation_set[numerical_cols] = min_max_scaler.fit_transform(
                validation_set[numerical_cols]
            )

        # Handle categorical columns
        if self.categorical_columns:
            label_encoder = LabelEncoder()
            for col in self.categorical_columns:
                validation_set[col] = label_encoder.fit_transform(validation_set[col])

        # Feature selection
        if self.features_select:
            validation_set = validation_set[self.features_select + [self.event_col]]

        # Sampling
        processed_data = validation_set.sample(frac=self.sample_frac)

        if artifacts_dir:
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            scaler_file = artifacts_dir / "scaler.pkl"
            # Optionally save the preprocessors
            with scaler_file.open("wb") as fp:
                pickle.dump(min_max_scaler, fp)

            if self.categorical_columns:
                label_encoder_file = artifacts_dir / "label_encoder.pkl"
                with label_encoder_file.open("wb") as fp:
                    pickle.dump(label_encoder, fp)

        return processed_data
