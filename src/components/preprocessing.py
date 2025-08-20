"""Module to create preprocessor"""

import sys
import os

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

from src.logger import logging
from src.exception import CustomException


class Preprocessor:
    """Class to transform data with preprocessor."""

    def __init__(
        self, df: pd.DataFrame, target_col: str, test_size: float, random_state: int
    ):
        self.preprocessor = None
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.numeric_features = [
            "n_distinct_categories",
            "n_sellers",
            "n_items",
            "payment_value",
            "total_freight_value",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]
        self.categorical_features = [
            "payment_type",
            "customer_state",
            "top20_categories_and_others",
            "installments_bins",
        ]

    def create_preprocessor(self):
        """Fucntion to create preprocessor to transform data."""

        logging.info("Function to create preprocessor has started.")

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

    def split_fit_transform(self):
        """Function to split and transform data with preprocessor.

        Returns:
            list: list of splited data into X/y train/test/val.
        """

        logging.info("Function to split and transform data has started.")

        self.create_preprocessor()

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

        try:
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_val_transformed = self.preprocessor.transform(X_val)
            X_test_transformed = self.preprocessor.transform(X_test)

            preprocessor_path = os.path.join(
                "models", "classification_ann_preprocessor.pkl"
            )
            joblib.dump(self.preprocessor, preprocessor_path)

            cat_encoder = self.preprocessor.named_transformers_["cat"]["onehot"]
            n_num_features = len(self.numeric_features)
            categorical_indexes = list(
                range(
                    n_num_features,
                    n_num_features + len(cat_encoder.get_feature_names_out()),
                )
            )

            sm = SMOTENC(
                categorical_features=categorical_indexes,
                random_state=self.random_state,
                sampling_strategy="auto",
            )

            X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)

        except Exception as e:
            logging.info(
                "Function to split and transform data has encountered a problem."
            )
            raise CustomException(e, sys) from e

        return (
            X_train_res,
            X_val_transformed,
            X_test_transformed,
            y_train_res,
            y_val,
            y_test,
        )
