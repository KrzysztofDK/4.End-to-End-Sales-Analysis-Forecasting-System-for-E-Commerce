"""Module to clean data after basic EDA in Power BI"""

import sys
import os
from typing import Type

import pandas as pd

from src.exception import CustomException
from src.logger import logging


class DataCleaning:
    """Class to clean data by converting column types, column renaming, row deleting, handling duplicates."""

    def __init__(self, df: pd.DataFrame, filename: str, cleaning_config: dict = None):
        self.original_df = df.copy()
        self.df = df
        self.filename = filename
        self.config = cleaning_config or {}

    def convert_column_type(self, columns: list[str], to_type: Type) -> None:
        """Converts given column to given type in given dataframe.

        Args:
            columns (list[str]): names of columns to convert.
            to_type (Type): requested data type.

        Raises:
            CustomException
        """

        if not columns:
            logging.info("No columns provided for function to convert column type.")
            return

        for column in columns:
            try:
                logging.info(
                    "Function to convert column types in DataCleaning class has started."
                )
                if to_type == "datetime" or to_type == pd.Timestamp:
                    self.df[column] = pd.to_datetime(
                        self.df[column], errors="coerce", format="%Y-%m-%d %H:%M:%S"
                    )

                else:
                    self.df[column] = self.df[column].astype(to_type)

            except Exception as e:
                logging.info(
                    "Function to convert column types in DataCleaning class has encountered a problem."
                )
                raise CustomException(e, sys) from e

    def rename_columns(self, rename_map: dict) -> None:
        """Function to rename given column to given name.

        Args:
            rename_map (dict): dictionary of columns names and new names to rename.
        """

        if not rename_map:
            logging.info("No rename_map provided for function to rename columns.")
            return

        logging.info("Function to rename columns has started.")
        self.df.rename(columns=rename_map, inplace=True)

    def drop_rows_with_missing(self, columns: list[str]) -> None:
        """Function to drop rows with missing values even of one feature.

        Args:
            columns (list[str]): columns to check.
        """

        if not columns:
            logging.info("No columns provided for function to drop rows.")
            return

        logging.info("Function to drop rows has started.")
        before = len(self.df)
        self.df.dropna(subset=columns, inplace=True)
        after = len(self.df)
        logging.info(f"Dropped {before - after} rows due to NaNs in {columns}.")

    def drop_duplicates(self) -> None:
        """Function to drop duplicated rows."""

        logging.info("Function to drop duplicates has started.")

        if self.filename == "geolocation":
            self.df.drop_duplicates(
                subset=["geolocation_zip_code_prefix"], inplace=True
            )
            logging.info("Function to drop duplicates detected geolocation file.")
        else:
            before = len(self.df)
            self.df.drop_duplicates(inplace=True)
            after = len(self.df)
            logging.info(f"Dropped {before - after} duplicated rows.")

    def run_all_cleaning_functions_and_save(self) -> pd.DataFrame:
        """Function to run all cleaning functions.

        Returns:
            pd.DataFrame: cleaned dataframe.
        """

        logging.info("Function to run all cleaning functions has started.")
        self.drop_duplicates()

        dropna_cols = self.config.get("dropna_columns", [])
        if dropna_cols:
            self.drop_rows_with_missing(dropna_cols)

        convert_map = self.config.get("convert_columns", {})
        type_map = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "datetime": "datetime",
        }
        for col, dtype_str in convert_map.items():
            to_type = type_map.get(dtype_str)
            if to_type is None:
                logging.warning(
                    f"Unknown type '{dtype_str}' for column '{col}'. Skipping."
                )
                continue
            self.convert_column_type([col], to_type)

        rename_map = self.config.get("rename_columns", {})
        if rename_map:
            self.rename_columns(rename_map)

        cleaned_df_path = os.path.join(
            "artifacts", "cleaned_data", f"{self.filename}.csv"
        )
        if self.filename == "order_reviews":
            self.df = self.df.drop(
                ["review_comment_title", "review_comment_message"], axis=1
            )

            cleaned_df_path_for_sql = os.path.join(
                "artifacts", "cleaned_data", f"{self.filename}_sql.csv"
            )

            self.df.to_csv(
                cleaned_df_path_for_sql,
                index=False,
                sep=";",
                decimal=",",
                date_format="%Y-%m-%d %H:%M:%S",
                lineterminator="\n",
            )

        self.df.to_csv(
            cleaned_df_path,
            index=False,
            sep=";",
            decimal=",",
            date_format="%Y-%m-%d %H:%M:%S",
            lineterminator="\n",
        )
        logging.info(f"Cleaned DataFrame saved to {cleaned_df_path}")

        return self.df
