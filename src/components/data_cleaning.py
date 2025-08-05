"""Module to clean data after basic EDA in Power BI"""

import sys
from typing import Type

import pandas as pd

from src.exception import CustomException
from src.logger import logging


class DataCleaning:
    """Class to clean data by converting column types, column renaming, row deleting, handling duplicates."""

    def __init__(self, df: pd.DataFrame, cleaning_config: dict = None):
        self.original_df = df.copy()
        self.df = df
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
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        logging.info(f"Dropped {before - after} duplicated rows.")

    def run_all_cleaning_functions(self) -> pd.DataFrame:
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
        type_map = {"int": int, "float": float, "str": str, "bool": bool}
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

        logging.info("Function to run all cleaning functions has ended.")
        return self.df
