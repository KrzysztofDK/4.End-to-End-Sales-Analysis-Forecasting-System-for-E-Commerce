"""Module to ingest raw data."""

import sys
import os
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Config class with paths to data files."""

    customers_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_customers_raw.csv"
    )
    geolocation_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_geolocation_raw.csv"
    )
    order_items_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_order_items_raw.csv"
    )
    order_payments_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_order_payments_raw.csv"
    )
    order_reviews_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_order_reviews_raw.csv"
    )
    orders_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_orders_raw.csv"
    )
    products_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_products_raw.csv"
    )
    sellers_data_path: str = os.path.join(
        "artifacts", "raw_data", "olist_sellers_raw.csv"
    )


class DataIngestion:
    """Class to ingest data."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> dict:
        """Function to initiate data ingest.

        Raises:
            CustomException

        Returns:
            dict: Dictionary of ingested data as pd.DataFrame.
        """

        logging.info("Function to ingest data has started.")

        try:
            config_dict = vars(self.ingestion_config)

            all_dataframes = {}

            for name, path in config_dict.items():
                df_name = name.replace("_data_path", "")
                logging.info(f"Reading {df_name} from {path}.")
                all_dataframes[df_name] = pd.read_csv(path)

            logging.info("All raw CSVs successfully loaded.")
            return all_dataframes

        except Exception as e:
            logging.info("Function to ingest data has encountered a problem.")
            raise CustomException(e, sys) from e
