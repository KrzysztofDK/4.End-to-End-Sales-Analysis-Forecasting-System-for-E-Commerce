"""Module to ingest data."""

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


@dataclass
class SqlDataIngestionConfig:
    """Config class with paths to data files."""

    customer_label_data_path: str = os.path.join("SQL", "data", "customer_label.csv")
    first_order_data_path: str = os.path.join("SQL", "data", "first_order.csv")
    first_order_items_data_path: str = os.path.join(
        "SQL", "data", "first_order_items.csv"
    )
    first_order_customer_data_path: str = os.path.join(
        "SQL", "data", "first_order_customer.csv"
    )
    first_order_payment_data_path: str = os.path.join(
        "SQL", "data", "first_order_payment.csv"
    )
    customer_classification_features_data_path: str = os.path.join(
        "SQL", "data", "customer_classification_features.csv"
    )


class DataIngestion:
    """Class to ingest data."""

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.sql_ingestion_config = SqlDataIngestionConfig()

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

    def initiate_sql_data_ingestion_agumentation_merging(self) -> pd.DataFrame:
        """Function to initiate sql data ingestion, agumentation and merging.

        Raises:
            CustomException

        Returns:
            pd.DataFrame: DataFrame of merged sql csv files.
        """

        logging.info("Function to ingest sql data has started.")

        try:
            first_order = pd.read_csv(self.sql_ingestion_config.first_order_data_path)
            first_order_items = pd.read_csv(
                self.sql_ingestion_config.first_order_items_data_path
            )
            first_order_payment = pd.read_csv(
                self.sql_ingestion_config.first_order_payment_data_path
            )
            first_order_customer = pd.read_csv(
                self.sql_ingestion_config.first_order_customer_data_path
            )
            customer_label = pd.read_csv(
                self.sql_ingestion_config.customer_label_data_path
            )

            df = (
                first_order.merge(first_order_items, on="customer_id", how="left")
                .merge(first_order_payment, on="customer_id", how="left")
                .merge(first_order_customer, on="customer_id", how="left")
                .merge(customer_label, on="customer_id", how="left")
            )

            df["dow"] = (
                pd.to_datetime(
                    df["t0_order_approved_at"],
                    errors="coerce",
                    format="%Y-%m-%d %H:%M:%S",
                ).dt.dayofweek
                + 1
            )

            df["month"] = pd.to_datetime(
                df["t0_order_approved_at"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
            ).dt.month

            df["first_gmv"] = df["total_items_value"] + df["total_freight_value"]

            df.to_csv(
                self.sql_ingestion_config.customer_classification_features_data_path,
                index=False,
                sep=",",
                decimal=".",
                date_format="%Y-%m-%d %H:%M:%S",
                encoding="utf-8",
            )

            return df

        except Exception as e:
            logging.info("Function to ingest sql data has encountered a problem.")
            raise CustomException(e, sys) from e
