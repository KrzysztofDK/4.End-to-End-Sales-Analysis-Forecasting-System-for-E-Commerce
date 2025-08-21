"""Module to ingest data."""

import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
import mysql.connector
from dotenv import load_dotenv

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
        "SQL", "data", "binary_classification.csv"
    )
    env_sql_config_path: str = os.path.join("configs", "sqlconfig.env")


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

    def load_sql_save_to_csv(self) -> None:
        """Function to load specific sql database and save to csv."""

        logging.info("Function to load sql db and save as csv has started.")

        try:
            load_dotenv(dotenv_path=self.sql_ingestion_config.env_sql_config_path)
            conn = mysql.connector.connect(
                host="localhost",
                user=os.getenv("MYSQL_USER"),
                password=os.getenv("MYSQL_PASSWORD"),
                database=os.getenv("MYSQL_DB"),
            )

            df = pd.read_sql(
                "SELECT customer_unique_id, y_repeat_90d FROM customer_label", conn
            )
            df.to_csv(
                self.sql_ingestion_config.customer_label_data_path,
                index=False,
                sep=",",
                quotechar='"',
                quoting=1,
                decimal=".",
            )

            conn.close()

        except Exception as e:
            logging.info(
                "Function to load sql db and save as csv has encountered a problem."
            )
            raise CustomException(e, sys) from e

    def initiate_classification_data_ingestion_agumentation_merging(
        self,
    ) -> pd.DataFrame:
        """Function to initiate sql data ingestion, agumentation and merging.

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
                first_order.merge(
                    first_order_items, on="customer_unique_id", how="left"
                )
                .merge(first_order_payment, on="customer_unique_id", how="left")
                .merge(first_order_customer, on="customer_unique_id", how="left")
                .merge(customer_label, on="customer_unique_id", how="left")
            )

            df["dow"] = (
                pd.to_datetime(
                    df["t0_order_date"],
                    errors="coerce",
                    format="%Y-%m-%d %H:%M:%S",
                ).dt.dayofweek
                + 1
            )

            df["month"] = pd.to_datetime(
                df["t0_order_date"], errors="coerce", format="%Y-%m-%d %H:%M:%S"
            ).dt.month

            df["first_gmv"] = df["total_items_value"] + df["total_freight_value"]

            df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)

            df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            bins = [0, 5, 11, 17, 24]
            labels = ["0-5", "6-11", "12-17", "18-24"]
            df["installments_bins"] = pd.cut(
                df["payment_installments"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            ).astype(str)

            top_20 = (
                df["top_category_by_priciest_item"]
                .value_counts()
                .nlargest(20)
                .index.tolist()
            )
            df["top20_categories_and_others"] = df[
                "top_category_by_priciest_item"
            ].apply(lambda x: x if x in top_20 else "others")

            df.to_csv(
                self.sql_ingestion_config.customer_classification_features_data_path,
                index=False,
                sep=",",
                decimal=".",
                date_format="%Y-%m-%d %H:%M:%S",
            )

            return df

        except Exception as e:
            logging.info("Function to ingest sql data has encountered a problem.")
            raise CustomException(e, sys) from e
