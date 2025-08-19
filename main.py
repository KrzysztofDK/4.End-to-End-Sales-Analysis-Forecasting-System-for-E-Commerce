import sys
import os
import yaml

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning


def main():
    logging.info("Main program has started.")

    loader = DataIngestion()
    raw_datasets_dict = loader.initiate_data_ingestion()

    cleaning_config_path = os.path.join("configs", "raw_data_cleaning_config.yaml")
    with open(cleaning_config_path, "r") as file:
        cleaning_configs = yaml.safe_load(file)

    cleaned_dataframes = {}

    for name, df in raw_datasets_dict.items():
        config = cleaning_configs.get(name, {})
        cleaning = DataCleaning(df=df, filename=name, cleaning_config=config)
        cleaned_dataframes[name] = cleaning.run_all_cleaning_functions_and_save()

    # loader.load_sql_save_to_csv()

    customer_classification_df = (
        loader.initiate_classification_data_ingestion_agumentation_merging()
    )

    cleaning_config_path = os.path.join(
        "configs", "classification_data_cleaning_config.yaml"
    )
    with open(cleaning_config_path, "r") as file:
        cleaning_configs = yaml.safe_load(file)

    df_name = "customer_classification_features"
    config = cleaning_configs.get(df_name, {})
    classification_cleaning = DataCleaning(
        df=customer_classification_df,
        filename=df_name,
        cleaning_config=config,
    )
    cleaned_customer_classification_df = (
        classification_cleaning.run_all_cleaning_functions_and_save()
    )

    logging.info("Main program ended.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("Critical error.")
        logging.info("Critical error.")
        raise CustomException(e, sys) from e
