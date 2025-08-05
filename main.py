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

    cleaning_config_path = os.path.join("configs", "data_cleaning_config.yaml")
    with open(cleaning_config_path, "r") as file:
        cleaning_configs = yaml.safe_load(file)

    cleaned_dataframes = {}

    for name, df in raw_datasets_dict.items():
        config = cleaning_configs.get(name, {})
        cleaning = DataCleaning(df, cleaning_config=config)
        cleaned_dataframes[name] = cleaning.run_all_cleaning_functions()

    logging.info("Main program ended.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("Critical error.")
        logging.info("Critical error.")
        raise CustomException(e, sys) from e
