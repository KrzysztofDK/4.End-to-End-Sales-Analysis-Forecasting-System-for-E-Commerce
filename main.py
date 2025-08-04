import sys

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion


def main():
    logging.info("Main program has started.")

    loader = DataIngestion()
    raw_datasets_dict = loader.initiate_data_ingestion()

    logging.info("Main program ended.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("Critical error.")
        logging.info("Critical error.")
        raise CustomException(e, sys) from e
