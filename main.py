import sys
import os
import yaml

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.preprocessing import Preprocessor
from src.components.model_creation import ANNModel, ProphetModel
from src.components.model_training import ANNTrainer, ProphetTrainer
from src.components.model_evaluation import ModelEvaluator


def main():
    logging.info("Main program has started.")

    loader = DataIngestion()
    # raw_datasets_dict = loader.initiate_raw_data_ingestion()

    # cleaning_config_path = os.path.join("configs", "raw_data_cleaning_config.yaml")
    # with open(cleaning_config_path, "r") as file:
    #     cleaning_configs = yaml.safe_load(file)

    # cleaned_dataframes = {}

    # for name, df in raw_datasets_dict.items():
    #     config = cleaning_configs.get(name, {})
    #     cleaning = DataCleaning(df=df, filename=name, cleaning_config=config)
    #     cleaned_dataframes[name] = cleaning.run_all_cleaning_functions_and_save()

    # loader.load_classification_data_save_to_csv()

    # binary_classification_df = (
    #     loader.initiate_classification_data_ingestion_agumentation_merging()
    # )

    # cleaning_config_path = os.path.join(
    #     "configs", "classification_data_cleaning_config.yaml"
    # )
    # with open(cleaning_config_path, "r") as file:
    #     cleaning_configs = yaml.safe_load(file)

    # df_name = "binary_classification"
    # config = cleaning_configs.get(df_name, {})
    # classification_cleaning = DataCleaning(
    #     df=binary_classification_df,
    #     filename=df_name,
    #     cleaning_config=config,
    # )
    # cleaned_binary_classification_df = (
    #     classification_cleaning.run_all_cleaning_functions_and_save()
    # )

    # preprocessor = Preprocessor(
    #     cleaned_binary_classification_df,
    #     "y_repeat_90d",
    #     test_size=0.2,
    #     random_state=42,
    # )
    # X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_fit_transform()

    # ann = ANNModel(input_dim=X_train.shape[1])

    # trainer = ANNTrainer()
    # trainer.train_and_save_ann_model(
    #     ann=ann,
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_val=X_val,
    #     y_val=y_val,
    # )

    # model_path = os.path.join("models", "classification_ann_model.h5")
    excel_path = os.path.join("reports", "models_evaluations.xlsx")

    # evaluator = ModelEvaluator(model_path, excel_path)
    # evaluator.find_best_threshold(X_val, y_val)
    # evaluator.evaluate_model(X_test, y_test)

    forecasting_df = loader.initiate_forecasting_data_ingestion()

    cleaning_config_path = os.path.join(
        "configs", "forecasting_data_cleaning_config.yaml"
    )
    with open(cleaning_config_path, "r") as file:
        cleaning_configs = yaml.safe_load(file)

    df_name = "forecasting_daily_revenue"
    config = cleaning_configs.get(df_name, {})
    forecasting_cleaning = DataCleaning(
        df=forecasting_df,
        filename=df_name,
        cleaning_config=config,
    )
    cleaned_forecasting_df = forecasting_cleaning.run_all_cleaning_functions_and_save()

    prophet_model = ProphetModel().get_model()
    prophet_trainer = ProphetTrainer(model=prophet_model)
    X_test, y_test = prophet_trainer.fit_split_save_test_model(
        cleaned_forecasting_df, test_size=30
    )

    model_path = os.path.join("models", "forecasting_prophet_splited_model.pkl")
    evaluator = ModelEvaluator(model_path, excel_path)
    evaluator.evaluate_model(X_test, y_test)

    prophet_model_final = ProphetModel().get_model()
    prophet_trainer_final = ProphetTrainer(model=prophet_model_final)
    prophet_model_final = prophet_trainer_final.fit_save_final_model(
        cleaned_forecasting_df
    )

    future = prophet_model_final.make_future_dataframe(periods=30, freq="D")
    forecast = prophet_model_final.predict(future)

    print(forecast.tail(30))

    logging.info("Main program ended.")


if __name__ == "__main__":
    try:
        main()

    except Exception as e:
        print("Critical error.")
        logging.info("Critical error.")
        raise CustomException(e, sys) from e
