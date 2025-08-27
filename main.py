"""
Main script to run machine learning pipelines:
- ANN classification
- Prophet forecasting
- BERT sentiment analysis
"""

import os
import sys
import yaml
import torch

from transformers import get_linear_schedule_with_warmup

from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_cleaning import DataCleaning
from src.components.preprocessing import (
    Preprocessor,
    TextPreprocessor,
    prepare_text_df,
)
from src.components.model_creation import (
    ANNModel,
    ProphetModel,
    BertModel,
)
from src.components.model_training import (
    ANNTrainer,
    ProphetTrainer,
    BertTrainer,
)
from src.components.model_evaluation import ModelEvaluator
from src.utils import save_forecast_and_plot


def run_raw_data_basic_cleaning(loader: DataIngestion) -> None:
    """_summary_

    Args:
        loader (DataIngestion): _description_
    """

    logging.info("Raw data basic cleaning has started.")

    raw_datasets_dict = loader.initiate_raw_data_ingestion()

    cleaning_config_path = os.path.join("configs", "raw_data_cleaning_config.yaml")
    with open(cleaning_config_path, "r") as file:
        cleaning_configs = yaml.safe_load(file)

    cleaned_dataframes = {}

    for name, df in raw_datasets_dict.items():
        config = cleaning_configs.get(name, {})
        cleaning = DataCleaning(df=df, filename=name, cleaning_config=config)
        cleaned_dataframes[name] = cleaning.run_all_cleaning_functions_and_save()

    loader.load_classification_data_save_to_csv()

    logging.info("Raw data basic cleaning has finished.")


def run_ann_pipeline(loader: DataIngestion, excel_path: str) -> None:
    """
    Train, evaluate and save an ANN model for binary classification.

    Args:
        loader (DataIngestion): Loader instance providing raw and prepared datasets.
        excel_path (str): Path to Excel file where metrics will be stored.
    """

    logging.info("ANN pipeline has started.")

    binary_classification_df = (
        loader.initiate_classification_data_ingestion_agumentation_merging()
    )
    cleaning_config_path = os.path.join(
        "configs", "classification_data_cleaning_config.yaml"
    )
    with open(cleaning_config_path, "r") as file:
        cleaning_configs = yaml.safe_load(file)
    df_name = "binary_classification"
    config = cleaning_configs.get(df_name, {})
    classification_cleaning = DataCleaning(
        df=binary_classification_df,
        filename=df_name,
        cleaning_config=config,
    )
    cleaned_binary_classification_df = (
        classification_cleaning.run_all_cleaning_functions_and_save()
    )

    preprocessor = Preprocessor(
        cleaned_binary_classification_df,
        "y_repeat_90d",
        test_size=0.2,
        random_state=42,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_fit_transform()

    ann = ANNModel(input_dim=X_train.shape[1])

    trainer = ANNTrainer()
    trainer.train_and_save_ann_model(
        ann=ann,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    model_path = os.path.join("models", "classification_ann_model.h5")
    evaluator = ModelEvaluator(model_path, excel_path)
    evaluator.find_best_threshold(X_val, y_val)
    evaluator.evaluate_model(X_test, y_test)

    logging.info("ANN pipeline has finished.")


def run_prophet_pipeline(loader: DataIngestion, excel_path: str) -> None:
    """
    Train, evaluate and save a Prophet model for time series forecasting.

    Args:
        loader (DataIngestion): Loader instance providing raw and prepared datasets.
        excel_path (str): Path to Excel file where metrics will be stored.
    """

    logging.info("Prophet pipeline has started.")

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

    save_forecast_and_plot(
        prophet_model=prophet_model_final,
        history_df=cleaned_forecasting_df,
        periods_list=[30, 60, 90],
    )

    logging.info("Prophet pipeline has finished.")


def run_bert_pipeline(loader: DataIngestion, excel_path: str) -> None:
    """
    Train and prepare evaluation for BERT sentiment analysis.

    Args:
        loader (DataIngestion): Loader instance providing raw and prepared datasets.
        excel_path (str): Path to Excel file where metrics will be stored.
    """
    logging.info("BERT pipeline has started.")

    num_epochs = 3
    batch_size = 32

    sentiment_df = loader.initiate_sentiment_data_ingestion()
    sentiment_df_model = prepare_text_df(sentiment_df)

    text_preprocessor = TextPreprocessor(
        df=sentiment_df_model,
        text_col="text_pt",
        target_col="label",
        max_len=128,
    )
    train_loader, val_loader, test_loader = text_preprocessor.create_datasets(
        batch_size=batch_size
    )

    bert_model = BertModel(num_labels=3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device=device)

    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)

    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    criterion = torch.nn.CrossEntropyLoss()

    bert_trainer = BertTrainer(model=bert_model, device=device)
    bert_trainer.train_and_save_bert_model(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=num_epochs,
    )

    evaluator = ModelEvaluator(
        model_path=os.path.join("models", "sentiment_bert_model.pt"),
        excel_path=excel_path,
        model=bert_model,
        device=device,
    )
    evaluator.evaluate_model(X_test=None, y_test=None, dataloader=test_loader)

    logging.info("BERT pipeline has finished.")


def main():
    logging.info("Main program has started.")

    try:
        loader = DataIngestion()
        excel_path = os.path.join("reports", "models_evaluations.xlsx")

        # run_raw_data_basic_cleaning(loader=loader)
        # run_ann_pipeline(loader=loader, excel_path=excel_path)
        # run_prophet_pipeline(loader=loader, excel_path=excel_path)
        run_bert_pipeline(loader=loader, excel_path=excel_path)

        logging.info("Main program ended.")

    except Exception as e:
        logging.info("Critical error occurred in main.")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    main()
