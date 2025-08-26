"""Module to evaluate trained models."""

import os
import sys
from typing import Optional, Union

import keras
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
)

from src.logger import logging
from src.exception import CustomException


class ModelEvaluator:
    """Class responsible for model evaluation (ANN or Prophet)"""

    _excel_cleared = False

    def __init__(
        self,
        model_path: str,
        excel_path: str,
        model: Optional[object] = None,
        device: Optional[object] = None,
    ):
        """Initialize ModelEvaluator.

        Args:
            model_path (str): Path to the model file (.h5 for ANN, .pkl for Prophet).
            excel_path (str): Path where Excel with metrics will be saved.
            model (object, optional): Already loaded model.
            device (object, optional): Torch device for BERT.
        """
        self.model_path = model_path
        self.excel_path = excel_path
        self.best_threshold = 0.5
        self.device = device

        try:
            if model is not None:
                self.model_type = "bert"
                self.model = model
                self.model_name = "bert_sentiment"
            elif model_path.endswith(".h5"):
                self.model_type = "ann"
                self.model = keras.models.load_model(model_path)
                self.model_name = os.path.splitext(os.path.basename(model_path))[0]
            elif model_path.endswith(".pkl"):
                self.model_type = "prophet"
                self.model = joblib.load(model_path)
                self.model_name = os.path.splitext(os.path.basename(model_path))[0]
            else:
                raise ValueError("Unsupported model type or missing model.")
        except Exception as e:
            logging.info("Initializing ModelEvaluator has failed.")
            raise CustomException(e, sys) from e

        if not ModelEvaluator._excel_cleared:
            if os.path.exists(excel_path):
                os.remove(excel_path)
            ModelEvaluator._excel_cleared = True

    def find_best_threshold(
        self, X_val: np.ndarray, y_val: Optional[np.ndarray] = None
    ) -> None:
        """Find best threshold based on F1 score from validation set.

        Args:
            X_val (np.ndarray): Validation features
            y_val (np.ndarray, optional): Validation labels. Required for ANN.
        """

        logging.info("Function to find best threshold has started.")

        preds_val = self.model.predict(X_val).ravel()
        precision, recall, thresholds = precision_recall_curve(y_val, preds_val)

        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)

        self.best_threshold = thresholds[best_idx]

        logging.info(f"Best threshold found: {self.best_threshold:.4f}")

    def evaluate_model(
        self,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None,
        dataloader: Optional[object] = None,
    ) -> None:
        """Evaluate model depending on its type (ANN, Prophet or BERT).

        Args:
            X_test : np.ndarray or pd.DataFrame, optional
                Test features. Required for ANN (np.ndarray) and Prophet (pd.DataFrame).
            y_test : np.ndarray or pd.Series, optional
                Test labels. Required for ANN (np.ndarray) and Prophet (pd.Series).
            dataloader : object, optional
                Torch DataLoader for BERT evaluation.
        """

        logging.info("Function to evaluate model has started.")

        try:
            if self.model_type == "ann":
                preds = self.model.predict(X_test).ravel()
                preds_binary = (preds > self.best_threshold).astype(int)

                report = classification_report(y_test, preds_binary, output_dict=True)
                auc = roc_auc_score(y_test, preds)

                df_report = pd.DataFrame(report).transpose()
                df_report["roc_auc"] = np.nan
                df_report.loc["overall", :] = np.nan
                df_report.loc["overall", "roc_auc"] = auc
                df_report.loc["overall", "threshold"] = self.best_threshold

            elif self.model_type == "prophet":
                forecast = self.model.predict(X_test)
                y_pred = forecast["yhat"].values
                y_true = y_test.values

                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = mse**0.5
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

                df_report = pd.DataFrame(
                    {
                        "mae": [mae],
                        "mse": [mse],
                        "rmse": [rmse],
                        "mape": [mape],
                    },
                    index=["metrics"],
                )

            elif self.model_type == "bert":
                import torch

                self.model.eval()
                all_preds, all_labels = [], []

                with torch.no_grad():
                    for batch in dataloader:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].cpu().numpy()

                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

                        all_preds.extend(preds)
                        all_labels.extend(labels)

                report = classification_report(all_labels, all_preds, output_dict=True)
                df_report = pd.DataFrame(report).transpose()

            with pd.ExcelWriter(
                self.excel_path,
                engine="openpyxl",
                mode="a" if os.path.exists(self.excel_path) else "w",
            ) as writer:
                df_report.to_excel(writer, sheet_name=self.model_name)

        except Exception as e:
            logging.info("Function to evaluate model has encountered a problem.")
            raise CustomException(e, sys) from e
