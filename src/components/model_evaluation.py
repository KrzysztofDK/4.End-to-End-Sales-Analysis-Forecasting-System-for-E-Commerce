"""Module to evaluate trained models."""

import os
import sys

import keras
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
)

from src.logger import logging
from src.exception import CustomException


class ModelEvaluator:
    """Class responsible for model evaluation with classification_raport and roc_auc"""

    _excel_cleared = False

    def __init__(self, model_path: str, excel_path: str):
        """Initialize ModelEvaluator.

        Args:
            model_path (str): Location of model to evaluate.
            excel_path (str): Location to wrire excel file with metrics.
        """
        self.model = keras.models.load_model(model_path)
        self.model_name = os.path.splitext(os.path.basename(model_path))[0]
        self.excel_path = excel_path
        self.best_threshold = 0.5

        if not ModelEvaluator._excel_cleared:
            if os.path.exists(excel_path):
                os.remove(excel_path)
            ModelEvaluator._excel_cleared = True

    def find_best_threshold(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Find best threshold based on F1 score from validation set.

        Args:
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
        """

        logging.info("Function to find best threshold has started.")

        preds_val = self.model.predict(X_val).ravel()
        precision, recall, thresholds = precision_recall_curve(y_val, preds_val)

        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)

        self.best_threshold = thresholds[best_idx]

        logging.info(f"Best threshold found: {self.best_threshold:.4f}")

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """Function to evaluate model

        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
        """

        logging.info("Function to evaluate model has started.")

        try:
            preds = self.model.predict(X_test).ravel()
            preds_binary = (preds > self.best_threshold).astype(int)

            report = classification_report(y_test, preds_binary, output_dict=True)
            auc = roc_auc_score(y_test, preds)

            df_report = pd.DataFrame(report).transpose()
            df_report["roc_auc"] = np.nan
            df_report.loc["overall", :] = np.nan
            df_report.loc["overall", "roc_auc"] = auc
            df_report.loc["overall", "threshold"] = self.best_threshold

            header = pd.DataFrame(
                [[f"Model: {self.model_name}"]], columns=["classification_report"]
            )

            if os.path.exists(self.excel_path):
                with pd.ExcelWriter(
                    self.excel_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="overlay",
                ) as writer:
                    existing = pd.read_excel(
                        self.excel_path, sheet_name="Sheet1", header=None
                    )
                    startrow = len(existing) + 2
                    header.to_excel(
                        writer, startrow=startrow, index=False, header=False
                    )
                    df_report.to_excel(writer, startrow=startrow + 2)
            else:
                with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                    header.to_excel(writer, startrow=0, index=False, header=False)
                    df_report.to_excel(writer, startrow=2)

        except Exception as e:
            logging.info("Function to evaluate model has encountered a problem.")
            raise CustomException(e, sys) from e
