"""Module to train and save models."""

import os
import sys

import joblib
import numpy as np
import keras_tuner as kt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prophet import Prophet

from src.logger import logging
from src.exception import CustomException
from src.components.model_creation import ANNModel, BertModel


class ANNTrainer:
    """
    Class responsible for hyperparameter tuning, final training and saving Keras models.
    """

    def __init__(self):
        self.model = None

    def train_and_save_ann_model(
        self,
        ann: ANNModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        tuner_max_trials: int = 15,
        tuner_epochs: int = 10,
        batch_size: int = 32,
        final_epochs: int = 20,
        model_path: str = os.path.join("models", "classification_ann_model.h5"),
        history_path: str = os.path.join("models", "classification_ann_history.pkl"),
    ) -> None:
        """Runs KerasTuner RandomSearch on the provided ANN model, picks the best model,
        optionally continues training, saves the model + history, and returns the best model.

        Args:
        ann (ANNModel): Wrapper object with `.build_model()` function
            used by KerasTuner to create Keras models.
        X_train (np.ndarray): Training input features of shape (n_samples, n_features).
        y_train (np.ndarray): Training labels of shape (n_samples,).
        X_val (np.ndarray): Validation input features of shape (n_samples, n_features).
        y_val (np.ndarray): Validation labels of shape (n_samples,).
        tuner_max_trials (int, optional): Maximum number of different hyperparameter
            configurations to test during tuning. Defaults to 15.
        tuner_epochs (int, optional): Number of epochs to train each trial model
            during hyperparameter search. Defaults to 10.
        batch_size (int, optional): Batch size used both in tuning and
            final training. Defaults to 32.
        final_epochs (int, optional): Additional number of epochs to continue
            training the best model after hyperparameter tuning.
            Set to 0 to skip this step. Defaults to 20.
        model_path (str, optional): File path where the best trained model
            will be saved in `.h5` format. Defaults to "models/classification_ann_model.h5".
        history_path (str, optional): File path where the training history
            (as dict) will be saved via joblib. Defaults to
            "models/classification_ann_history.pkl".
        tuner_project (str, optional): Project name inside the tuner directory,
            used to group related experiments. Defaults to "binary_classification".
        """

        logging.info("Function to train and save model has started.")

        try:
            tuner = kt.RandomSearch(
                hypermodel=ann.build_model,
                objective="val_AUC",
                max_trials=tuner_max_trials,
                executions_per_trial=1,
                overwrite=True,
            )

            tuner.search(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=tuner_epochs,
                batch_size=batch_size,
            )

            self.model = tuner.get_best_models(num_models=1)[0]
            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            logging.info(f"Best HPs: {best_hp.values}")

            history = None
            if final_epochs and final_epochs > 0:
                logging.info("Final training of best model started.")
                history = self.model.fit(
                    X_train,
                    y_train,
                    validation_data=(X_val, y_val),
                    epochs=final_epochs,
                    batch_size=batch_size,
                )

            self.model.save(model_path)
            if history is not None:
                joblib.dump(history.history, history_path)

        except Exception as e:
            logging.info("Function to train and save model has encountered a problem.")
            raise CustomException(e, sys) from e


class ProphetTrainer:
    """Class responsible for training Prophet model."""

    def __init__(
        self,
        model: Prophet,
    ):
        """Initialize trainer with Prophet model and path for saving.

        Args:
            model (Prophet): An initialized Prophet model instance to train.
        """

        self.model = model
        self.X_test = None
        self.y_test = None

    def fit_split_save_test_model(
        self,
        df: pd.DataFrame,
        test_size: int = 30,
        model_path: str = os.path.join(
            "models", "forecasting_prophet_splited_model.pkl"
        ),
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit Prophet model on training set, prepare test set, and save the trained model.
        The last `test_size` rows are held out for evaluation, while the rest is used for training.

        Args:
            df (pd.DataFrame): DataFrame with ['ds', 'y'].
            test_size (int): Number of last observations to keep as test.
            model_path (str, optional): File path where the trained model will be saved.
                                        Default is 'models/forecasting_prophet_splited_model.pkl'.

        Returns:
        tuple[pd.DataFrame, pd.Series]:
            - X_test: DataFrame with the datetime column(s) for prediction
            - y_test: Series with the actual target values for evaluation
        """
        logging.info("Function to split and train Prophet model has started.")

        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        self.X_test = test_df[["ds"]]
        self.y_test = test_df["y"]

        self.model.fit(train_df)
        joblib.dump(self.model, model_path)
        logging.info(f"Final Prophet model saved to {model_path}.")

        return self.X_test, self.y_test

    def fit_save_final_model(
        self,
        df: pd.DataFrame,
        model_path: str = os.path.join("models", "forecasting_prophet_model.pkl"),
    ) -> Prophet:
        """Fit and save to pkl final Prophet model on provided data.

        Args:
            df (pd.DataFrame): DataFrame with ['ds', 'y'].
            model_path (str, optional): File path where the trained model will be saved.
                                        Default is 'models/forecasting_prophet_model.pkl'.

        Returns:
            Prophet: Returns trained Prophet model.
        """

        logging.info("Function to fit final Prophet model has started.")

        self.model.fit(df)
        joblib.dump(self.model, model_path)
        logging.info(f"Final Prophet model saved to {model_path}.")

        return self.model


class BertTrainer:
    """
    Trainer class for fine-tuning, validating, and saving BERT sentiment analysis models.
    """

    def __init__(self, model: BertModel, device: torch.device):
        """Initialize the BERT trainer with a model and device.

        Args:
            model (BertModel): Instance of BERT sentiment classifier.
            device (torch.device, optional): Device to use ("cuda" or "cpu").
                If None, automatically detects GPU if available. Defaults to None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def train_and_save_bert_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        num_epochs: int = 3,
        model_path: str = os.path.join("models", "sentiment_bert_model.pt"),
        history_path: str = os.path.join("models", "sentiment_bert_history.pkl"),
    ) -> None:
        """Train BERT sentiment classifier and save both model and training history.

        Training history is stored separately as a pickle file.
        The method does not return history in order to keep artifacts
        consistent and lightweight.

        Args:
            train_loader (DataLoader): DataLoader providing training batches.
            val_loader (DataLoader): DataLoader providing validation batches.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters (e.g., AdamW).
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
            num_epochs (int, optional): Number of epochs to train. Defaults to 3.
            model_path (str, optional): Path to save the trained model
                (PyTorch `.pt` format). Defaults to "models/bert_sentiment_model.pt".
            history_path (str, optional): Path to save training history
                (Pickle `.pkl` format). Defaults to "models/bert_sentiment_history.pkl".
        """

        logging.info("Function to train BERT model has started.")

        history = {"train_loss": [], "val_loss": []}

        try:
            for epoch in range(num_epochs):
                self.model.train()
                total_loss = 0
                scaler = torch.amp.GradScaler("cuda")
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids, attention_mask, labels = (
                        batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                        batch["labels"].to(self.device),
                    )
                    with torch.cuda.amp.autocast():
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        loss = criterion(outputs.logits, labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    total_loss += loss.item()

                avg_train_loss = total_loss / len(train_loader)
                history["train_loss"].append(avg_train_loss)

                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids, attention_mask, labels = (
                            batch["input_ids"].to(self.device),
                            batch["attention_mask"].to(self.device),
                            batch["labels"].to(self.device),
                        )
                        outputs = self.model(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        loss = criterion(outputs.logits, labels)
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                history["val_loss"].append(avg_val_loss)

                logging.info(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
                )

            torch.save(self.model.state_dict(), model_path)
            joblib.dump(history, history_path)

            logging.info(f"BERT model saved to {model_path}.")
            logging.info(f"BERT training history saved to {history_path}.")

        except Exception as e:
            logging.error("Function to train BERT model has encountered a problem.")
            raise CustomException(e, sys) from e
