"""Module to train and save models."""

import os
import sys

import numpy as np
import joblib
import keras

from src.logger import logging
from src.exception import CustomException


class ModelTrainer:
    """
    Class responsible for training and saving Keras models.
    """

    def __init__(self, model: keras.Model):
        """Initialize ModelTrainer.

        Args:
            model (keras.Model): A compiled Keras model that will be trained.
        """
        self.model = model

    def train_and_save(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
        batch_size: int = 32,
    ) -> None:
        """Function to train a compiled Keras model.

        Args:
            X_train (np.ndarray): Training input features of shape (n_samples, n_features).
            y_train (np.ndarray): Training labels of shape (n_samples).
            X_val (np.ndarray): Validation input features of shape (n_samples, n_features).
            y_val (np.ndarray): Validation labels of shape (n_samples).
            epochs (int, optional): Number of epochs to train the model (default is 20).
            batch_size (int, optional): Number of samples per gradient update (default is 32).
        """

        logging.info("Function to train model has started.")

        try:
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
            )

            model_path = os.path.join("models", "classification_ann_model.h5")
            history_path = os.path.join("models", "classification_ann_history.pkl")

            self.model.save(model_path)
            joblib.dump(history.history, history_path)

        except Exception as e:
            logging.info("Function to train model has encountered a problem.")
            raise CustomException(e, sys) from e
