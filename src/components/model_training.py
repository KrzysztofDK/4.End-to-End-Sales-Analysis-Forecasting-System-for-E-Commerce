"""Module to train and save models with hyperparameter tuning."""

import os
import sys
import joblib
import numpy as np
import keras_tuner as kt

from src.logger import logging
from src.exception import CustomException
from src.components.model_creation import ANNModel


class ModelTrainer:
    """
    Class responsible for hyperparameter tuning, final training and saving Keras models.
    """

    def __init__(self):
        self.model = None

    def train_and_save(
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
                project_name=tuner_project,
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
