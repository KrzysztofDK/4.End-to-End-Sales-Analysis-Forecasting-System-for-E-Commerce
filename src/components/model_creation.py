"""Module to create models"""

from keras import layers, models
from keras.losses import BinaryFocalCrossentropy

from src.logger import logging


class ANNModel:
    """Class to create ANN model"""

    def __init__(self, input_dim: int):
        """Initialize ANNModel.

        Args:
            input_dim (int): Number of input features (X_train.shape[1]).
        """

        self.input_dim = input_dim
        self.model = None

    def build_model(self):
        """Function to build and compile ANN model."""

        logging.info("Function to build model has started.")

        self.model = models.Sequential(
            [
                layers.InputLayer(input_shape=(self.input_dim,)),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
            metrics=["accuracy", "AUC"],
        )
        return self.model
