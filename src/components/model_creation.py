"""Module to create models with hyperparameter tuning"""

from keras import layers, models, optimizers
from keras.losses import BinaryFocalCrossentropy

from src.logger import logging


class ANNModel:
    """Class to create ANN model for hyperparameter tuning"""

    def __init__(self, input_dim: int):
        """Initialize ANNModel.

        Args:
            input_dim (int): Number of input features (X_train.shape[1]).
        """
        self.input_dim = input_dim

    def build_model(self, hp):
        """Function for KerasTuner to build a tunable ANN model."""

        logging.info("Function to build tunable model has started.")

        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.input_dim,)))

        model.add(
            layers.Dense(
                units=hp.Int("units1", min_value=64, max_value=256, step=64),
                activation="relu",
            )
        )
        model.add(
            layers.Dropout(
                rate=hp.Float("dropout1", min_value=0.2, max_value=0.5, step=0.1)
            )
        )

        model.add(
            layers.Dense(
                units=hp.Int("units2", min_value=64, max_value=256, step=64),
                activation="relu",
            )
        )
        model.add(
            layers.Dropout(
                rate=hp.Float("dropout2", min_value=0.2, max_value=0.5, step=0.1)
            )
        )

        model.add(layers.Dense(1, activation="sigmoid"))

        lr = hp.Choice("lr", values=[1e-3, 5e-4, 3e-4])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=BinaryFocalCrossentropy(gamma=2.0, alpha=0.75),
            metrics=["accuracy", "AUC"],
        )
        return model
