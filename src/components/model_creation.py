"""Module to create models"""

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig
from keras import layers, models, optimizers
from keras.losses import BinaryFocalCrossentropy
from prophet import Prophet

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


class ProphetModel:
    """Class responsible for creating Prophet model."""

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "additive",
        changepoint_prior_scale: float = 0.3,
        seasonality_prior_scale: float = 10.0,
    ):
        """Initialize Prophet model with given params.

        Args:
            yearly_seasonality (bool, optional): Whether to include yearly seasonality in the model. Default is True.
            weekly_seasonality (bool, optional): Whether to include weekly seasonality in the model. Default is True.
            daily_seasonality (bool, optional): Whether to include daily seasonality in the model. Default is False.
            seasonality_mode (str, optional): Seasonality interaction mode, "additive" or "multiplicative". Defaults to "multiplicative".
            changepoint_prior_scale (float, optional): Flexibility of the trend; higher values allow the trend to change faster. Defaults to 0.05.
            seasonality_prior_scale (float, optional): Strength of the seasonality model. Higher values make seasonality fit stronger. Defaults to 10.0.
        """

        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
        )

        logging.info("Prophet model creation initialized.")

    def get_model(self):
        """Return Prophet model instance.

        Returns:
            Prophet: The initialized Prophet model.
        """

        return self.model


class BertModel(nn.Module):
    """BERT-based sentiment classifier using BERTimbau."""

    def __init__(
        self,
        pretrained_model_name: str = "neuralmind/bert-base-portuguese-cased",
        num_labels: int = 3,
    ) -> None:
        """Initialize BertModel.

        Args:
            pretrained_model_name (str, optional): Name of the pretrained HuggingFace model. Defaults to "neuralmind/bert-base-portuguese-cased".
            num_labels (int, optional): Number of sentiment classes. Defaults to 3 (negative, neutral, positive).
        """
        super(BertModel, self).__init__()

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, config=self.config, use_safetensors=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs with shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Tensor indicating padded elements (1 = real token, 0 = padding).
            labels (torch.Tensor | None, optional): Tensor of ground-truth labels with shape (batch_size,). Defaults to None.

        Returns:
            dict: Dictionary containing:
                - loss (torch.Tensor, optional): Training loss if labels provided.
                - logits (torch.Tensor): Raw output logits of shape (batch_size, num_labels).
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
