"""Module to create preprocessors"""

import sys
import os

import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.logger import logging
from src.exception import CustomException


def prepare_text_df(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare sentiment dataframe for model training.

    Args:
        sentiment_df (pd.DataFrame): Input dataframe containing 'text_pt' and 'label' columns.

    Returns:
        pd.DataFrame: Cleaned dataframe with:
            - text_pt : str (processed text without extra spaces)
            - label : int (converted to integer type)
    """
    df = sentiment_df[["text_pt", "label"]].copy()
    df["text_pt"] = (
        df["text_pt"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df = df.replace({"text_pt": {"": pd.NA}}).dropna(subset=["text_pt", "label"])
    df["label"] = df["label"].astype(int)
    return df


class Preprocessor:
    """Class to transform data with preprocessor."""

    def __init__(
        self, df: pd.DataFrame, target_col: str, test_size: float, random_state: int
    ):
        self.preprocessor = None
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

        self.numeric_features = [
            "n_distinct_categories",
            "n_sellers",
            "n_items",
            "payment_value",
            "total_freight_value",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]
        self.categorical_features = [
            "payment_type",
            "customer_state",
            "top20_categories_and_others",
            "installments_bins",
        ]

    def create_preprocessor(self):
        """Fucntion to create preprocessor to transform data."""

        logging.info("Function to create preprocessor has started.")

        numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop="if_binary", sparse_output=False
                    ),
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features),
            ]
        )

    def split_fit_transform(self):
        """Function to split and transform data with preprocessor.

        Returns:
            list: list of splited data into X/y train/test/val.
        """

        logging.info("Function to split and transform data has started.")

        self.create_preprocessor()

        X = self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=self.random_state,
        )

        try:
            X_train_transformed = self.preprocessor.fit_transform(X_train)
            X_val_transformed = self.preprocessor.transform(X_val)
            X_test_transformed = self.preprocessor.transform(X_test)

            cat_encoder = self.preprocessor.named_transformers_["cat"]["onehot"]
            n_num_features = len(self.numeric_features)
            categorical_indexes = list(
                range(
                    n_num_features,
                    n_num_features + len(cat_encoder.get_feature_names_out()),
                )
            )

            sm = SMOTENC(
                categorical_features=categorical_indexes,
                random_state=self.random_state,
                sampling_strategy="auto",
            )

            X_train_res, y_train_res = sm.fit_resample(X_train_transformed, y_train)

            return (
                X_train_res,
                X_val_transformed,
                X_test_transformed,
                y_train_res,
                y_val,
                y_test,
            )

        except Exception as e:
            logging.info(
                "Function to split and transform data has encountered a problem."
            )
            raise CustomException(e, sys) from e


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis using BERTimbau."""

    def __init__(
        self, texts: list[str], labels: list[int], tokenizer, max_len: int = 128
    ):
        """Initialize SentimentDataset.

        Args:
            texts (list[str]): List of input texts.
            labels (list[int]): List of sentiment labels corresponding to each text.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer from HuggingFace (e.g., BERTimbau tokenizer).
            max_len (int, optional): Maximum sequence length for tokenization. Defaults to 128.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieve one sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - "input_ids" (torch.Tensor): Encoded token IDs.
                - "attention_mask" (torch.Tensor): Attention mask for padding.
                - "labels" (torch.Tensor): Target label as tensor.
        """
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TextPreprocessor:
    """Class to handle preprocessing for BERTimbau sentiment analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str,
        target_col: str,
        test_size: float = 0.2,
        random_state: int = 42,
        max_len: int = 128,
    ):
        """Initialize TextPreprocessor.

        Args:
            df (pd.DataFrame): Input dataframe with text and labels.
            text_col (str): Name of the column containing text.
            target_col (str): Name of the column containing labels.
            test_size (float, optional): Proportion of dataset for validation + test split. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            max_len (int, optional): Maximum sequence length for tokenization. Defaults to 128.
        """
        self.df = df
        self.text_col = text_col
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased"
        )

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test sets.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes.
        """
        logging.info("Function to split data for BERT has started.")

        train_df, temp_df = train_test_split(
            self.df,
            test_size=self.test_size,
            stratify=self.df[self.target_col],
            random_state=self.random_state,
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            stratify=temp_df[self.target_col],
            random_state=self.random_state,
        )

        return train_df, val_df, test_df

    def create_datasets(
        self, batch_size: int = 16
    ) -> tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
        """Create PyTorch DataLoaders for train/val/test.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 16.

        Returns:
            tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]: DataLoaders for train, validation, and test sets, and the tokenizer.
        """
        train_df, val_df, test_df = self.split_data()

        logging.info("Function to create DataLoaders has started.")

        train_dataset = SentimentDataset(
            texts=train_df[self.text_col].to_numpy(),
            labels=train_df[self.target_col].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )

        val_dataset = SentimentDataset(
            texts=val_df[self.text_col].to_numpy(),
            labels=val_df[self.target_col].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )

        test_dataset = SentimentDataset(
            texts=test_df[self.text_col].to_numpy(),
            labels=test_df[self.target_col].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.max_len,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        return train_loader, val_loader, test_loader, self.tokenizer
