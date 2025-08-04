"""Common functionalities."""

import os
import sys
from typing import Any

import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    """Function to save given object in given path.

    Args:
        file_path (str): Path where to save object.
        obj (_type_): Object to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def evaluate_model(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    models: dict,
    params: dict,
) -> dict:
    """Function to evaluate model with r2 score.

    Args:
        X_train: X train data.
        y_train: y train data.
        X_test: X test data.
        y_test: y test data.
        models: Dictionary of models.
        params: Dictionaty of model paramaters to tune.

    Returns:
        dict: Dictionary of models names and r2 scores.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> Any:
    """Function to load object from given path.

    Args:
        file_path (str): Path to file.

    Returns:
        Any: Loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e
