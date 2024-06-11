import logging

import pandas as pd 
from zenml import step

from model.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train the model on the ingested data.

    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e: 
        logging.error("Error in training model: {}".format(e))
        raise e
    
