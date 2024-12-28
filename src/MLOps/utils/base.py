"""Base class for all estimators in the project. Stand-in for the sklearn BaseEstimator class, 
where fit, predict, and score methods are not implemented, throwing of the typechecker."""

from typing import Protocol, Any
import numpy as np


class BaseEstimator(Protocol):
    """
    BaseEstimator is a protocol that defines the basic structure for machine learning estimators.
    Methods:
        fit(X, y) -> Any:
            Fits the model to the provided data.
                X: Training data.
                y: Target values.
            Returns:
                Any: The fitted model.
        predict(X) -> np.ndarray | Any:
            Predicts target values for the given data.
                X: Data to predict.
            Returns:
                np.ndarray | Any: Predicted values.
        score(X, y) -> float | Any:
            Returns the score of the model on the provided test data and labels.
                X: Test data.
                y: True labels for X.
            Returns:
                float | Any: The score of the model.
        __init__(*args, **kwargs) -> None:
            Initializes the estimator with given arguments.
        get_params() -> dict[str, int | float | str | bool]:
            Returns the parameters of the estimator.
            Returns:
                dict[str, int | float | str | bool]: Dictionary of parameter names mapped to their values.
    """

    def fit(self, X, y) -> Any:
        ...
    def predict(self, X) -> np.ndarray | Any:
        ...
    def score(self, X, y) -> float | Any:
        ...
    def __init__(self, *args, **kwargs) -> None:
        ...
    def get_params(self) -> dict[str, int | float | str | bool]:
        ...