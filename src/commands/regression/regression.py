from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import numpy as np
from typing import Any

from src.commands.regression.reg_utils import generic_regression


def linreg(X: np.ndarray, y: np.ndarray, *args, **kwargs
                       ) -> tuple[np.ndarray, float, np.ndarray[Any, Any]]:
    """
    Perform linear regression with k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        **kwargs: Additional keyword arguments for k-fold cross-validation.

    Returns:
        tuple[np.ndarray, float, np.ndarray[Any, Any]]: 
            - Predictions from the cross-validation.
            - Intercept of the final model.
            - Coefficients of the final model.
    """
    
    predictions, scores, final_model = generic_regression(LinearRegression(), X, y, **kwargs)

    model_weights: np.ndarray[Any, Any] = final_model.coef_
    
    return np.array(predictions), float(final_model.intercept_), model_weights

def mlpreg(X: np.ndarray, y: np.ndarray, *args, **kwargs
                       ) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Perform linear regression with k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        **kwargs: Additional keyword arguments for k-fold cross-validation.

    Returns:
        tuple[np.ndarray, float, np.ndarray]: 
            - Predictions from the cross-validation.
            - Intercept of the final model.
            - Coefficients of the final model.
    """
    predictions, scores, final_model = generic_regression(MLPRegressor(), X, y, *args, **kwargs)

    model_weights: np.ndarray = final_model.coefs_
    
    return np.array(predictions), final_model.intercepts_, model_weights
