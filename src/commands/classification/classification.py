from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from typing import Any

from src.commands.classification.clas_utils import generic_classification


def naivebayes(X: np.ndarray, y: np.ndarray, *args, **kwargs
                       ) -> tuple[np.ndarray,  np.ndarray[Any, Any]]:
    """
    Perform Naive Bayes classification with k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.

    Returns:
        tuple[np.ndarray, np.ndarray[Any, Any]]: 
            - Predictions from the cross-validation.
            - Class priors of the final model.
    """
    
    predictions, scores, final_model = generic_classification(GaussianNB(), X, y, **kwargs)

    model_priors: np.ndarray[Any, Any] = final_model.class_prior_
    
    return np.array(predictions), model_priors

def mlpclas(X: np.ndarray, y: np.ndarray, *args, **kwargs
                       ) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Perform MLP classification with k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        **kwargs: Additional keyword arguments for k-fold cross-validation.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Predictions from the cross-validation.
            - Intercepts of the final model.
            - Weights of the final model.
    """
    predictions, scores, final_model = generic_classification(MLPClassifier(), X, y, *args, **kwargs)

    model_weights: np.ndarray = final_model.coefs_
    
    return np.array(predictions), final_model.intercepts_, model_weights

def logisticreg(X: np.ndarray, y: np.ndarray, *args, **kwargs
                       ) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Perform logistic regression with k-fold cross-validation.

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
    predictions, scores, final_model = generic_classification(LogisticRegression(), X, y, *args, **kwargs)

    model_weights: np.ndarray = final_model.coef_
    
    return np.array(predictions), final_model.intercept_, model_weights