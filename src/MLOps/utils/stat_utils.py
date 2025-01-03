from scipy.stats import chi2
import numpy as np

def mse_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray, p: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Computes a confidence interval for the MSE of a regression model 
    under OLS assumptions using the chi-square distribution.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth (observed) values of shape (n,).
    y_pred : np.ndarray
        Predicted values of shape (n,).
    p : int
        Number of parameters in the model (including intercept).
    alpha : float, optional
        Significance level for the (1 - alpha) confidence interval. 
        Default is 0.05, which yields a 95% CI.

    Returns
    -------
    mse : float
        The observed MSE = SSE / (n - p).
    ci_lower : float
        Lower bound for the MSE at the (1 - alpha) confidence level.
    ci_upper : float
        Upper bound for the MSE at the (1 - alpha) confidence level.
    """
    n = len(y_true)
    dof = n - p  # degrees of freedom in OLS for error variance
    if dof <= 0:
        raise ValueError("Degrees of freedom (n - p) must be > 0.")
    
    sse = np.sum((y_true - y_pred)**2)
    
    mse = sse / dof
    
    chi2_lower = chi2.ppf(1 - alpha/2, dof)
    chi2_upper = chi2.ppf(alpha/2, dof)
    
    ci_lower = dof * mse / chi2_lower
    ci_upper = dof * mse / chi2_upper
    
    return float(mse), float(ci_lower), float(ci_upper)

import numpy as np
from scipy.stats import norm, beta

def accuracy_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05
) -> tuple[float, float, float]:
    """
    Computes a confidence interval for classification accuracy (a proportion).
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels of shape (n,).
    y_pred : np.ndarray
        Predicted labels of shape (n,).
    alpha : float, optional
        Significance level for the (1 - alpha) confidence interval.
        Default is 0.05 (95% CI).
        
    Returns
    -------
    accuracy : float
        The observed accuracy = (# correct) / n.
    ci_lower : float
        Lower bound of the (1 - alpha) confidence interval.
    ci_upper : float
        Upper bound of the (1 - alpha) confidence interval.
    """
    n = len(y_true)
    if n == 0:
        raise ValueError("y_true cannot be empty.")
    
    correct = np.sum(y_true == y_pred)
    accuracy = correct / n

    z = norm.ppf(1 - alpha / 2)
    
    se = np.sqrt(accuracy * (1 - accuracy) / n)
    
    ci_lower = accuracy - z * se
    ci_upper = accuracy + z * se
        
    
    return accuracy, ci_lower, ci_upper

