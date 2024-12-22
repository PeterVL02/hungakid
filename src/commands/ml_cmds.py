from src.commands.regression import linreg as linreg_impl, mlpreg as mlpreg_impl
from src.commands.command_utils import MlModel
from src.commands.project_store_protocol import Model

import numpy as np

def linreg(model: Model, *args, **kwargs) -> str:
    """
    Fits a linear regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    if model.projects[model.current_project].X is None or  model.projects[model.current_project].y is None:
        raise ValueError("No data to fit model to. Please load data first.")
    X: np.ndarray = model.projects[model.current_project].X
    y: np.ndarray = model.projects[model.current_project].y

    predictions, intercept, weights = linreg_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.LINEAR_REGRESSION, predictions = predictions, params = {}, intercept = intercept, weights = weights)

def mlpreg(model: Model, *args, **kwargs) -> str:
    """
    Fits a multi-layer perceptron regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    if model.projects[model.current_project].X is None or  model.projects[model.current_project].y is None:
        raise ValueError("No data to fit model to. Please load data first.")
    X: np.ndarray = model.projects[model.current_project].X
    y: np.ndarray = model.projects[model.current_project].y

    predictions, intercept, weights = mlpreg_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.MLPREG, predictions = predictions, params = {})