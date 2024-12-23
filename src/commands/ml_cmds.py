from src.MLOps.regression.regression import linreg as linreg_impl, mlpreg as mlpreg_impl
from src.MLOps.classification.classification import (naivebayes as naivebayes_impl, mlpclas as mlpclas_impl, 
                                                    logisticreg as logisticreg_impl, decisiontree as 
                                                    decisiontree_impl, randomforest as randomforest_impl, 
                                                    gradientboosting as gradientboosting_impl
                                                        )
from src.commands.command_utils import MlModel
from src.commands.project_store_protocol import Model

import numpy as np

def retrieve_X_y(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the feature matrix `X` and target vector `y` from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix and target vector.
    """
    if model.projects[model.current_project].X is None or  model.projects[model.current_project].y is None:
        raise ValueError("No data to fit model to. Please load data first.")
    return model.projects[model.current_project].X, model.projects[model.current_project].y # type: ignore (sorry mypy, we checked above)

def linreg(model: Model, *args, **kwargs) -> str:
    """
    Fits a linear regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

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
    X, y = retrieve_X_y(model = model)

    predictions, intercept, weights = mlpreg_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.MLPREG, predictions = predictions, params = {})

def naivebayes(model: Model, *args, **kwargs) -> str:
    """
    Fits a naive bayes classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

    predictions, model_priors = naivebayes_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.NAIVE_BAYES, predictions = predictions, params = {}, model_priors = model_priors)

def mlpclas(model: Model, *args, **kwargs) -> str:
    """
    Fits a multi-layer perceptron classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

    predictions, intercept, weights = mlpclas_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.MLPCLASS, predictions = predictions, params = {})

def logisticreg(model: Model, *args, **kwargs) -> str:
    """
    Fits a logistic regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

    predictions, intercept, weights = logisticreg_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.LOGISTIC_REGRESSION, predictions = predictions, params = {}, intercept = intercept, weights = weights)

def decisiontree(model: Model, *args, **kwargs) -> str:
    """
    Fits a decision tree classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

    predictions, model_importances, final_model = decisiontree_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.DECISION_TREE, predictions = predictions, params = {}, importances = model_importances, final_model = final_model)

def randomforest(model: Model, *args, **kwargs) -> str:
    """
    Fits a random forest classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

    predictions, model_importances, final_model = randomforest_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.RANDOM_FOREST, predictions = predictions, params = {}, importances = model_importances, final_model = final_model)

def gradientboosting(model: Model, *args, **kwargs) -> str:
    """
    Fits a gradient boosting classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model)

    predictions, model_importances, final_model = gradientboosting_impl(X, y, *args, **kwargs)
    return model.log_model(MlModel.GRADIENT_BOOSTING_CLASSIFIER, predictions = predictions, params = {}, importances = model_importances, final_model = final_model)