from src.MLOps.regression.regression import linreg as linreg_impl, mlpreg as mlpreg_impl
from src.MLOps.classification.classification import (naivebayes as naivebayes_impl, mlpclas as mlpclas_impl, 
                                                    logisticreg as logisticreg_impl, decisiontree as 
                                                    decisiontree_impl, randomforest as randomforest_impl, 
                                                    gradientboosting as gradientboosting_impl
                                                        )
from src.commands.command_utils import MlModel
from src.commands.project_store_protocol import Model
from src.cliresult import CLIResult, chain

import numpy as np

@chain
def retrieve_X_y(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the feature matrix `X` and target vector `y` from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        tuple[np.ndarray, np.ndarray]: Feature matrix and target vector.
    """
    try:
        if model.projects[model.current_project].X is None or  model.projects[model.current_project].y is None:
            raise ValueError("No data to fit model to. Please load data first.")
    except KeyError:
        raise ValueError("No current project set.")
    return model.projects[model.current_project].X, model.projects[model.current_project].y # type: ignore (sorry mypy, we checked above)

@chain
def linreg(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a linear regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, intercept, weights = linreg_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.LINEAR_REGRESSION, predictions = predictions, params = {}, intercept = intercept, weights = weights)

@chain
def mlpreg(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a multi-layer perceptron regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, intercept, weights = mlpreg_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.MLPREG, predictions = predictions, params = {})

@chain
def naivebayes(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a naive bayes classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, model_priors = naivebayes_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.NAIVE_BAYES, predictions = predictions, params = {}, model_priors = model_priors)

@chain
def mlpclas(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a multi-layer perceptron classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, intercept, weights = mlpclas_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.MLPCLASS, predictions = predictions, params = {})

@chain
def logisticreg(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a logistic regression model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, intercept, weights = logisticreg_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.LOGISTIC_REGRESSION, predictions = predictions, params = {}, intercept = intercept, weights = weights)

@chain
def decisiontree(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a decision tree classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, model_importances, final_model = decisiontree_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.DECISION_TREE, predictions = predictions, params = {}, importances = model_importances, final_model = final_model)

@chain
def randomforest(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a random forest classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, model_importances, final_model = randomforest_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.RANDOM_FOREST, predictions = predictions, params = {}, importances = model_importances, final_model = final_model)

@chain
def gradientboosting(model: Model, *args, **kwargs) -> CLIResult:
    """
    Fits a gradient boosting classification model to the current project's data.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    X, y = retrieve_X_y(model = model).result

    predictions, model_importances, final_model = gradientboosting_impl(X, y, *args, **kwargs)
    project = model.get_current_project()
    return project.log_model(MlModel.GRADIENT_BOOSTING_CLASSIFIER, predictions = predictions, params = {}, importances = model_importances, final_model = final_model)

@chain
def log_from_best(model: Model, *args, **kwargs) -> CLIResult:
    """
    Trains, tests and logs performance from multiple models based on the project type.

    Depending on whether the current project is a classification or regression task,
    this function will log predictions using a set of predefined models suitable for 
    the task.

    Args:
        model (Model): The model object containing project details and methods for logging predictions.
        *args: Additional positional arguments to pass to the logging method.
        **kwargs: Additional keyword arguments to pass to the logging method.

    Returns:
        CLIResult: A log string containing the predictions from the best performing model.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    
    project = model.get_current_project()
    
    if project.project_type == 'classification':
        return project.log_predictions_from_best(GaussianNB(), MLPClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), *args, **kwargs)
    return project.log_predictions_from_best(LinearRegression(), MLPRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), *args, **kwargs)