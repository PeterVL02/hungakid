from src.MLOps.utils.base import BaseEstimator
from src.MLOps.classification.clas_utils import generic_classification
from src.MLOps.regression.reg_utils import generic_regression
from src.cliexception import chain, add_warning, add_note

import numpy as np
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from sklearn.utils._testing import ignore_warnings
import re
import os
from colorama import Style, Fore



def infer_param_grid(model: BaseEstimator, n_values: int = 3) -> dict[str, list[float | int]]:
    """
    Infers a basic parameter grid for an sklearn model by
    examining the model's default parameters. Generates
    'n_values' values for each numeric parameter.
    
    :param model: A scikit-learn estimator.
    :param n_values: How many values to generate per parameter.
    :return: A dictionary of parameter names mapped to lists of candidate values.
    """
    tunable_dir = r'src\MLOps\tunables'
    files = os.listdir(tunable_dir)
    for file in files:
        if re.match(model.__class__.__name__, file):
            with open(tunable_dir + '/' +  file, 'r') as f:
                param_names = f.read().split('\n')
                model_params = model.get_params()
                params = {param: model_params[param] for param in param_names}
            break
    else:
        params = model.get_params()
    grid = {}

    for param, default in params.items():
        if isinstance(default, bool):
            if param == 'verbose':
                grid[param] = [False]
            else:
                grid[param] = [True, False]

        elif isinstance(default, int) and default > 0:
            if n_values == 1:
                grid[param] = [default]
            elif param == 'random_state':    
                grid[param] = [42]
            elif param == 'max_iter':
                start = min(default, 500)
                end = default * 3
                vals = np.unique(
                    np.linspace(start, end, n_values, dtype=int)
                ).tolist()
                grid[param] = vals
            else:
                start = max(1, default // 2)
                end = default * 2
                vals = np.unique(
                    np.linspace(start, end, n_values, dtype=int)
                ).tolist()
                grid[param] = vals

        elif isinstance(default, float) and 0 < default < 1:
            if n_values == 1 or param == 'tol':
                grid[param] = [default]
            else:
                start = max(1e-9, default / 2)
                end = min(1.0, default * 2)
                
                vals = np.unique(
                    np.linspace(start, end, n_values)
                ).tolist()
                grid[param] = vals

        else:
            if param == 'hidden_layer_sizes':
                start = 100
                end = 500
                vals = np.unique(
                    np.linspace(start, end, n_values, dtype=int)
                ).tolist()
                grid[param] = [(val,) for val in vals]

    return grid


def make_model_grids(*models: BaseEstimator) -> dict[str, dict[str, list[int | float]]]:
    return {model.__class__.__name__: infer_param_grid(model) for model in models}

@ignore_warnings() # type: ignore
def tune_hyperparameters(model: BaseEstimator, X: np.ndarray, y: np.ndarray, param_grid: dict[str, list[float | int]], cv: int = 10) -> dict[str, float | int | str]:
    """
    Tune hyperparameters for a given model using GridSearchCV.
    
    :param model: A scikit-learn estimator.
    :param X: Feature matrix.
    :param y: Target vector.
    :param param_grid: A dictionary of parameter names mapped to lists of candidate values.
    :param cv: Number of cross-validation folds.
    :return: A fitted model with the best hyperparameters.
    """
    if 'MLP' in model.__class__.__name__: n_jobs = 1 ## Because ConvergenceWarning is raised infinetely many times
    else: n_jobs = -1
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, verbose=0) # type: ignore
    grid_search.fit(X, y)
    return grid_search.best_estimator_.get_params()

def tune_models(*models: BaseEstimator, X: np.ndarray, y: np.ndarray, cv: int = 10, n_values: int = 3) -> list[tuple[BaseEstimator, dict[str, float | int | str]]]:
    """
    Tune hyperparameters for a list of models using GridSearchCV.
    
    :param models: A list of scikit-learn estimators.
    :param X: Feature matrix.
    :param y: Target vector.
    :param cv: Number of cross-validation folds.
    :return: A dictionary of model names mapped to fitted models with the best hyperparameters.
    """
    
    params: list[dict[str, float | int | str]] = []
    for model in tqdm(models, desc="Tuning models"):
        param_grid = infer_param_grid(model, n_values=n_values)
        params.append(tune_hyperparameters(model, X, y, param_grid, cv))
    
    return list(zip(models, params))

@chain
@ignore_warnings()
def log_predictions_from_best(*models: BaseEstimator, project: "ShellProject",  cv: int = 10, n_values: int = 3) -> None: # type: ignore to avoid circular import #TODO fix it
    """
    Get predictions from the best hyperparameters for a list of models using GridSearchCV.
    
    :param models: A list of scikit-learn estimators.
    :param X: Feature matrix.
    :param y: Target vector.
    :param cv: Number of cross-validation folds.
    :return: A dictionary of model names mapped to fitted models with the best hyperparameters.
    """
    type_ = project.project_type
    X, y = project.X, project.y

    data = tune_models(*models, X = X, y = y, cv = cv, n_values = n_values)
    if type_ == 'classification':
        for model, params in tqdm(data, desc=f"Getting predictions from model"):
            try:
                preds = generic_classification(model, X, y, **params)[0]
                project.log_model(model.__class__.__name__, preds, params)
            except RuntimeError as e:
                add_warning(project, f"Model {model.__class__.__name__} failed. Skipping...")

    elif type_ == 'regression':
        for model, params in tqdm(data, desc=f"Getting predictions from model"):
            try:
                preds = generic_regression(model, X, y, **params)[0]
                project.log_model(model.__class__.__name__, preds, params)
            except RuntimeError as e:
                add_warning(project, f"Model {model.__class__.__name__} failed. Skipping...")

def _main() -> None:
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd

    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    

    df = pd.read_csv('data/Iris.csv')
    X = df.drop(['Species'], axis=1).values
    y = np.array(df['Species'].values)

    blobs_par = infer_param_grid(MLPRegressor())
    for key, value in blobs_par.items():
        print(f"{key}: {value}")
    print()
    blobs_par = infer_param_grid(LinearRegression())
    for key, value in blobs_par.items():
        print(f"{key}: {value}")


    blobs = tune_models(
        GaussianNB(), MLPClassifier(), LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()
        , X = X, y = y, cv = 5, n_values=2
    )
    
    for model, params in blobs:
        print(f"{model.__class__.__name__}:\n{params}\n")

if __name__ == "__main__":
    _main()