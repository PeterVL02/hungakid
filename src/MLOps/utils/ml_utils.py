from sklearn.model_selection import KFold
import numpy as np
from pandas import DataFrame, get_dummies, concat
from pandas.api.types import is_string_dtype
import numpy as np
from typing import Any
from tqdm import tqdm
from src.MLOps.utils.base import BaseEstimator

def k_fold_cross(X: np.ndarray, y: np.ndarray, shuffle: bool, n_splits: int, random_state: int | None) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Perform K-Fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix. Fetched from the current project.
        y (np.ndarray): Target vector. Fetched from the current project.
        shuffle (bool): Whether to shuffle the data before splitting. Fetched from kwargs.
        n_splits (int): Number of folds. Fetched from kwargs.
        random_state (int | None): Random seed for reproducibility. Fetched from kwargs.

    Returns:
        tuple[np.ndarray, np.ndarray]: Indices for training and validation splits.
    """
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    return list(kf.split(X, y))

def standard_pipeline(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize the data using the mean and standard deviation of the training set.

    Args:
        X_train (np.ndarray): Training data.
        X_test (np.ndarray): Testing data.

    Returns:
        tuple[np.ndarray, np.ndarray]: Standardized training and testing data.
    """
    mu, sig = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - mu) / sig
    X_test =   (X_test - mu) / sig
    return X_train, X_test

def onehot_encode_string_columns(df: DataFrame, ignore_columns: list[str]) -> DataFrame:
    """
    Detects columns containing strings in `df` and one-hot encodes them.
    Returns a new DataFrame with the transformations applied.
    """
    df_encoded = df.copy()

    string_cols = []
    for col in df_encoded.columns:
        if col in ignore_columns:
            continue
        if is_string_dtype(df_encoded[col]):
            non_null_values = df_encoded[col].dropna()
            if all(isinstance(val, str) for val in non_null_values):
                string_cols.append(col)
    
    for col in string_cols:
        dummies = get_dummies(df_encoded[col], prefix=col)
        df_encoded.drop(col, axis=1, inplace=True)
        df_encoded = concat([df_encoded, dummies], axis=1)
    
    return df_encoded





def generic_ml(mlmodel: BaseEstimator, X: np.ndarray, y: np.ndarray, *args, **kwargs) -> tuple[np.ndarray, list[float], Any]:
    """
    Perform k-fold cross-validation on a given machine learning model and return predictions, scores, and the final trained model.
    Args:
        mlmodel (BaseEstimator): The machine learning model to be trained and evaluated.
        X (np.ndarray): The input features for the model.
        y (np.ndarray): The target values for the model.
        *args: Additional positional arguments to pass to the model's __init__ method.
        **kwargs: Additional keyword arguments to pass to the model's __init__ method and to control cross-validation.
            n_splits (int, optional): Number of splits for k-fold cross-validation. Default is 10.
            shuffle (bool, optional): Whether to shuffle the data before splitting into batches. Default is False.
            random_state (int, optional): Random seed for shuffling. Default is 42 if shuffle is True, otherwise None.
    Returns:
        tuple[np.ndarray, list[float], Any]: A tuple containing:
            - np.ndarray: The predictions made by the model during cross-validation.
            - list[float]: The scores obtained during cross-validation.
            - Any: The final trained model.
    """
                       
    
    
    predictions: list[float] = []
    scores: list[float] = []
    n_splits: int = kwargs.pop('n_splits', 10)
    shuffle: bool  = kwargs.pop('shuffle', False)
    random_state: int | None = kwargs.pop('random_state', 42) if shuffle else None
    for train_index, test_index in tqdm(k_fold_cross(X, y, n_splits=n_splits, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle), desc=f'Cross Validating {mlmodel.__class__.__name__}'):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = standard_pipeline(X_train, X_test)
        model = mlmodel
        model.__init__(**kwargs)
        model.fit(X_train, y_train)
        predictions.extend(model.predict(X_test))
        scores.append(float(model.score(X_test, y_test)))
    
    final_model = mlmodel
    final_model.__init__(**kwargs)
    X, _ = standard_pipeline(X, X)
    final_model.fit(X, y)

    return np.array(predictions), scores, final_model