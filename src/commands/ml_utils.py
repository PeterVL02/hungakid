from sklearn.model_selection import KFold
import numpy as np
from pandas import DataFrame, get_dummies, concat
from pandas.api.types import is_string_dtype

def k_fold_cross(X: np.ndarray, y: np.ndarray, shuffle: bool, n_splits: int, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
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
    return tuple(kf.split(X, y))

def regression_pipeline(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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

def onehot_encode_string_columns(df: DataFrame) -> DataFrame:
    """
    Detects columns containing strings in `df` and one-hot encodes them.
    Returns a new DataFrame with the transformations applied.
    """
    df_encoded = df.copy()

    string_cols = []
    for col in df_encoded.columns:
        if is_string_dtype(df_encoded[col]):
            non_null_values = df_encoded[col].dropna()
            if all(isinstance(val, str) for val in non_null_values):
                string_cols.append(col)
    
    for col in string_cols:
        dummies = get_dummies(df_encoded[col], prefix=col)
        df_encoded.drop(col, axis=1, inplace=True)
        df_encoded = concat([df_encoded, dummies], axis=1)
    
    return df_encoded
