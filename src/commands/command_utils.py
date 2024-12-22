from enum import StrEnum
import numpy as np
from pandas import DataFrame, get_dummies
from pandas.api.types import is_string_dtype
from pandas import concat, get_dummies

class MlModel(StrEnum):
    NAIVE_BAYES = "naive_bayes"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    K_MEANS = "k_means"
    KNN = "knn"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    MLPREG = "mlpreg"
    MLPCLASS = "mlpclass"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"

class ProjectType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

def convert_to_type(type_str: str) -> ProjectType:
    try:
        return ProjectType[type_str.upper()]
    except ValueError:
        raise ValueError(f"Invalid project type: {type_str}")
    
def convert_to_ml_type(type_str: str) -> MlModel:
    try:
        return MlModel[type_str.upper()]
    except ValueError:
        raise ValueError(f"Invalid ML model: {type_str}")
    
def print_unused(*args, **kwargs) -> str:
    return "Unused args: " + str(args) + str(kwargs)

def regression_pipeline(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
