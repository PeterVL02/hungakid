from enum import StrEnum


class MlModel(StrEnum):
    NAIVE_BAYES = "naive_bayes"
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    GRADIENT_BOOSTING_CLASSIFIER = "gradient_boosting_classifier"
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
    """
    Convert a string to a ProjectType enum.

    Args:
        type_str (str): The project type as a string.

    Raises:
        ValueError: If the provided string does not match any ProjectType.

    Returns:
        ProjectType: The corresponding ProjectType enum.
    """
    try:
        return ProjectType[type_str.upper()]
    except ValueError:
        raise ValueError(f"Invalid project type: {type_str}")
    
def convert_to_ml_type(type_str: str) -> MlModel:
    """
    Convert a string to an MlModel enum.

    Args:
        type_str (str): The ML model type as a string.

    Raises:
        ValueError: If the provided string does not match any MlModel.

    Returns:
        MlModel: The corresponding MlModel enum.
    """
    try:
        return MlModel[type_str.upper()]
    except ValueError:
        raise ValueError(f"Invalid ML model: {type_str}")
