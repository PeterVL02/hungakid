import numpy as np
from typing import Any
from tqdm import tqdm

from src.MLOps.utils.ml_utils import k_fold_cross, standard_pipeline
from src.MLOps.utils.base import BaseEstimator

def generic_regression(regressor: BaseEstimator,  X: np.ndarray, y: np.ndarray, *args, **kwargs
                       ) -> tuple[np.ndarray, list[float], Any]:
    """
    Perform linear regression with k-fold cross-validation.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        **kwargs: Additional keyword arguments for k-fold cross-validation.

    Returns:
        tuple[np.ndarray, float, np.ndarray[Any, Any]]: 
            - Predictions from the cross-validation.
            - Intercept of the final model.
            - Coefficients of the final model.
    """
    
    predictions: list[float] = []
    scores: list[float] = []
    n_splits: int = kwargs.pop('n_splits', 10)
    shuffle: bool  = kwargs.pop('shuffle', False)
    random_state: int | None = kwargs.pop('random_state', 42) if shuffle else None
    for train_index, test_index in tqdm(k_fold_cross(X, y, n_splits=n_splits, 
                                                     random_state=random_state, 
                                                     shuffle=shuffle), desc='Cross Validating'):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, X_test = standard_pipeline(X_train, X_test)
        model = regressor
        model.__init__(**kwargs)
        model.fit(X_train, y_train)
        predictions.extend(model.predict(X_test))
        scores.append(float(model.score(X_test, y_test)))
    
    final_model = regressor
    final_model.__init__(**kwargs)
    X, _ = standard_pipeline(X, X)
    final_model.fit(X, y)
    print('Generic Model Scores', np.mean(scores))

    return np.array(predictions), scores, final_model
