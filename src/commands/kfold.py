from sklearn.model_selection import KFold
import numpy as np

def k_fold_cross(X: np.ndarray, y: np.ndarray, shuffle: bool, n_splits: int, random_state: int | None) -> tuple[np.ndarray, np.ndarray]:
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
    return tuple(kf.split(X, y))