from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def scale(X: np.ndarray) -> np.ndarray:
    """Scales the input data."""
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def pca_fit(X: np.ndarray) -> PCA:
    """Performs PCA transformation on the input data."""
    scaled = scale(X)
    pca =  PCA()
    return pca.fit(scaled)

def plot_pca(pca: PCA, X: np.ndarray,  y:np.ndarray, task: str) -> None:
    """
    Plots the PCA visualization of the input data.
    
    Parameters
    ----------
    pca : PCA object (fitted)
    
    X : np.ndarray
        Input data.
        
    y : np.ndarray
        Target data.
    
    task : str {'classification', 'regression'}
    
    Returns
    -------
    None"""

    X_pca = pca.transform(scale(X))

    if task == 'classification':
        unique_labels = np.unique(y)

        for label in unique_labels:
            idx = (y == label)
            plt.scatter(
                X_pca[idx, 0],
                X_pca[idx, 1],
                label=str(label),  # convert to string for label
                alpha=0.7
            )

        plt.legend(title='Class Label')
        plt.title(f'PCA Visualization')

    elif task == 'regression':
        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))
        
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=y,
            cmap='viridis', 
            norm=norm,
            alpha=0.7
        )
        plt.colorbar(scatter, label='Target Value')
        plt.title(f'PCA Visualization')

    else:
        raise ValueError("task must be either 'classification' or 'regression'.")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
def plot_explained_var(pca: PCA) -> None:
    """
    Plots the explained variance of the PCA components.
    
    Parameters
    ----------
    pca : PCA object (fitted)
    
    Returns
    -------
    None"""
    plt.plot(pca.explained_variance_ratio_, marker='o', color='b', label='Individual Explained Variance')
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='r', label='Cumulative Explained Variance')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio of PCA Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
def barplot_pcs(pca: PCA, cols: list[str], n: int | None = None) -> None:
    """
    Plots the explained variance of the first n PCA components.
    
    Parameters
    ----------
    pca : PCA object (fitted)
    
    n : int
        Number of components to plot.
    
    Returns
    -------
    None"""
    if n is None:
        n = len(cols)
    
    components = pca.components_[:n]
    legendStrs = [f"PC{i+1}" for i in range(n)]
    bw = 0.2
    r = np.arange(1, len(cols) + 1)
    for i, pc in enumerate(components):
        plt.bar(r + i * bw, pc, width=bw, label=legendStrs[i])
    plt.xlabel("Features")
    plt.ylabel("Feature Weights")
    plt.title("Feature Weights of PCA Components")
    plt.xticks(r + bw, cols)
    plt.legend()
    plt.grid(True, alpha=0.3)

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    data = load_iris()
    cols = data['feature_names'] # type: ignore
    pca = pca_fit(X) # type: ignore
    plot_explained_var(pca)
    plt.show()
    plot_pca(pca, X, y, task='classification') # type: ignore
    plt.show()
    barplot_pcs(pca, cols=cols) # type: ignore
    plt.show()