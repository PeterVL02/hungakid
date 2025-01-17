from src.MLOps.visuals.pca.pca import plot_explained_var, plot_pca, barplot_pcs

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Callable, Any
from sklearn.decomposition import PCA

PlotCommandFn = Callable[..., Any]

class Plotter:
    """Class to plot various types of plots."""
    def __init__(self):
        self.plot_cnt: int = 0
        self.plot_funcs: list[Callable] = []
        self.plot_data: list[dict[str, Any]] = []
        
    def plot_hist(self, series: np.ndarray, label: str, show: bool = True) -> None:
        """Plots a histogram of the given series."""
        plt.hist(series, color='g')
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {label}")
        if show:
            plt.show()
            
    def _plot_hist_wraps(self, series: np.ndarray, label: str) -> None:
        self.plot_data.append({"series": series, "label": label, "show": False})
        self.plot_funcs.append(self.plot_hist)
        self.plot_cnt += 1
        
    def plot_boxpl(self, series: np.ndarray, label: str, show: bool = True) -> None:
        """Plots a boxplot of the given series."""
        plt.boxplot(series)
        plt.xlabel(label)
        plt.title(f"Boxplot of {label}")
        if show:
            plt.show()
    
    def _plot_boxpl_wraps(self, series: np.ndarray, label: str) -> None:
        self.plot_data.append({"series": series, "label": label, "show": False})
        self.plot_funcs.append(self.plot_boxpl)
        self.plot_cnt += 1
        
    def plot_scatter(self, series: list[np.ndarray], labels: list[str], show: bool = True) -> None:
        """Plots a scatter plot of the given series."""
        plt.scatter(series[0], series[1], color = 'g')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        plt.title(f"Scatter plot of {labels[0]} vs {labels[1]}")
        if show:
            plt.show()
            
    def _plot_scatter_wraps(self, series: list[np.ndarray], labels: list[str]) -> None:
        self.plot_data.append({
            "series": series,
            "labels": labels,
            "show": False
        })
        self.plot_funcs.append(self.plot_scatter)
        self.plot_cnt += 1
        
    def show(self, *args, **kwargs) -> None:
        """Show the plots in as close to a square layout as possible."""
        plt.ion()
        if not self.plot_cnt:
            plt.ioff()
            raise ValueError("No plots to show.")
        
        rows = int(math.ceil(math.sqrt(self.plot_cnt)))
        cols = int(math.ceil(self.plot_cnt / rows))
        
        for i, (func, data) in enumerate(zip(self.plot_funcs, self.plot_data)):
            plt.subplot(rows, cols, i + 1)
            func(**data)
        
        plt.tight_layout()
        plt.show()
        self.plot_cnt = 0
        self.plot_funcs = []
        self.plot_data = []
        plt.ioff()
        
    def close(self, *args, **kwargs) -> None:
        """Close the plot."""
        plt.close()
        plt.ioff()
    
    def pca_plot(self, pca: PCA, X: np.ndarray, y: np.ndarray, task: str,
                 cols: list[str], show: bool = False) -> None:
        """Plots the PCA visualization of the input data."""
        self.plot_data.append({'pca': pca, 'X': X, 'y': y, 'task': task})
        self.plot_funcs.append(plot_pca)
        self.plot_data.append({'pca' : pca})
        self.plot_funcs.append(plot_explained_var)
        self.plot_data.append({'pca' : pca, 'cols': cols})
        self.plot_funcs.append(barplot_pcs)
        
        self.plot_cnt += 3
        if show:
            self.show()
        
        
    def plot_interact(self, cmd: str, series: np.ndarray | list[np.ndarray] | None = None, label: str | list[str] | None = None, show: bool = False) -> None:
        if label is not None and series is not None:
            if isinstance(label, list):
                assert len(series) == len(label), "Length of series and labels must be same."
                
        if not cmd in {'show', 'close'}:
            assert bool(label), f"Label must be provided for cmd {cmd}."

        if show:
            PLOTCOMMANDS: dict[str, PlotCommandFn] = {
                "hist": self.plot_hist,
                "box": self.plot_boxpl,
                "scatter": self.plot_scatter,
                'close': self.close,
                'show': self.show
            }
        else:
            PLOTCOMMANDS: dict[str, PlotCommandFn] = {
                "hist":    self._plot_hist_wraps,
                "box":     self._plot_boxpl_wraps,
                "scatter": self._plot_scatter_wraps,
                'close':   self.close,
                'show':    self.show
            }
        plt.ion()
        try:
            PLOTCOMMANDS[cmd](series, label)
        except KeyError:
            raise ValueError(f"Invalid command {cmd}.")
    
if __name__ == "__main__":
    pltr = Plotter()
    pltr._plot_hist_wraps(np.random.randn(100), "Random1")
    pltr._plot_boxpl_wraps(np.random.randn(100), "Random1")
    pltr._plot_scatter_wraps([np.random.randn(100), np.random.randn(100)], ["Random1", "Random1.5"])
    pltr._plot_hist_wraps(np.random.randn(10000), "Random2")
    pltr.show()
    pltr.close()