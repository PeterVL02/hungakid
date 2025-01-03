import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Callable, Any

PlotCommandFn = Callable[..., Any]

class Plotter:
    """Class to plot various types of plots."""
    def __init__(self):
        self.plot_cnt: int = 0
        self.plot_funcs: list[Callable] = []
        self.plot_data: list[dict[str, np.ndarray | str | int | list[str] | list[np.ndarray]]] = []
        
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
        
    def show(self) -> None:
        """Show the plots in as close to a square layout as possible."""
        if not self.plot_cnt:
            plt.show()
            return
        
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
        
    def plot_interact(self, cmd: str, series: np.ndarray | list[np.ndarray] | None = None, label: str | list[str] | None = None, show: bool = False) -> None:
        if label is not None and series is not None:
            if isinstance(label, list):
                assert len(series) == len(label)
                
        if cmd != 'show':
            assert bool(label), f"Label must be provided for cmd {cmd}."

        if show:
            PLOTCOMMANDS: dict[str, PlotCommandFn] = {
                "hist": self.plot_hist,
                "box": self.plot_boxpl,
                "scatter": self.plot_scatter,
                'close': self.close
            }
        else:
            PLOTCOMMANDS: dict[str, PlotCommandFn] = {
                "hist":    self._plot_hist_wraps,
                "box":     self._plot_boxpl_wraps,
                "scatter": self._plot_scatter_wraps,
                'close':   self.close
            }
        plt.ion()
        PLOTCOMMANDS[cmd](series, label)
    
if __name__ == "__main__":
    pltr = Plotter()
    pltr._plot_hist_wraps(np.random.randn(100), "Random1")
    pltr._plot_boxpl_wraps(np.random.randn(100), "Random1")
    pltr._plot_scatter_wraps([np.random.randn(100), np.random.randn(100)], ["Random1", "Random1.5"])
    pltr._plot_hist_wraps(np.random.randn(10000), "Random2")
    pltr.show()
    pltr.close()