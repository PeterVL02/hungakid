from src.commands.command_utils import ProjectType
from src.shell_project import ShellProject
from src.commands.command_utils import MlModel
from src.MLOps.utils.base import BaseEstimator
from src.cliresult import CLIResult

from typing import Protocol
import numpy as np



class Model(Protocol):
    projects: dict[str, ShellProject]
    current_project: ...
    def create(self, alias: str, type: ProjectType) -> CLIResult: ...

    def delete(self, alias: str, from_dir: bool) -> CLIResult: ...

    def list_projects(self) -> CLIResult: ...

    def set_current_project(self, alias: str) -> CLIResult: ...

    def pcp(self) -> CLIResult: ...

    def add_data(self, df_name: str, delimiter: str = ',') -> CLIResult: ...

    def read_data(self, head: int = 5) -> CLIResult: ...
    
    def list_cols(self) -> CLIResult: ...

    def make_X_y(self, target: str) -> CLIResult: ...

    def clean_data(self) -> CLIResult: ...

    def log_model(self, model_name: "MlModel", predictions: np.ndarray, params: dict[str, float | int | str], **kwargs) -> CLIResult: ...

    def summary(self) -> CLIResult: ...

    def log_predictions_from_best(self, *models: BaseEstimator, **kwargs) -> CLIResult: ...
    
    def save(self, overwrite: bool) -> CLIResult: ...
    
    def load_project_from_file(self, alias: str) -> CLIResult: ...
    
    def plot(self, cmd: str, labels: str | list[str], show: bool = False) -> CLIResult: ...
    
    def show(self) -> CLIResult: ...
    
    def stats(self) -> CLIResult: ...