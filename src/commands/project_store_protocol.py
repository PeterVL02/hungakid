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
    
    def load_project_from_file(self, alias: str) -> CLIResult: ...
    
    def get_current_project(self) -> ShellProject: ...