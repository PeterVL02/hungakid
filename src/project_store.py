from src.commands.project_store_protocol import Model
from src.shell_project import ShellProject, ProjectType
from src.commands.command_utils import MlModel, ProjectType

from pandas import DataFrame
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ProjectStore(Model):
    projects: dict[str, ShellProject] = field(default_factory=dict)
    current_project: str | None  = None

    def create(self, alias: str, type: ProjectType) -> str:
        if alias in self.projects:
            raise ValueError(f"Project {alias} already exists")
        
        self.projects[alias] = ShellProject(project_type=type, project_name=alias)
        self.set_current_project(alias)
        return f'Project created successfully. {alias} is now the current project.'
        
    def delete(self, alias: str) -> str:
        if alias not in self.projects:
            raise ValueError(f"Project {alias} does not exist")

        del self.projects[alias]
        if self.current_project == alias:
            self.current_project = None
        return f"Project {alias} deleted successfully. Current project is {self.current_project}."

    def list_projects(self) -> list[str]:
        return list(self.projects.keys())
    
    def set_current_project(self, alias: str) -> str:
        if alias not in self.projects:
            raise ValueError(f"Project {alias} does not exist")
        
        self.current_project = alias
        return f"Current project set to {alias}"

    def pcp(self) -> ShellProject | str:
        if not self.current_project:
            return "No current project set"
        
        return self.projects[self.current_project]
    
    def add_data(self, df_name: str) -> str:
        if not self.current_project:
            raise ValueError("No current project set")
        
        return self.projects[self.current_project].add_df(df_name)

    def read_data(self, head: int = 5) -> DataFrame:
        if not self.current_project:
            raise ValueError("No current project set")
        
        return self.projects[self.current_project].read_data(head)
    
    def make_X_y(self, target: str) -> str:
        if not self.current_project:
            raise ValueError("No current project set")
        
        return self.projects[self.current_project].make_X_y(target)

    def clean_data(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set")
        
        return self.projects[self.current_project].clean_data()

    def log_model(self, model_name: MlModel, predictions: np.ndarray, params: dict[str, float], **kwargs) -> str:
        if kwargs:
            print("kwargs: ", kwargs)
        if not self.current_project:
            raise ValueError("No current project set")
        
        return self.projects[self.current_project].log_model(model_name, predictions, params)
    
    def summary(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set")
        
        return self.projects[self.current_project].summary()