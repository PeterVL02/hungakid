from src.commands.project_store_protocol import Model
from src.shell_project import ShellProject, ProjectType
from src.commands.command_utils import MlModel
from src.MLOps.utils.base import BaseEstimator
from src.cliresult import chain, add_warning, CLIResult

from dataclasses import dataclass, field
import numpy as np
import os
import json
from typing import Any, Callable

@dataclass
class ProjectStore(Model):
    projects: dict[str, ShellProject] = field(default_factory=dict)
    current_project: str | None  = None

    @chain
    def create(self, alias: str, type: ProjectType) -> str:
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
            
        projects_dir = paths['projects_dir']
            
        if alias in self.projects:
            raise ValueError(f"Project {alias} already exists.")
        
        if not os.path.exists(projects_dir):
            os.makedirs(projects_dir)
        
        elif alias in os.listdir(projects_dir):
            add_warning(self, f"Warning: Project {alias} already exists in projects directory.")
        
        self.projects[alias] = ShellProject(project_type=type, project_name=alias)
        self.set_current_project(alias)
        return f'Project created successfully. {alias} is now the current project.'
        
    def delete(self, alias: str, from_dir: bool = False) -> str:
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        project_dir = paths['projects_dir'] + alias + '/'
        original_dir = os.getcwd()
        if from_dir:
            if not os.path.exists(project_dir):
                raise ValueError(f"Project {alias} does not exist in projects directory.")
            os.chdir(project_dir)
            for file in os.listdir():
                assert file in ['metadata.json', 'df.csv', 'modeldata.json', 'X.npy', 'y.npy'], f"Unexpected file {file} in project directory."
                if file in ['metadata.json', 'df.csv', 'modeldata.json', 'X.npy', 'y.npy']:
                    os.remove(file)
                
            os.chdir('..')
            os.rmdir(alias)
            os.chdir(original_dir)
            return f"Project {alias} deleted successfully from projects directory."
        
        if alias not in self.projects:
            raise ValueError(f"Project {alias} does not exist.")

        del self.projects[alias]
        if self.current_project == alias:
            self.current_project = None
        return f"Project {alias} deleted successfully. Current project is {self.current_project}."

    def list_projects(self) -> str:
        in_use = str(list(self.projects.keys()))
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        projects_dir = paths['projects_dir']
        saved_projects = os.listdir(projects_dir)
        return f"Projects in use: {in_use}\nProjects saved in projects directory: {str(saved_projects)}"
    
    def set_current_project(self, alias: str) -> str:
        if alias not in self.projects:
            raise ValueError(f"Project {alias} does not exist.")
        
        self.current_project = alias
        return f"Current project set to {alias}."

    def pcp(self) -> str:
        if not self.current_project:
            return "No current project set."
        
        return self.projects[self.current_project].__str__()
    
    def add_data(self, df_name: str, delimiter: str = ',') -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].add_df(df_name, delimiter=delimiter)

    def read_data(self, head: int = 5) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].read_data(head)
    
    def list_cols(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].list_cols()
    
    def make_X_y(self, target: str) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].make_X_y(target)

    def clean_data(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].clean_data()

    @chain
    def log_model(self, model_name: MlModel, predictions: np.ndarray, params: dict[str, float | int | str], **kwargs) -> str | CLIResult:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].log_model(model_name, predictions, params)
    
    def summary(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].summary()
    
    def log_predictions_from_best(self, *models: BaseEstimator, **kwargs) -> str | CLIResult:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].log_predictions_from_best(*models, **kwargs)
    
    def save(self, overwrite: bool = False) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].save(overwrite=overwrite)
    
    def load_project_from_file(self, alias: str) -> str:
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        project_path = paths['projects_dir'] + alias + '/'
        if os.path.exists(project_path):
            with open(project_path + 'metadata.json', 'r') as f:
                metadata = json.load(f)
            type_ = ProjectType(metadata['type'])
            is_cleaned = metadata['cleaned']
            description = metadata['description']
            self.create(alias, type_)
            self.projects[alias].project_description = description
            self.projects[alias].is_cleaned = is_cleaned
        else:
            raise ValueError(f"Project {alias} not found.")
        if not self.current_project:
            raise ValueError("No current project set.")
        return self.projects[self.current_project].load_project_from_file(alias = alias)

    def plot(self, cmd: str, labels: str | list[str], show: bool = False) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        try:
            return self.projects[self.current_project].plot(cmd, labels, show)
        except KeyError as e:
            raise ValueError(f"KeyError: {e}")
    
    def show(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        return self.projects[self.current_project].show()
    
    def stats(self) -> str:
        if not self.current_project:
            raise ValueError("No current project set.")
        
        return self.projects[self.current_project].stats()