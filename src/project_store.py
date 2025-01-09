from src.commands.project_store_protocol import Model
from src.shell_project import ShellProject, ProjectType
from src.cliresult import chain, add_warning, CLIResult

from dataclasses import dataclass, field
import os
import json

@dataclass
class ProjectStore(Model):
    projects: dict[str, ShellProject] = field(default_factory=dict)
    current_project: str | None  = None

    @chain
    def create(self, alias: str, type: ProjectType) -> CLIResult:
        """
        Create a new project with the given alias and type.
        
        Args:
            alias (str): The alias for the project.
            type (ProjectType): The type of the project.
            
        Returns:
            CLIResult: Result of the operation.
        """
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
        return CLIResult(f'Project created successfully. {alias} is now the current project.')
        
    def delete(self, alias: str, from_dir: bool = False) -> CLIResult:
        """
        Delete a project with the given alias.
        
        Args:
            alias (str): The alias of the project to delete.
            from_dir (bool): Whether to delete the project directory as well.
            
        Returns:
            CLIResult: Result of the operation.
        """
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
            return CLIResult(f"Project {alias} deleted successfully from projects directory.")
        
        if alias not in self.projects:
            raise ValueError(f"Project {alias} does not exist.")

        del self.projects[alias]
        if self.current_project == alias:
            self.current_project = None
        return CLIResult(f"Project {alias} deleted successfully. Current project is {self.current_project}.")

    def list_projects(self) -> CLIResult:
        in_use = str(list(self.projects.keys()))
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        projects_dir = paths['projects_dir']
        saved_projects = os.listdir(projects_dir)
        return CLIResult(f"Projects in use: {in_use}\nProjects saved in projects directory: {str(saved_projects)}")
    
    def set_current_project(self, alias: str) -> CLIResult:
        if alias not in self.projects:
            raise ValueError(f"Project {alias} does not exist.")
        
        self.current_project = alias
        return CLIResult(f"Current project set to {alias}.")

    def pcp(self) -> CLIResult:
        if not self.current_project:
            return CLIResult("No current project set.")
        
        return CLIResult(self.projects[self.current_project].__str__())
    
    def load_project_from_file(self, alias: str) -> CLIResult:
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        project_path = paths['projects_dir'] + alias + '/'
        if os.path.exists(project_path):
            with open(project_path + 'metadata.json', 'r') as f:
                metadata = json.load(f)
            type_ = ProjectType(metadata['type'])
            is_cleaned = metadata['cleaned']
            description = metadata['description']
            feature_names = metadata['feature_names']
            
            self.create(alias, type_)
            self.projects[alias].project_description = description
            self.projects[alias].is_cleaned = is_cleaned
            self.projects[alias].feature_names = feature_names
        else:
            raise ValueError(f"Project {alias} not found.")
        if not self.current_project:
            raise ValueError("No current project set.")
        return self.projects[self.current_project].load_project_from_file(alias = alias)
    
    def get_current_project(self) -> ShellProject:
        if not self.current_project:
            raise ValueError("No current project set.")
        return self.projects[self.current_project]