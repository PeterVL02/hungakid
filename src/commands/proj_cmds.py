from src.commands.project_store_protocol import Model
from src.commands.command_utils import convert_to_type, convert_to_ml_type

from pandas import DataFrame
import numpy as np

def create(model: Model, alias: str, type: str, *args, **kwargs) -> str:
    """Creates a new project with the given alias and type.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to create.
        type (str): Type of project. Must be one of 'regression', 'classification', or 'clustering'.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    new_type = convert_to_type(type)
    return model.create(alias, new_type)

def delete(model: Model, alias: str, from_dir: bool = False, *args, **kwargs) -> str:
    """
    Deletes the project with the given alias.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to delete.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
    
    return model.delete(alias, from_dir = from_dir)

def list_projects(model: Model, *args, **kwargs) -> str:
    """
    Lists all projects in the model.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        list[str]: A list of project aliases.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.list_projects()

def set_current_project(model: Model, alias: str, *args, **kwargs) -> str:
    """
    Sets the current project to the one with the given alias.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to set as current.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.set_current_project(alias)

def pcp(model: Model) -> str:
    """
    Retrieves the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: The current project or an error message if no project is set.
    """
    return model.pcp()

def add_data(model: Model, df_name: str, *args, **kwargs) -> str:
    """
    Adds a DataFrame to the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        df_name (str): Name of the DataFrame to add.

    Returns:
        None
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.add_data(df_name)

def read_data(model: Model, head: int = 5, *args, **kwargs) -> str:
    """
    Reads the data from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        head (int): Number of rows to display.

    Returns:
        str: A string representation of the DataFrame.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.read_data(head)

def make_X_y(model: Model, target: str, *args, **kwargs) -> str:
    """
    Creates the X and y arrays from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        target (str): Name of the target column.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.make_X_y(target)

def clean_data(model: Model, *args, **kwargs) -> str:
    """
    Cleans the data in the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.clean_data()

def log_model(model: Model, model_name: str, predictions: np.ndarray, params: dict[str, float | int | str], *args, **kwargs) -> str:
    """
    Logs a model in the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        model_name (str): Name of the model.
        predictions (list[float]): Predictions from the model.
        params (dict[str, float]): Parameters of the model.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    model_name = convert_to_ml_type(model_name)
    return model.log_model(model_name, predictions, params)

def summary(model: Model, *args, **kwargs) -> str:
    """
    Summarizes the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: A summary of the current project.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.summary()

def save(model: Model, overwrite: bool = False, *args, **kwargs) -> str:
    """
    Saves the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        overwrite (bool): Whether to overwrite the existing file if it exists.

    Returns:
        str: Optional message to display to the user.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
    
    return model.save(overwrite=overwrite)

def load_project_from_file(model: Model, alias: str, *args, **kwargs) -> str:
    """
    Loads a project from a file.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to load.

    Returns:
        str: Optional message to display to the user.
    """        
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.load_project_from_file(alias)

def plot(model: Model, cmd: str, labels: str | list[str] | None = None, show: bool = False, *args, **kwargs) -> str:
    """
    Generates a plot based on the given command and labels.

    Args:
        model (Model): The model containing the project data. Parsed automatically.
        cmd (str): The command to generate the plot.
        labels (str | list[str]): The labels to be used in the plot.
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        str: The result of the plot command.
    """
    if labels is None:
        labels = []
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.plot(cmd, labels, show)

def show(model: Model, *args, **kwargs) -> str:
    """
    Displays the plot for the current project.

    Args:
        model (Model): The model containing the project data. Parsed automatically.

    Returns:
        str: The result of the show command.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.show()

def stats(model: Model, *args, **kwargs) -> str:
    """
    Displays the statistics for the current project.

    Args:
        model (Model): The model containing the project data. Parsed automatically.

    Returns:
        str: The result of the stats command.
    """
    if args:
        print(f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        print(f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.stats()