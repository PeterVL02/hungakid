from src.commands.model import Model
from src.main_project import ShellProject
from src.commands.command_utils import convert_to_type, convert_to_ml_type
from src.commands.command_utils import MlModel

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
    new_type = convert_to_type(type)
    return model.create(alias, new_type)

def delete(model: Model, alias: str, *args, **kwargs) -> str:
    """
    Deletes the project with the given alias.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to delete.

    Returns:
        str: Optional message to display to the user.
    """
    return model.delete(alias)

def list_projects(model: Model, *args, **kwargs) -> list[str]:
    """
    Lists all projects in the model.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        list[str]: A list of project aliases.
    """
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
    return model.set_current_project(alias)

def pcp(model: Model) -> ShellProject | str:
    """
    Retrieves the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        ShellProject | str: The current project or an error message if no project is set.
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
    return model.add_data(df_name)

def read_data(model: Model, head: int = 5, *args, **kwargs) -> DataFrame:
    """
    Reads the data from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        head (int): Number of rows to display.

    Returns:
        str: A string representation of the DataFrame.
    """
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
    return model.make_X_y(target)

def clean_data(model: Model, *args, **kwargs) -> str:
    """
    Cleans the data in the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        str: Optional message to display to the user.
    """
    return model.clean_data()

def log_model(model: Model, model_name: str, predictions: np.ndarray, params: dict[str, float], *args, **kwargs) -> str:
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
    return model.summary()