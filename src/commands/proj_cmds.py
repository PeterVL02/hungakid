from src.commands.project_store_protocol import Model
from src.commands.command_utils import convert_to_type, convert_to_ml_type
from src.cliresult import chain, add_warning
from src.cliresult import CLIResult

import numpy as np

@chain
def create(model: Model, alias: str, type: str, *args, **kwargs) -> CLIResult:
    """Creates a new project with the given alias and type.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to create.
        type (str): Type of project. Must be one of 'regression', 'classification', or 'clustering'.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    new_type = convert_to_type(type)
    return model.create(alias, new_type)

@chain
def delete(model: Model, alias: str, from_dir: bool = False, *args, **kwargs) -> CLIResult:
    """
    Deletes the project with the given alias.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to delete.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
    
    return model.delete(alias, from_dir = from_dir)

@chain
def list_projects(model: Model, *args, **kwargs) -> CLIResult:
    """
    Lists all projects in the model.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: A list of project aliases.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.list_projects()

@chain
def set_current_project(model: Model, alias: str, *args, **kwargs) -> CLIResult:
    """
    Sets the current project to the one with the given alias.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to set as current.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.set_current_project(alias)

def pcp(model: Model) -> CLIResult:
    """
    Retrieves the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: The current project or an error message if no project is set.
    """
    return model.pcp()

@chain
def add_data(model: Model, df_name: str, delimiter: str = ',', *args, **kwargs) -> CLIResult:
    """
    Adds a DataFrame to the current project.
    
    Supported file extensions:
    - .csv
    - .txt
    - .xls, .xlsx
    - .json
    - .xml
    - .html        

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Name of the DataFrame to add.
        delimiter (str): Delimiter used in the DataFrame (ignored if format is not '.txt' or '.csv').
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.add_df(df_name, delimiter = delimiter)

@chain
def read_data(model: Model, head: int = 5, *args, **kwargs) -> CLIResult:
    """
    Reads the data from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        head (int): Number of rows to display.

    Returns:
        CLIResult: A string representation of the DataFrame.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.read_data(head)

@chain
def list_cols(model: Model, *args, **kwargs) -> CLIResult:
    """
    Lists the columns of dataframe in the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: A list of columns in the current project.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.list_cols()

@chain
def make_X_y(model: Model, target: str, *args, **kwargs) -> CLIResult:
    """
    Creates the X and y arrays from the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        target (str): Name of the target column.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.make_X_y(target)

@chain
def clean_data(model: Model, *args, **kwargs) -> CLIResult:
    """
    Cleans the data in the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
    
    return project.clean_data()

@chain
def log_model(model: Model, model_name: str, predictions: np.ndarray, params: dict[str, float | int | str], *args, **kwargs) -> CLIResult:
    """
    Logs a model in the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        model_name (str): Name of the model.
        predictions (list[float]): Predictions from the model.
        params (dict[str, float]): Parameters of the model.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    model_name = convert_to_ml_type(model_name)
    
    project = model.get_current_project()
    
    return project.log_model(model_name, predictions, params)

@chain
def summary(model: Model, *args, **kwargs) -> CLIResult:
    """
    Summarizes the current project.

    Args:
        model (Model): Parsed automatically by the command parser.

    Returns:
        CLIResult: A summary of the current project.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.summary()

@chain
def save(model: Model, overwrite: bool = False, *args, **kwargs) -> CLIResult:
    """
    Saves the current project.

    Args:
        model (Model): Parsed automatically by the command parser.
        overwrite (bool): Whether to overwrite the existing file if it exists.

    Returns:
        CLIResult: Optional message to display to the user.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
    
    return project.save(overwrite=overwrite)

@chain
def load_project_from_file(model: Model, alias: str, *args, **kwargs) -> CLIResult:
    """
    Loads a project from a file.

    Args:
        model (Model): Parsed automatically by the command parser.
        alias (str): Alias of the project to load.

    Returns:
        CLIResult: Optional message to display to the user.
    """        
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    return model.load_project_from_file(alias)

@chain
def plot(model: Model, cmd: str, labels: str | list[str] | None = None, show: bool = False, *args, **kwargs) -> CLIResult:
    """
    Generates a plot based on the given command and labels.

    Args:
        model (Model): The model containing the project data. Parsed automatically.
        cmd (str): The command to generate the plot.
        labels (str | list[str]): The labels to be used in the plot.
        show (bool, optional): Whether to display the plot. Defaults to False.

    Returns:
        CLIResult: The result of the plot command.
    """
    if labels is None:
        labels = []
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.plot(cmd, labels, show)

@chain
def show(model: Model, *args, **kwargs) -> CLIResult:
    """
    Displays the plot for the current project.

    Args:
        model (Model): The model containing the project data. Parsed automatically.

    Returns:
        CLIResult: The result of the show command.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.show()

@chain
def stats(model: Model, *args, **kwargs) -> CLIResult:
    """
    Displays the statistics for the current project.

    Args:
        model (Model): The model containing the project data. Parsed automatically.

    Returns:
        CLIResult: The result of the stats command.
    """
    if args:
        add_warning(model, f"Warning: extra arguments {args} will be ignored.")
    elif kwargs:
        add_warning(model, f"Warning: extra arguments {kwargs} will be ignored.")
        
    project = model.get_current_project()
        
    return project.stats()