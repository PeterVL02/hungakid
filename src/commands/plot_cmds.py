from src.cliresult import chain, add_warning, CLIResult
from src.commands.project_store_protocol import Model

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
def pca_(model: Model, cmd: str, show: bool = False) -> CLIResult:
    """
    Performs PCA on the current project.
    
    Args:
        model (Model): The model containing the project data. Parsed automatically.
        cmd (str): The command to run. Must be one of 'run' or 'plot'.
        show (bool, optional): Whether to display the plot. Ignored for 'run'.  Defaults to False.
        
    Returns:
        CLIResult: The result of the PCA command.
    """
    if cmd is None:
        raise ValueError("PCA Command must be provided.")
    
    project = model.get_current_project()
    if cmd == 'run':
        if show:
            add_warning(model, 'show argument will be ignored for run command.')
        return project.run_pca()
    elif cmd == 'plot':
        return project.plot_pca(show = show)
    else:
        raise ValueError(f"Invalid PCA command {cmd}.")
    

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