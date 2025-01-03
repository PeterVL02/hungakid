from typing import Callable, Any, Optional
from functools import wraps
from colorama import Fore
from dataclasses import dataclass
    
@dataclass    
class CLIResult:
    """
    Custom class to return a result and optional warning message or notes from a command.
    The result is either created by the command or instantiated by the "execute_cmd" function.
    """
    result: Any
    warning: str = ''
    note: str = ''
    c_message: str = Fore.WHITE
    c_warn: str = Fore.YELLOW
    c_note: str = Fore.CYAN
    
class InplaceModel:
    """
    Class to cheat the system.
    """
    def __init__(self):
        self.projects = {}
        self.current_project = None
        
    def _add_warning(self, message: str) -> None:
        """Bogus method that will never be called. Cheats the type checker."""
        self._warning = message
        self._note = message
        
def _parse_models(*args: Any) -> tuple[InplaceModel, InplaceModel]:
    """
    Recieves arguments parsed to the chain decorator. Will usually contain a model (ProjectStore instance).
    Attempt to parse the model and current project (ShellProject) from the arguments.
    If only one or neither are found, return an InplaceModel instance.

    Returns:
        tuple[InplaceModel, InplaceModel]: Tuple containing the model and project or InplaceModel instances.
    """
    try:
        model = args[0]
        try:
            project = model.projects[model.current_project]
        except (AttributeError, IndexError, KeyError):
            project = InplaceModel()
    except IndexError:
        model = InplaceModel()
        project = InplaceModel()
    return model, project

def _remove_duplicates(message: str) -> str:
    """
    Removes duplicate strings from a message.

    Args:
        message (str): The message to clean.

    Returns:
        str: The cleaned message.
    """
    return '\n'.join(list(set(message.split('\n'))))

def chain(func: Callable) -> Callable:
    """
    Decorator to chain without a model into the result chain.
    If a decorated function adds a warning or note, it will be picked up by the decorator, 
    deleted from the object, and added to the result.
    Ultimately, the result of the function will be returned with any warnings or notes as a CLIResult object.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> CLIResult:
        funcres = func(*args, **kwargs)
        model, project = _parse_models(*args)
        if isinstance(funcres, CLIResult):
            RESULT = funcres
        else:
            print('DEBUG NOT CLIRESULT')
            RESULT = CLIResult(result = funcres, 
                               warning = '', 
                               note = '', 
                               c_warn = Fore.RED, 
                               c_note = Fore.WHITE)
        if hasattr(project, "_warning"):
            RESULT.warning += '\n'.join(project._warning) + '\n'
            del project._warning
        if hasattr(model, "_warning"):
            RESULT.warning += '\n'.join(model._warning) + '\n'
            del model._warning
            
        if hasattr(project, "_note"):
            RESULT.note += '\n'.join(project._note) + '\n'
            del project._note
        if hasattr(model, "_note"):
            RESULT.note += '\n'.join(model._note) + '\n'
            del model._note
            
        if RESULT.warning:
            RESULT.warning = _remove_duplicates(RESULT.warning)
        if RESULT.note:
            RESULT.note = _remove_duplicates(RESULT.note)
            
        return RESULT
    return wrapper


def add_warning(cls: Any, message: str) -> None:
    """
    Adds a warning message to a class instance.
    Only works on functions decorated with the chain decorator.
    The function attaches the warning message to the class instance, which is then picked up by the chain decorator, 
    and eventually deleted from the class.

    Args:
        cls (Any): The class instance.
        message (str): The warning message.
    """
    if not hasattr(cls, "_warning"):
        cls._warning = []
    cls._warning.append(message)
    
def add_note(cls: Any, message: str) -> None:
    """
    Adds a message to a class instance.
    Only works on functions decorated with the chain decorator.
    The function attaches the message to the class instance, which is then picked up by the chain decorator,
    and eventually deleted from the class.

    Args:
        cls (Any): The class instance.
        message (str): The message.
    """
    if not hasattr(cls, "_note"):
        cls._note = []
    cls._note.append(message)