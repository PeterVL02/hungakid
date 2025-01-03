from typing import Callable, Any, Optional
from functools import wraps
from colorama import Fore
from dataclasses import dataclass
    
@dataclass    
class CLIResult:
    """
    Custom class to return a result and a warning message from a command.
    """
    result: Any
    warning: Optional[str] = None
    note: Optional[str] = None
    c_message: str = Fore.WHITE
    c_warn: str = Fore.WHITE
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
    try:
        model = args[0]
        try:
            project = model.projects[model.current_project]
        except (AttributeError, IndexError):
            project = InplaceModel()
    except IndexError:
        model = InplaceModel()
        project = InplaceModel()
    return model, project

def _remove_duplicates(message: str) -> str:
    """
    Removes duplicate lines from a message.

    Args:
        message (str): The message to clean.

    Returns:
        str: The cleaned message.
    """
    return '\n'.join(list(set(message.split('\n'))))

def chain(func: Callable) -> Callable:
    """
    Decorator to chain without a model into the result chain.

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
            message = funcres.warning if funcres.warning else ''
            note = funcres.note if funcres.note else ''
            result = funcres.result
            c_warn = funcres.c_warn
            c_note = funcres.c_note
        else:
            result = funcres
            message: str = ''
            note: str = ''
            c_warn = Fore.WHITE
            c_note = Fore.WHITE
        if hasattr(project, "_warning"):
            message += '\n'.join(project._warning) + '\n'
            del project._warning
            c_warn = Fore.YELLOW
        if hasattr(model, "_warning"):
            message += '\n'.join(model._warning) + '\n'
            del model._warning
            c_warn = Fore.YELLOW
            
        if hasattr(project, "_note"):
            note += '\n'.join(project._note) + '\n'
            del project._note
            c_note = Fore.CYAN
        if hasattr(model, "_note"):
            note += '\n'.join(model._note) + '\n'
            del model._note
            c_note = Fore.CYAN
            
        if message:
            message = _remove_duplicates(message)
        if note:
            note = _remove_duplicates(note)
            
        return CLIResult(result = result, warning = message if message else None, note = note if note else None, c_warn = c_warn, c_note = c_note)
    return wrapper


def add_warning(cls: Any, message: str) -> None:
    """
    Adds a warning message to a class instance.

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

    Args:
        cls (Any): The class instance.
        message (str): The message.
    """
    if not hasattr(cls, "_note"):
        cls._note = []
    cls._note.append(message)