from src.commands.regression.regression import linreg
from src.commands.proj_cmds import (create, set_current_project, 
                                    list_projects, delete, pcp, 
                                    add_data, read_data, make_X_y, 
                                    clean_data, log_model, summary
                                    )
from src.commands.ml_cmds import linreg, mlpreg, naivebayes, mlpclas, logisticreg

from typing import Any, Callable
from pandas import DataFrame

CommandFn = Callable[..., Any]

def list_cmds(*args, **kwargs) -> str:
    """
    Lists all available commands with their descriptions. You just used me.

    Returns:
        str: A formatted string listing all commands and their descriptions.
    """
    cmds = {name: cmd.__doc__ for name, cmd in COMMANDS.items()}
    return "\n".join(f"{name}: {desc}" for name, desc in cmds.items())

COMMANDS: dict[str, CommandFn] = {
    "linreg": linreg,
    "mlpreg": mlpreg,
    "naivebayes": naivebayes,
    "mlpclas": mlpclas,
    "logisticreg": logisticreg,
    "create": create,
    "set_current_project": set_current_project,
    "list_projects": list_projects,
    "delete": delete,
    "pcp" : pcp,
    "help" : list_cmds,
    "add_data": add_data,
    "read_data": read_data,
    "make_X_y": make_X_y,
    "clean_data": clean_data,
    "log_model": log_model,
    "summary": summary
}


def cmd_exists(cmd: str) -> bool:
    return cmd in COMMANDS


def execute_cmd(cmd: str, *args, **kwargs: Any) -> None:
    result = COMMANDS[cmd](*args, **kwargs)
    if isinstance(result, str): print(result)
    elif isinstance(result, DataFrame): print(result)