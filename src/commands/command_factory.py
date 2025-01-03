from src.cliresult import CLIResult
from src.commands.proj_cmds import (create, set_current_project, 
                                    list_projects, delete, pcp, 
                                    add_data, read_data, make_X_y, 
                                    clean_data, summary,
                                    save, load_project_from_file,
                                    plot, show, stats, list_cols
                                    )
from src.commands.ml_cmds import (linreg, mlpreg, naivebayes, mlpclas, 
                                  logisticreg, decisiontree, randomforest, 
                                  gradientboosting, log_from_best)
from src.commands.config_cmds import config

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
    "decisiontree": decisiontree,
    "randomforest": randomforest,
    "gradientboosting": gradientboosting,
    "create": create,
    "chproj": set_current_project,
    "listproj": list_projects,
    "delete": delete,
    "pcp" : pcp,
    "help" : list_cmds,
    "add_data": add_data,
    "list_cols": list_cols,
    "read_data": read_data,
    "make_x_y": make_X_y,
    "clean_data": clean_data,
    "summary": summary,
    "log_best" : log_from_best,
    "save": save,
    "load": load_project_from_file,
    "plot" : plot,
    "show" : show,
    "stats" : stats,
    "config" : config
}


def cmd_exists(cmd: str) -> bool:
    return cmd in COMMANDS


def execute_cmd(cmd: str, *args, **kwargs: Any) -> None | CLIResult:
    result = COMMANDS[cmd](*args, **kwargs)
    if isinstance(result, DataFrame): 
        result = result.to_string()
        result = CLIResult(result)
    elif isinstance(result, str):
        result = CLIResult(result)
    return result
    