from src.cliresult import CLIResult
from src.commands.proj_cmds import (create, set_current_project, 
                                    list_projects, delete, pcp, 
                                    add_data, read_data, make_X_y, 
                                    clean_data, summary,
                                    save, load_project_from_file,
                                    stats, list_cols
                                    )
from src.commands.ml_cmds import (linreg, mlpreg, naivebayes, mlpclas, 
                                  logisticreg, decisiontree, randomforest, 
                                  gradientboosting, log_from_best)
from src.commands.config_cmds import config
from src.commands.plot_cmds import plot, show, pca_

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
    "linearregression": linreg, 
    "mlpregressor": mlpreg, 
    "gaussiannb": naivebayes, 
    "mlpclassifier": mlpclas, 
    "logisticregression": logisticreg, 
    "decisiontreeclassifier": decisiontree, 
    "randomforestclassifier": randomforest, 
    "gradientboostingclassifier": gradientboosting, 
    "create": create,
    "chproj": set_current_project,
    "listproj": list_projects,
    "delete": delete,
    "pcp" : pcp,
    "help" : list_cmds,
    "read": add_data, 
    "listcols": list_cols, 
    "view": read_data, 
    "makexy": make_X_y, #TODO: update references + readme
    "clean": clean_data, 
    "summary": summary,
    "runall" : log_from_best, # TODO: update references + readme
    "save": save,
    "load": load_project_from_file,
    "plot" : plot,
    "show" : show,
    "stats" : stats,
    "config" : config,
    "pca" : pca_
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
    