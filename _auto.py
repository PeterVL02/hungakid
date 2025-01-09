from src.project_store import ProjectStore
from src.shell import Shell

import sys
from io import StringIO

def simulate_cli(commands: list[str]) -> None:
    project_store = ProjectStore()
    shell = Shell(project_store)
    
    original_stdin = sys.stdin
    sys.stdin = StringIO("\n".join(commands) + "\n")
    
    try:
        shell.run()
    finally:
        sys.stdin = original_stdin

def main() -> None:
    reg_commands = [
        "create reg_project regression",
        "read Iris",
        "view",
        "makexy SepalLengthCm",
        "linearregression",
        "mlpregressor --max_iter 1000",
        "summary",
        "exit",
    ]
    clas_commands = [
        "create clas_project classification",
        "read Iris",
        "view",
        "makexy Species",
        "gaussiannb",
        "mlpclassifier max_iter=1000",
        "logisticregression",
        "summary",
        "exit",
    ]
    full_commands = reg_commands[:-2] + [
        "",
        "",
        "create clas_project classification",
        "read Iris",
        "clean",
        "makexy Species",
        "view",
        "gaussiannb",
        "mlpclassifier -max_iter 1000",
        "logisticregression",
        "summary",
        "chproj reg_project",
        "summary",
        "pcp",
        "listproj",
        "chproj clas_project",
        "runall -n_values 2",
        "summary",
        "save",
        "plot hist sepallengthcm",
        "plot scatter [sepallengthcm, sepalwidthcm]",
        "plot box sepallengthcm",
        "show",
        "stats",
        "plot box sepalwidthcm -show True",
        "delete clas_project -from_dir True",
        "exit",
    ]

    simulate_cli(full_commands)

if __name__ == "__main__":
    main()