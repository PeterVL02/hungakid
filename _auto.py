import sys
from io import StringIO
from src.project_store import ProjectStore
from src.shell import Shell

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
        "add_data Iris",
        "read_data",
        "make_X_y SepalLengthCm",
        "linreg",
        "mlpreg --max_iter 1000",
        "summary",
        "exit",
    ]
    clas_commands = [
        "create clas_project classification",
        "add_data Iris",
        "read_data",
        "make_X_y Species",
        "naivebayes",
        "mlpclas max_iter=1000",
        "logisticreg",
        "summary",
        "exit",
    ]
    full_commands = reg_commands[:-2] + [
        "",
        "",
        "create clas_project classification",
        "add_data Iris",
        "clean_data",
        "make_X_y Species",
        "read_data",
        "naivebayes",
        "mlpclas -max_iter 1000",
        "logisticreg",
        "summary",
        "chproj reg_project",
        "summary",
        "pcp",
        "listproj",
        "chproj clas_project",
        "log_best -n_values 2",
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