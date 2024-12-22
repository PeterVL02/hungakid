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
        "create my_project regression",
        "add_data Iris",
        "make_X_y SepalLengthCm",
        "linreg",
        "mlpreg max_iter=1000",
        "summary",
        "exit"
    ]
    clas_commands = [
        "create my_project classification",
        "add_data Iris",
        "read_data",
        "make_X_y Species",
        "naivebayes",
        "mlpclas max_iter=1000",
        "logisticreg",
        "summary",
        "exit"
    ]
    
    simulate_cli(reg_commands)
    simulate_cli(clas_commands)

if __name__ == "__main__":
    main()
