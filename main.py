"""
This is the main entry point for the application.
It initializes the ProjectStore and Shell, then starts the shell. Example usage:
>>> python main.py
>>> >> create bonk regression; add_data Iris; read_data; make_X_y SepalLengthCm; linreg; mlpreg max_iter=1000; summary
"""

from src.shell import Shell
from src.project_store import ProjectStore

def main() -> None:
    project_store = ProjectStore()
    shell = Shell(project_store)
    shell.run()

if __name__ == "__main__":
    main()