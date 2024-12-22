from src.shell import Shell
from src.project_store import ProjectStore

if __name__ == "__main__":
    project_store = ProjectStore()
    shell = Shell(project_store)
    shell.run()