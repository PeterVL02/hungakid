from src.commands.project_store_protocol import Model
 
import json
import os
import shutil

def config(model: Model, cmd: str, dir: str | None = None, newpath: str | None = None) -> str:
    """
    Executes a configuration command.
    Parameters:
    cmd (str): The command to execute. Must be one of 'show', 'set', or 'get'.
    dir (str | None): The directory to operate on. Must be one of 'projects_dir' or 'data_dir'.
    newpath (str | None): The new path to set for the directory. Required if cmd is 'set'.
    Returns:
    str: The result of the command execution.
    Raises:
    AssertionError: If 'newpath' is provided without 'dir', or if 'dir' is not one of the valid options.
    KeyError: If 'cmd' is not one of the valid commands.
    """
    if newpath:
        assert (dir and newpath), "Both directory and new path must be provided."
        if not newpath.endswith('/'):
            newpath += '/'
    if dir:
        assert dir in ['projects_dir',  'data_dir'], "Invalid directory"
        if cmd == 'set':
            assert newpath is not None, "New path must be provided."
    
    if cmd != 'show':
        assert dir is not None, "Directory must be provided."
        
    commands = {
        'show' : _show,
        'set' : _set,
        'get' : _get
    }
    return commands[cmd](dir, newpath)

def _show(dir: str | None, newpath: str | None) -> str:
    with open('config/paths.json', 'r') as f:
        paths = json.load(f)
    if dir:
        return paths[dir]
    return str(paths)

def _move_files(dir: str, newpath: str) -> None:
    with open('config/paths.json', 'r') as f:
        paths = json.load(f)
    
    old_path = paths[dir]
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for filename in os.listdir(old_path):
        old_file = os.path.join(old_path, filename)
        new_file = os.path.join(newpath, filename)
        shutil.move(old_file, new_file)


def _set(dir: str, newpath: str) -> str:
    _move_files(dir, newpath)
    with open('config/paths.json', 'r') as f:
        paths = json.load(f)
    old_path = paths[dir]
    paths[dir] = newpath
    with open('config/paths.json', 'w') as f:
        json.dump(paths, f, indent=4)
    os.rmdir(old_path)
        
    
    return f"Path {dir} set to {newpath}. Files moved accordingly."

def _get(dir: str | None, newpath: str | None) -> str:
    with open('config/paths.json', 'r') as f:
        paths = json.load(f)
    return paths[dir] if dir else str(paths)

