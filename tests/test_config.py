import unittest
import json

from tests.helpers import simulate_cli, convert_expected

class TestConfig(unittest.TestCase):
    def test_show(self):
        commands = [
            "config show",
            "exit",
        ]
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        result = simulate_cli(commands)
        self.assertEqual(result, str(paths))
        
    def test_set(self):
        with open('config/paths.json', 'r') as f:
            old_paths = json.load(f)
        commands = [
            "config set projects_dir test_dir",
            "config show",
            "config set projects_dir " + old_paths['projects_dir'],
            "exit",
        ]
        result = simulate_cli(commands)
        with open('config/paths.json', 'r') as f:
            new_paths = json.load(f)
        
        simulated_paths = old_paths.copy()
        simulated_paths['projects_dir'] = 'test_dir/'
        
        converted = convert_expected(f"Path projects_dir set to test_dir/. Files moved accordingly.", simulated_paths, "Path projects_dir set to " + old_paths['projects_dir'] + ". Files moved accordingly.")
        self.assertEqual(result, converted)
        self.assertEqual(new_paths, old_paths)
        
    def test_set_no_path(self):
        commands = [
            "config set projects_dir",
            "exit",
        ]
        result = simulate_cli(commands)
        self.assertEqual(result, "Error: New path must be provided.")
        
    def test_set_no_dir(self):
        commands = [
            "config set",
            "exit",
        ]
        result = simulate_cli(commands)
        self.assertEqual(result, "Error: Directory must be provided.")
        
    def test_set_invalid_dir(self):
        commands = [
            "config set invalid_dir new_path",
            "exit",
        ]
        result = simulate_cli(commands)
        self.assertEqual(result, "Error: Invalid directory")
        
    def test_get(self):
        commands = [
            "config get projects_dir",
            "exit",
        ]
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        result = simulate_cli(commands)
        self.assertEqual(result, paths['projects_dir'])