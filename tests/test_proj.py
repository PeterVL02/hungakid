from tests.helpers import simulate_cli, convert_expected

import unittest
import os
import json

expected = {
    'lowercasewarning' : 'Note: Command will be converted to lowercase.',
    'load bad' : 'Error: Project nonexistingproject not found.',
    'create' : 'Project created successfully. temporaryproj is now the current project.',
    'add_data' : 'Dataframe iris.csv added successfully.',
    'bad_data' : 'Error: Dataframe nonexistingdata not found.',
    'save' : 'Project temporaryproj saved successfully.',
    'clean_data' : 'Data cleaned successfully. Observations dropped: 0',
    'make_x_y' : 'X and y created successfully.',
    'load' : 'Project temporaryproj loaded successfully.',
    'modeldata not found' : 'Warning: Model data not found.',
    'delete' : 'Project temporaryproj deleted successfully from projects directory.',
}

class TestCommands(unittest.TestCase):
    def test_load(self):
        commands = [
            "load NonExistingProject",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = convert_expected(expected['lowercasewarning'], expected["load bad"])
        self.assertEqual(result, converted)
        
    def test_create_reg(self):
        commands = [
            "create temporaryproj regression",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = convert_expected(expected["create"])
        self.assertEqual(result, converted)
    
    def test_create_clas(self):
        commands = [
            "create temporaryproj classification",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = convert_expected(expected["create"])
        self.assertEqual(result, converted)
        
    def test_add_data(self):
        commands = [
            "create temporaryproj regression",
            "add_data Iris",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = convert_expected(expected['create'], expected['lowercasewarning'], expected['add_data'])
        self.assertEqual(result, converted)
        
    def test_add_wrong_data(self):
        commands = [
            "create temporaryproj regression",
            "add_data nonexistingdata",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = convert_expected(expected['create'], expected['bad_data'])
        self.assertEqual(result, converted)
    
    def test_save_load_and_delete(self):
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        projects_dir = paths['projects_dir']
        original_saved_projects = os.listdir(projects_dir)
        commands = [
            "create temporaryproj r",
            "add_data iris",
            "clean_data",
            "make_x_y sepallengthcm",
            "save",
            "exit",
        ]
        result1 = simulate_cli(commands)
        converted1 = convert_expected(expected['create'], expected['add_data'], expected['clean_data'], expected['make_x_y'], expected['save'])
        
        self.assert_(os.path.exists(f"{projects_dir}/temporaryproj/"))
        self.assert_(len(os.listdir(projects_dir)) == len(original_saved_projects) + 1)
        
        commands = [
            "load temporaryproj",
            "exit",
        ]
        result2 = simulate_cli(commands)
        converted2 = convert_expected(expected['modeldata not found'], expected['load'])
        
        commands = [
            "delete temporaryproj -from_dir",
            "exit",
        ]
        result3 = simulate_cli(commands)
        converted3 = convert_expected(expected['delete'])
        
        self.assert_(not os.path.exists(f"{projects_dir}/temporaryproj/"))
        self.assert_(os.listdir(projects_dir) == original_saved_projects)
        self.assertEqual(result1, converted1)
        self.assertEqual(result2, converted2)
        self.assertEqual(result3, converted3)
        