import unittest

from tests.helpers import simulate_cli, convert_expected

expected = {
    'lowercasewarning' : 'Note: Command will be converted to lowercase.',
    'load bad' : 'Error: Project nonexistingproject not found.',
    'create' : 'Project created successfully. temporaryproj is now the current project.',
    'add_data' : 'Dataframe added successfully.',
    'bad_data' : 'Error: Dataframe nonexistingdata not found.',
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
    
    