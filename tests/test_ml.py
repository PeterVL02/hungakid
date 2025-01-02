import unittest

from tests.helpers import simulate_cli, convert_expected

expected = {
    'lowercasewarning' : 'Note: Command will be converted to lowercase.',
    'load bad' : 'Error: Project nonexistingproject not found.',
    'create' : 'Project created successfully. temporaryproj is now the current project.',
    'add_data' : 'Dataframe added successfully.',
    'bad_data' : 'Error: Dataframe nonexistingdata not found.',
    'x_y' : 'X and y created successfully.',
    'not cleaned' : 'Warning: Data not cleaned. Run clean_data to clean data and rerun make_X_y to be safe...',
}

results = {
    'linreg' : 'CI: [0.080, 0.125] <==> 0.099 +- 0.027',
    'linreg_end' : 'Model linear_regression logged successfully.',
}

class TestML(unittest.TestCase):
    def test_linear(self):
        commands = [
            "create temporaryproj r",
            "add_data iris",
            "make_x_y sepallengthcm",
            "linreg",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = convert_expected(expected['create'], expected['add_data'], expected['not cleaned'], expected['x_y'],  results['linreg_end'], results['linreg'])
        self.assertEqual(result, converted)
        
    def test_full_run(self):
        commands = [
            "create reg_project regression",
            "add_data Iris",
            "read_data",
            "make_X_y SepalLengthCm",
            "linreg",
            "mlpreg --max_iter 1000",
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
            "log_best -n_values 1",
            "summary",
            "save",
            "plot hist sepallengthcm",
            "plot scatter [sepallengthcm, sepalwidthcm]",
            "plot box sepallengthcm",
            "show",
            "plot close",
            "stats",
            "plot box sepalwidthcm -show",
            "plot close",
            "delete clas_project -from_dir",
            "exit",
         ]
        result = simulate_cli(commands)
