from tests.helpers import simulate_cli, extract_ci_bounds

import unittest

expected = {
    'lowercasewarning' : 'Note: Command will be converted to lowercase.',
    'load bad' : 'Error: Project nonexistingproject not found.',
    'create' : 'Project created successfully. temporaryproj is now the current project.',
    'add_data' : 'Dataframe iris.csv added successfully.',
    'bad_data' : 'Error: Dataframe nonexistingdata not found.',
    'x_y' : 'X and y created successfully.',
    'not cleaned' : 'Warning: Data not cleaned. Run clean to clean data and rerun makexy to be safe...',
}

results = {
    'linreg' : 'CI: [0.080, 0.125] <==> 0.099 +- 0.027',
    'linreg_end' : 'Model linear_regression logged successfully.',
    'logreg' : 'CI: [0.9107, 0.9826] <==> 0.9467 +- 0.0360',
    'logreg_end' : 'Model logistic_regression logged successfully.',
}

class TestML(unittest.TestCase):        
    def test_linear(self):
        commands = [
            "create temporaryproj r",
            "read iris",
            "makexy sepallengthcm",
            "linearregression",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = results['linreg']

        # Extract CI bounds from both strings
        result_ci_bounds = extract_ci_bounds(result)
        converted_ci_bounds = extract_ci_bounds(converted)

        if None in result_ci_bounds or None in converted_ci_bounds:
            self.fail("CI bounds extraction returned None")

        result_ci_low, result_ci_high = result_ci_bounds
        converted_ci_low, converted_ci_high = converted_ci_bounds

        assert result_ci_low is not None and converted_ci_low is not None
        assert result_ci_high is not None and converted_ci_high is not None

        # Compare with tolerance
        self.assertLess(abs(result_ci_low - converted_ci_low), 0.001)
        self.assertLess(abs(result_ci_high - converted_ci_high), 0.001)
        self.assert_(not 'Error' in result)
        
    def test_logistic(self):
        commands = [
            "create temporaryproj c",
            "read iris",
            "makexy species",
            "logisticregression",
            "exit",
        ]
        result = simulate_cli(commands)
        converted = results['logreg']
        
        # Extract CI bounds from both strings
        result_ci_bounds = extract_ci_bounds(result)
        converted_ci_bounds = extract_ci_bounds(converted)
        
        if None in result_ci_bounds or None in converted_ci_bounds:
            self.fail("CI bounds extraction returned None")
            
            
        result_ci_low, result_ci_high = result_ci_bounds
        converted_ci_low, converted_ci_high = converted_ci_bounds
        
        assert result_ci_low is not None and converted_ci_low is not None
        assert result_ci_high is not None and converted_ci_high is not None
        
        # Compare with tolerance
        self.assertLess(abs(result_ci_low - converted_ci_low), 0.001)
        self.assertLess(abs(result_ci_high - converted_ci_high), 0.001)
        self.assert_(not 'Error' in result)

    def test_full_run(self):
        commands = [
            "create reg_project regression",
            "read Iris",
            "view",
            "makexy SepalLengthCm",
            "linearregression",
            "mlpregressor --max_iter 1000",
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
            "runall -n_values 1",
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
        self.assert_(not 'Error' in result)

    def test_full_run_2(self):
        commands = ["create test c; read iris; makexy species; runall -n_values 1; summary; exit"]
        result = simulate_cli(commands)
        self.assert_(not 'Error' in result)