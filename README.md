# Project README

## Overview
This project provides a command-line interface (CLI) for managing machine learning projects. Users can create, manipulate, and evaluate datasets and models interactively through a shell interface. The system is designed to be modular, extensible, and user-friendly, catering to both novice and advanced users.

## Features
- **Project Management:**
   - Create and manage multiple machine learning projects.
   - Switch between projects seamlessly.

- **Data Handling:**
   - Load datasets from CSV files.
   - Clean and preprocess data.
   - Generate feature (`X`) and target (`y`) matrices.
   - Create basic plots.
   - View summary statistics of the data.

- **Machine Learning Operations:**
   - Train regression and classification models such as Linear Regression, MLPs, Random Forests, and Gradient Boosting Classifiers.
   - Perform k-fold cross-validation.
   - Log and summarize model performance.

- **Interactive Shell:**
   - Execute commands interactively.
   - Chain multiple commands for complex workflows.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries:
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `colorama`
   - `tqdm`

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/PeterVL02/hungakid.git
    cd hungakid
    ```
2. Run the CLI:
    ```bash
    python main.py
    ```

## Usage
Place your CSV files in the `data` directory by default or configure a custom data directory with the `config` command.


### Starting the Shell
Run the following command to start the interactive shell:
```bash
python main.py
```

### Example Commands
1. **Create a new project:**
    ```
    >> create my_project regression
    ```
2. **Load a dataset:**
    ```
    >> add_data my_dataset
    ```
    *(Ensure the file `data/my_dataset.csv` exists.)*

3. **Preview data:**
    ```
    >> read_data -head 10
    ```

4. **Preprocess data:**
    ```
    >> clean_data; make_x_y target_column
    ```
    *(Replace `target_column` with the name of the target column.)*

5. **Plot data:**
    ```
    >> plot hist data_column
    >> plot scatter data_column target_column
    >> show
    >> plot close
    ```
    *(Replace `data_column` and `target_column` with the names of the columns to plot.)*

6. **Train a linear regression model:**
    ```
    >> linreg
    ```

7. **View project summary:**
    ```
    >> summary
    ```
8. **Tune hyperparameters of multiple models and log their performance and parameters:**
    ```
    >> log_best -n_values 3; summary
    ```

9. **Save and exit the shell:**
    ```
    >> save
    >> exit
    ```

## Get a Quick Feeling for it
Run the following command to start an automated stream of commands that showcases the features:
```bash
python _auto.py
```

## Documentation
Below is a list of commands. The commands have obligatory and optional parameters. The obligatory parameters are required for the command to execute successfully. The optional parameters are not required, but they can be used to modify the behavior of the command. To provide an optional parameter, you can add it as follows:
```bash
>> command_name parameter -optional_parameter value
>> command_name parameter -optional_parameter (if boolean, this will set the value to True)
>> command_name parameter --optional_parameter value
>> command_name parameter --optional_parameter (if boolean, this will set the value to True)
```

### Command (Basic)
```bash
>> help
```

```javascript
/**
 * Displays help information.
 *
 * @description
 * Use this function to display help information about the available commands. If a command is specified, it will display detailed information about that command.
 */
```

### Command (Basic)
```bash
>> create
```

```javascript
/**
 * Creates a new project using an alias and project type.
 *
 * @param {string} alias - A short, descriptive label for the project.
 * @param {string} projectType - Defines whether the project is for classification or regression
 * (accepts "classification" or "regression", or shortened forms "c" or "r").
 *
 * @description
 * Use this function to set up a project quickly by providing an alias and specifying the project type.
 * The project type determines the kind of modeling or analysis that will be performed.
 * Alias is a required identifier for naming or referencing the newly created resource.
 */
```

### Command (Basic)
```bash
>> add_data
```

```javascript
/**
 * Loads a dataset from a CSV file.
 *
 * @param {string} datasetName - The name of the dataset to be loaded. At this time only accepts CSV files. Leave out ".csv".
 *
 * @description
 * Use this function to load a dataset from a CSV file into the current project.
 * The dataset will be stored as the only dataset in the project.
 * The dataset name is a required identifier for referencing the loaded dataset.
 */
```

### Command (Data)
```bash
>> read_data
```

```javascript
/**
 * Displays the head of the dataset.
 *
 * @param {int} [head=5] - The number of rows to display.
 *
 * @description
 * Use this function to read the dataset. The number of rows and columns to display can be specified.
 * The mode determines whether to display the first or last rows of the dataset.
 */
```

### Command (Data)
```bash
>> clean_data
```

```javascript
/**
 * Cleans the dataset by removing missing values.
 *
 * @description
 * Use this function to clean the dataset by removing rows with missing values. It is recommended that you do your own cleaning before using the CLI.
 * The cleaned dataset will replace the original dataset within the project. It does not overwrite the original CSV file.
 */
```

### Command (Data)
```bash
>> make_x_y
```

```javascript
/**
 * Generates feature and target matrices from the dataset.
 *
 * @param {string} targetColumn - The name of the target column in the dataset.
 *
 * @description
 * Use this function to generate feature (`X`) and target (`y`) matrices from the dataset.
 * The target column is a required identifier for specifying the column to be used as the target variable.
 */
```

### Command (Data)
```bash
>> stats
```

```javascript
/**
 * Displays summary statistics of the dataset.
 *
 * @description
 * Use this function to display summary statistics of the dataset.
 */
```

### Command (Plotting)
```bash
>> plot
```

```javascript
/**
 * Plots the data.
 *
 * @param {string} plotType - The type of plot to generate (accepts "hist", "scatter", "box", or "close").
 * @param {string} ColumnName(s) - The name of the column(s) to plot.
 *
 * @description
 * Use this function to generate plots of the data. The plot type determines the kind of plot to generate.
 * The column name is a required identifier for specifying the column to plot. When using scatterplots, the columns must be specified as [column1, column2].
 * Using "close" closes any active plots. You can also add "-show" or "-show True" to immediately display the plot.
 */
```

### Command (Plotting)
```bash
>> show
```

```javascript
/**
 * Displays the active plot.
 *
 * @description
 * Use this function to display the active plot. This function is useful when the plot is not displayed automatically.
 */
```

### Command (Basic)
```bash
>> save
```

```javascript
/**
 * Saves the project.
 *
 * @param {boolean} [overwrite=false] - If true, overwrites the existing saved project with the same name. This action is irreversible.
 * 
 * @description
 * Use this function to save the current project. The project will be saved as a "projects" directory that can be configured.
 */
```

### Command (Basic)
```bash
>> load
```

```javascript
/**
 * Loads a project.
 *
 * @param {string} alias - The name of the project to load.
 *
 * @description
 * Use this function to load a previously saved project. The project name is a required identifier for specifying the project to load. This project will be set as the current project.
 */
```

### Command (Basic)
```bash
>> chproj
```

```javascript
/**
 * Changes the current project.
 *
 * @param {string} alias - The name of the project to switch to.
 *
 * @description
 * Use this function to switch between projects. The project name is a required identifier for specifying the project to switch to. You can only switch to projects that have been created or loaded within the session.
 */
```

### Command (Basic)
```bash
>> pcp
```

```javascript
/**
 * Prints the current project.
 *
 * @description
 * Use this function to print the current project. The project details will be displayed in the console.
 */
```

### Command (Basic)
```bash
>> listproj
```

```javascript
/**
 * Lists all projects.
 *
 * @description
 * Use this function to list all projects. Shows both saved projects and projects created in current session.
 */
```

### Command (Basic)
```bash
>> delete
```

```javascript
/**
 * Deletes a project.
 *
 * @param {string} alias - The name of the project to delete.
 * @param {boolean} [from_dir=false] - If true, delete a previously saved project from the disk.
 *
 * @description
 * Use this function to delete a project. The project name is a required identifier for specifying the project to delete.
 * By default, the project is only deleted from the session, but if you provide `-from_dir` or `-from_dir True`, it will remove the saved project from the disk as well.
 * This action is irreversible.
 */
```

### Command (ML)
```bash
>> linreg
```

```javascript
/**
 * Trains a linear regression model.
 *
 * @param {int} [n_splits = 10] - The number of splits (and folds) for cross-validation.
 * @param {int} [random_state = 42] - The random state for reproducibility.
 * @param {Any} [kwargs = None] - Additional keyword arguments to pass to the model. Visit the scikit-learn documentation for more information: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html
 * 
 * @description
 * Use this function to train a linear regression model on the dataset. The model will be trained using the feature and target matrices generated from the dataset.
 */
```

### Command (ML)
```bash
>> mlpreg
```

```javascript
/**
 * Trains a multi-layer perceptron regression model.
 *
 * @param {int} [n_splits = 10] - The number of splits (and folds) for cross-validation.
 * @param {int} [random_state = 42] - The random state for reproducibility.
 * @param {Any} [kwargs = None] - Additional keyword arguments to pass to the model. Visit the scikit-learn documentation for more information: https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPRegressor.html
 * 
 * @description
 * Use this function to train a multi-layer perceptron regression model on the dataset. The model will be trained using the feature and target matrices generated from the dataset.
 */
```

### Continuation
Commands seen above are just a two of the available commands. The pattern repeats. To see the full list of commands, run the `help` command in the CLI. For additional information, please refer to the specific command documentation through the `help` command, and visit scikit-learn's documentation.

## Command (ML)
```bash
>> log_best
```

```javascript
/**
 * Logs the best hyperparameters for multiple models. The models are trained using GridSearchCV. You can specify the number of values to try for each hyperparameter. Be aware that this function can take a long time to run.
 *
 * @param {int} [n_values = 3] - The number of values to try for each hyperparameter.
 * @param {int} [n_splits = 10] - The number of splits (and folds) for cross-validation.
 * @param {int} [random_state = 42] - The random state for reproducibility.
 * @param {Any} [kwargs = None] - Additional keyword arguments to pass to the model. Visit the scikit-learn documentation for more information: https://scikit-learn.org/1.5/modules/generated/sklearn.model_name.html
 * 
 * @description
 * Use this function to tune hyperparameters of multiple models and log their performance and parameters.
 */
```

## Command (ML)
```bash
>> summary
```

```javascript
/**
 * Displays a summary of the project. This includes statistics about model performance, as well as the parameters for each model.
 *
 * @description
 * Use this function to display a summary of the project. The summary includes information about the project, the dataset, and the models trained.
 */
```

## Command (Basic)
```bash
>> exit
```

```javascript
/**
 * Exits the CLI.
 *
 * @description
 * Use this function to exit the CLI. The session will be terminated, and any unsaved progress will be lost.
 */
```

## Command (Configuration)
```bash
>> config
```

```javascript
/**
 * Configures the CLI path settings.
 *
 * @param {string} command - Can be either "show", "get", or "set".
 * @param {string} directory - The directory to configure. Should be either "data_dir" or "projects_dir".
 * @param {string} [value] - The new path to set for the directory.
 *
 * @description
 * Use this function to configure the CLI path settings. You can set the paths for the data and projects directories. When using "show", no additional parameters are required. When using "get", specify the directory to get the current path. When using "set", specify the directory and the new path to set.
 */
```