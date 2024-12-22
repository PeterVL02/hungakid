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

- **Machine Learning Operations:**
  - Train models like linear regression and multi-layer perceptron regression.
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
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Run the CLI:
   ```bash
   python main.py
   ```

## Usage

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
   >> read_data
   ```

4. **Preprocess data:**
   ```
   >> clean_data; make_X_y target_column
   ```

5. **Train a linear regression model:**
   ```
   >> linreg
   ```

6. **View project summary:**
   ```
   >> summary
   ```

7. **Exit the shell:**
   ```
   >> exit
   ```

## Directory Structure
```
project-folder/
├── src/
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── ml_utils.py
│   │   ├── ml_cmds.py
│   │   ├── reg_utils.py
│   │   ├── proj_cmds.py
│   │   ├── command_utils.py
│   │   ├── command.py
│   │   ├── command_factory.py
│   │   ├── project_store_protocol.py
│   │   ├── regression.py
│   │   └── stat_utils.py
│   │
│   ├── __init__.py
│   ├── project_store.py
│   ├── shell_project.py
│   └── shell.py
│   
├── data/
│   └── [your-datasets].csv
│
├── __init__.py
├── main.py
└── README.md
```

## Key Modules

### `main.py`
The entry point of the application. Initializes the project store and starts the shell interface.

### `project_store.py`
Manages multiple projects, including creation, deletion, and switching between projects.

### `shell.py`
Handles user interaction through the CLI. Parses and executes commands.

### `ml_utils.py`
Provides utility functions for data preprocessing and MLOps.

### `reg_utils.py`
Implements regression models with k-fold cross-validation.

### `command_factory.py`
Defines and maps available commands to their corresponding functions.

## Future Enhancements
- Expand model support (e.g., classification, clustering).
- Add a logging framework for better debugging.
- Integrate visualization tools for data and model performance.
- Implement a test suite for unit and integration testing.

## Acknowledgments
- `scikit-learn` for machine learning utilities.
- `pandas` and `numpy` for data manipulation.
- `colorama` for enhancing CLI appearance.

