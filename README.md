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
