from src.MLOps.utils.stat_utils import accuracy_confidence_interval, mse_confidence_interval
from src.commands.command_utils import MlModel, ProjectType
from src.MLOps.utils.ml_utils import onehot_encode_string_columns, clean_dict
from src.MLOps.utils.base import BaseEstimator
from src.MLOps.tuning import log_predictions_from_best
from src.MLOps.visuals.crud.cruds import Plotter
from src.cliresult import chain, add_warning, add_note, CLIResult
from src.MLOps.visuals.pca.pca import pca_fit

from pandas import DataFrame, read_csv, read_json, read_excel, read_xml, read_html
from dataclasses import dataclass, field
import numpy as np
from sklearn.decomposition import PCA
import os
import json


@dataclass
class ShellProject:
    project_type: ProjectType
    project_name: str
    project_description: str = ''
    is_cleaned: bool = False
    df: DataFrame | None = None
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    feature_names: list[str] | None = None
    plotter: Plotter = Plotter()
    pca : PCA | None = None
    
    modeldata: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    
    def add_df(self, df_name: str, delimiter: str = ',') -> CLIResult:
        """
        Loads data from a file into a pandas DataFrame.
        
        Supported file extensions:
        - .csv
        - .txt
        - .xls, .xlsx
        - .json
        - .xml
        - .html
        
        :param df_name: The name of the file to load.
        :param delimiter: The delimiter to use for CSV and TXT files.
        """
        EXTENSIONS = {".csv" : read_csv, 
                      ".txt" : read_csv, 
                      ".xls" : read_excel, 
                      ".xlsx" : read_excel, 
                      ".json" : read_json, 
                      ".xml" : read_xml, 
                      ".html" : read_html
                      }
        
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        data_dir = paths['data_dir']
        for file in os.listdir(data_dir):
            os.rename(data_dir + file, data_dir + file.lower())
        
        file = None
        for ext, load_func in EXTENSIONS.items():
            if os.path.exists(data_dir + f'{df_name}{ext}'):
                file = data_dir + f'{df_name}{ext}'
                break
        else:
            raise ValueError(f"Dataframe {df_name} not found.")
        

        if ext in {".csv", ".txt"}:
            self.df = load_func(file, delimiter=delimiter)
        else:
            self.df = load_func(file)
        
        if self.df is None:
            raise ValueError("No dataframe could be loaded.")
        
        for col in self.df.columns:
            if col.lower() == 'id':
                self.df.drop(col, axis=1, inplace=True)
            else:
                self.df.rename(columns={col: col.lower().strip()}, inplace=True)
        self.is_cleaned = False
        self.plotter = Plotter()
        self.pca = None
        self.X, self.y = None, None
        return CLIResult(f"Dataframe {file.split('/')[-1]} added successfully.")

    @chain
    def read_data(self, head: int = 5) -> CLIResult:
        if not self.is_cleaned:
            add_warning(self, "Warning: Data not cleaned. Run clean to clean data and rerun view to be safe...")
        if self.df is not None:
            return CLIResult(self.df.head(head).to_string())
        raise ValueError("Project has no dataframe")
    
    @chain
    def list_cols(self) -> CLIResult:
        if not self.is_cleaned:
            add_warning(self, "Warning: Data not cleaned. Run clean to clean data and rerun listcols to be safe...")
        if self.df is None:
            raise ValueError("Project has no dataframe.")
        return CLIResult(str(self.df.columns.tolist()))

    @chain 
    def make_X_y(self, target: str) -> CLIResult:
        if not self.is_cleaned:
            add_warning(self, "Warning: Data not cleaned. Run clean to clean data and rerun makexy to be safe...")
        if self.df is None:
            raise ValueError("Project has no dataframe. Use read to add a dataframe.")
        if target not in self.df.columns:
            raise ValueError(f"Target column {target} not in dataframe.")
        for col in self.df.columns:
            if col.lower() == 'id':
                self.df.drop(col, axis=1, inplace=True)
        
        if self.project_type == ProjectType.CLASSIFICATION:
            if isinstance(self.df[target][0], float):
                raise ValueError("Target column is not categorical. Use regression project type. Otherwise, convert target to string.")
            num_unique = len(self.df[target].unique())
            if num_unique > 15:
                add_warning(self, f"Warning: Target column has {num_unique} unique values. Consider reducing unique values for better performance.")
            else:
                add_note(self, f"Note: Target column has {num_unique} unique values.")
        elif self.project_type == ProjectType.REGRESSION:
            if isinstance(self.df[target][0], str):
                raise ValueError("Target column is not numerical. Use classification project type. Otherwise, convert target to float.")
            if len(self.df[target].unique()) < 15:
                add_warning(self, "Warning: Target column has few unique values.")
            
        self.df = onehot_encode_string_columns(self.df, ignore_columns=[target])
        self.y = np.array(self.df[target].values)

        self.X = self.df.drop(target, axis=1).values.astype(float)
        self.feature_names = self.df.drop(target, axis=1).columns.tolist()
        
        return CLIResult("X and y created successfully.")

    def clean_data(self) -> CLIResult:
        if self.df is None:
            raise ValueError("Project has no dataframe.")
        obs_pre = len(self.df)
        self.df.dropna(inplace=True)
        self.is_cleaned = True
        obs_post = len(self.df)
        return CLIResult(f"Data cleaned successfully. Observations dropped: {obs_pre - obs_post}")

    @chain
    def log_model(self, model_name: MlModel | str, predictions: np.ndarray, params: dict[str, float | int | str], **kwargs: dict[str, float | int | str]) -> CLIResult:
        if self.X is None or self.y is None:
            raise ValueError("X and y not set. Run makexy first.")
        if self.project_type == ProjectType.CLASSIFICATION:
            score, CI_lower, CI_upper = accuracy_confidence_interval(self.y, predictions)

        elif self.project_type == ProjectType.REGRESSION:
            score, CI_lower, CI_upper = mse_confidence_interval(self.y, predictions, len(params))

        else:
            raise ValueError(f"Project type {self.project_type} not recognized.")


        add_note(self, f'CI: [{CI_lower:.4f}, {CI_upper:.4f}] <==> {score:.4f} +- {(CI_upper - score):.4f}' )
        
        previous_model = self.modeldata.get(model_name, None)
        if previous_model and previous_model['score'] >= score: # type: ignore
                return CLIResult(f"Model {model_name} not logged. Previous model has higher score.")
        
        self.modeldata[model_name] = { # type: ignore
            'score': score,
            'CI_lower': CI_lower,
            'CI_upper': CI_upper,
            'additionals' : clean_dict(kwargs),
            'parameters' : clean_dict(params),
        } 
        return CLIResult(f"Model {model_name} logged successfully.")
    
    def summary(self) -> CLIResult:
        if not self.modeldata:
            return CLIResult("No models logged yet.")
        
        summary_str = "Model Summary:\n"
        sorted_models = sorted(self.modeldata.items(), key=lambda item: item[1]['score'], reverse=False)
        
        for model_name, data in sorted_models:
            summary_str += f"Model: {model_name}\n"
            for key, value in data.items():
                summary_str += f"  {key}: {value}\n"
            summary_str += "\n"
        
        return CLIResult(summary_str[:-2])
    
    @chain
    def log_predictions_from_best(self, *models: BaseEstimator, cv: int = 10, n_values: int = 3) -> CLIResult:
        if self.X is None or self.y is None:
            raise ValueError("X and y not set. Run makexy first.")
        if not models:
            raise ValueError("No models provided.")
        return log_predictions_from_best(*models, project=self, cv=cv, n_values=n_values)
    
    @chain   
    def save(self, overwrite: bool = False) -> CLIResult:
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
        if not os.path.exists(paths['projects_dir']):
            os.makedirs(paths['projects_dir'])
            
        project_path = paths['projects_dir'] + self.project_name + '/'
        
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        elif not overwrite:
            raise ValueError(f"Project {self.project_name} already exists. Use -overwrite to overwrite.")
        else:
            add_warning(self, f"Warning: Overwriting project {self.project_name}.")
        
        if self.df is not None:
            self.df.to_csv(project_path + f'df.csv', index=False)
        if self.X is not None:
            np.save(project_path + 'X.npy', self.X)
        if self.y is not None:
            np.save(project_path + 'y.npy', self.y)
        if self.modeldata:
            modeldata_path = project_path + 'modeldata.json'
            with open(modeldata_path, 'w') as f:
                json.dump(self.modeldata, f, indent=4)
        type_path = project_path + 'metadata.json'
        metadata = {
            'description': self.project_description,
            'type': self.project_type,
            'cleaned': self.is_cleaned,
            'feature_names': self.feature_names
        }
        with open(type_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return CLIResult(f"Project {self.project_name} saved successfully.")
    
    @chain
    def load_project_from_file(self, alias: str) -> CLIResult:
        with open('config/paths.json', 'r') as f:
            paths = json.load(f)
            
        project_path = paths['projects_dir'] + alias + '/'

        if not os.path.exists(project_path):
            raise ValueError(f"Project {alias} not found.")
        try:
            self.df = read_csv(project_path + 'df.csv')
        except FileNotFoundError:
            add_warning(self, "Warning: Dataframe not found.")
        try:
            self.X = np.load(project_path + 'X.npy', allow_pickle=True)
            self.y = np.load(project_path + 'y.npy', allow_pickle=True)
        except FileNotFoundError:
            add_warning(self, "Warning: X and y not found.")
        try:
            with open(project_path + 'modeldata.json', 'r') as f:
                self.modeldata = json.load(f)
        except FileNotFoundError:
            add_warning(self, "Warning: Model data not found.")
        return CLIResult(f"Project {alias} loaded successfully.")
    
    def plot(self, cmd: str, labels: str | list[str], show: bool = False) -> CLIResult:
        if self.df is None:
            raise ValueError("Project has no dataframe.")
        if isinstance(labels, str):
            assert labels in self.df.columns, f"Column {labels} not in dataframe. Columns are {self.df.columns.tolist()}."
            self.plotter.plot_interact(cmd = cmd, series = np.array(self.df[labels].values), label = labels, show = show)
        elif isinstance(labels, list):
            assert all(isinstance(label, str) for label in labels), "Labels must be of type str."
            assert all(label in self.df.columns for label in labels), f"Columns {labels} not in dataframe. Columns are {self.df.columns.tolist()}."
            _series: list[np.ndarray] = []
            for label in labels:
                _series.append(np.array(self.df[label].values))
            self.plotter.plot_interact(cmd = cmd, series = _series, label = labels, show = show)
        else:
            raise ValueError('Labels must be of type str or list of strings.')
        return CLIResult('Success.')
    
    @chain
    def plot_pca(self, show: bool = False) -> CLIResult:
        if self.X is None or self.y is None or self.feature_names is None:
            if self.df is None:
                raise ValueError("Project has no dataframe. Use read to add a dataframe.")
            raise ValueError("X and y not set. Run makexy first.")
        if self.pca is None:
            self.run_pca()
        assert self.pca is not None, "PCA not run successfully."
        self.plotter.pca_plot(self.pca, self.X, self.y, task=self.project_type.value, cols = self.feature_names, show=show)
        return CLIResult('PCA plot created successfully.')
        
    
    def show(self) -> CLIResult:
        self.plotter.show()
        return CLIResult('Plots shown successfully.')
    
    def stats(self) -> CLIResult:
        if self.df is None:
            raise ValueError("Project has no dataframe.")
        return CLIResult(self.df.describe().to_string())
    
    def run_pca(self) -> CLIResult:
        if self.X is None or self.y is None:
            if self.df is None:
                raise ValueError("Project has no dataframe. Use read to add a dataframe.")
            raise ValueError("X and y not set. Run makexy first.")
        self.pca = pca_fit(self.X)
        add_note(self, f"Note: PCA explained variance: {self.pca.explained_variance_ratio_}")
        return CLIResult("Ran PCA successfully.")
        
        
    def __str__(self) -> str:
        return f"Project: {self.project_name}, Type: {self.project_type}"