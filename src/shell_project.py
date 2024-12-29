from pandas import DataFrame, read_csv
from dataclasses import dataclass, field
import numpy as np
import os
import pickle

from src.MLOps.utils.stat_utils import accuracy_confidence_interval, mse_confidence_interval
from src.commands.command_utils import MlModel, ProjectType
from src.MLOps.utils.ml_utils import onehot_encode_string_columns
from src.MLOps.utils.base import BaseEstimator
from src.MLOps.tuning import log_predictions_from_best


@dataclass
class ShellProject:
    project_type: ProjectType
    project_name: str
    project_description: str = ''
    is_cleaned: bool = False
    df: DataFrame | None = None
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    
    modeldata: dict[str, dict[str, float | int | str]] = field(default_factory=dict)
    
    def add_df(self, df_name: str) -> str:
        if not os.path.exists(f'data/{df_name}.csv'):
            raise ValueError(f"Dataframe {df_name} not found")
        self.df = read_csv(f'data/{df_name}.csv')
        return "Dataframe added successfully."

    def read_data(self, head: int = 5) -> DataFrame:
        if self.df is not None:
            return self.df.head(head)
        raise ValueError("Project has no dataframe")

    def make_X_y(self, target: str) -> str:
        if not self.is_cleaned:
            print("Warning: Data not cleaned. Run clean to clean data and rerun make_X_y to be safe...")
        if self.df is None:
            raise ValueError("Project has no dataframe. Use add_data to add a dataframe.")
        if target not in self.df.columns:
            raise ValueError(f"Target column {target} not in dataframe.")
        for col in self.df.columns:
            if col.lower() == 'id':
                self.df.drop(col, axis=1, inplace=True)

        self.df = onehot_encode_string_columns(self.df, ignore_columns=[target])
        self.y = np.array(self.df[target].values)

        self.X = self.df.drop(target, axis=1).values.astype(float)
        
        return "X and y created successfully."

    def clean_data(self) -> str:
        if self.df is None:
            raise ValueError("Project has no dataframe")
        obs_pre = len(self.df)
        self.df.dropna(inplace=True)
        self.is_cleaned = True
        obs_post = len(self.df)
        return f"Data cleaned successfully. Observations dropped: {obs_pre - obs_post}"

    def log_model(self, model_name: MlModel | str, predictions: np.ndarray, params: dict[str, float | int | str]) -> str:
        if self.X is None or self.y is None:
            raise ValueError("X and y not set. Run make_X_y first.")
        if self.project_type == ProjectType.CLASSIFICATION:
            score, CI_lower, CI_upper = accuracy_confidence_interval(self.y, predictions)

        elif self.project_type == ProjectType.REGRESSION:
            score, CI_lower, CI_upper = mse_confidence_interval(self.y, predictions, len(params))

        else:
            raise ValueError(f"Project type {self.project_type} not recognized")

        print(f'CI: [{CI_lower}, {CI_upper}] <==> {score} +- {CI_upper - score}' )
        self.modeldata[model_name] = {
            'score': score,
            'CI_lower': CI_lower,
            'CI_upper': CI_upper,
            **params
        }
        return f"Model {model_name} logged successfully."
    
    def summary(self) -> str:
        if not self.modeldata:
            return "No models logged yet."
        
        summary_str = "Model Summary:\n"
        sorted_models = sorted(self.modeldata.items(), key=lambda item: item[1]['score'], reverse=False)
        
        for model_name, data in sorted_models:
            summary_str += f"Model: {model_name}\n"
            for key, value in data.items():
                summary_str += f"  {key}: {value}\n"
            summary_str += "\n"
        
        return summary_str[:-2]
    
    def log_predictions_from_best(self, *models: BaseEstimator, cv: int = 10, n_values: int = 3) -> str:
        if self.X is None or self.y is None:
            raise ValueError("X and y not set. Run make_X_y first.")
        if not models:
            raise ValueError("No models provided.")
        log_predictions_from_best(*models, project=self, cv=cv, n_values=n_values)
        return "Predictions logged successfully."
    
    def save(self, overwrite: bool = False) -> str:
        project_path = f'projects/{self.project_name}'
        if not os.path.exists(project_path):
            os.makedirs(project_path)
        elif not overwrite:
            raise ValueError(f"Project {self.project_name} already exists. Use -overwrite=True to overwrite.")
        else:
            print(f"Warning: Overwriting project {self.project_name}.")
        
        if self.df is not None:
            self.df.to_csv(f'{project_path}/df.csv', index=False)
        if self.X is not None:
            np.save(f'{project_path}/X.npy', self.X)
        if self.y is not None:
            np.save(f'{project_path}/y.npy', self.y)
        if self.modeldata:
            modeldata_path = f'{project_path}/modeldata.pkl'
            with open(modeldata_path, 'wb') as f:
                pickle.dump(self.modeldata, f)
        with open(f'{project_path}/type.txt', 'w') as f:
            f.write(self.project_type)
        
        return f"Project {self.project_name} saved successfully."
    
    def load_project_from_file(self, alias: str) -> str:
        project_path = f'projects/{alias}'
        if not os.path.exists(project_path):
            raise ValueError(f"Project {alias} not found")
        try:
            self.df = read_csv(f'{project_path}/df.csv')
        except FileNotFoundError:
            print("Warning: Dataframe not found.")
        try:
            self.X = np.load(f'{project_path}/X.npy', allow_pickle=True)
            self.y = np.load(f'{project_path}/y.npy', allow_pickle=True)
        except FileNotFoundError:
            print("Warning: X and y not found.")
        try:
            with open(f'{project_path}/modeldata.pkl', 'rb') as f:
                self.modeldata = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Model data not found.")
        return f"Project {alias} loaded successfully."
            
        
    def __str__(self) -> str:
        return f"Project: {self.project_name}, Type: {self.project_type}"