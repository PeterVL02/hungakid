from pandas import DataFrame, read_csv
from dataclasses import dataclass, field
import numpy as np
import os

from src.MLOps.utils.stat_utils import accuracy_confidence_interval, mse_confidence_interval
from src.commands.command_utils import MlModel, ProjectType
from src.MLOps.utils.ml_utils import onehot_encode_string_columns


@dataclass
class ShellProject:
    project_type: ProjectType
    project_name: str
    project_description: str = ''
    is_cleaned: bool = False
    df: DataFrame | None = None
    X: np.ndarray | None = None
    y: np.ndarray | None = None
    
    modeldata: dict[str, dict[str, float]] = field(default_factory=dict)
    
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
        self.df.dropna(inplace=True)
        self.is_cleaned = True
        return "Data cleaned successfully."

    def log_model(self, model_name: MlModel, predictions: np.ndarray, params: dict[str, float]) -> str:
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
        for model_name, data in self.modeldata.items():
            summary_str += f"Model: {model_name}\n"
            for key, value in data.items():
                summary_str += f"  {key}: {value}\n"
            summary_str += "\n"
        
        return summary_str[:-2]

    def __str__(self) -> str:
        return f"Project: {self.project_name}, Type: {self.project_type}"