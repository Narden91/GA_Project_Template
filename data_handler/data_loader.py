import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path


class DataLoader:
    def __init__(self, file_path: Path, config_path: str = "config/parameters.yaml"):
        self.file_path = file_path
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        """Load CSV file and perform train-test split."""
        df = pd.read_csv(self.file_path)
        
        # Assuming the last column is the target variable
        X = df.iloc[:, :-1].values  
        y = df.iloc[:, -1].values  

        # Perform train-test split
        test_size = self.config.get("data", {}).get("test_size", 0.2)
        random_state = self.config.get("data", {}).get("random_state", 42)

        return train_test_split(X, y, test_size=test_size, random_state=random_state)