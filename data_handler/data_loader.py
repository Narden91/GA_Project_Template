import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import yaml


class DataLoader:
    def __init__(self, file_path: str, config_path: str = "config/parameters.yaml"):
        self.file_path = file_path
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_data(self):
        """Load CSV file into a DataFrame and perform train-test split."""
        df = pd.read_csv(self.file_path)
        
        # Assuming the last column is the target
        X = df.iloc[:, :-1]  
        y = df.iloc[:, -1]   

        # Perform train-test split
        test_size = self.config.get("data", {}).get("test_size", 0.2)
        random_state = self.config.get("data", {}).get("random_state", 42)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Perform feature selection if enabled
        if self.config.get("data", {}).get("use_feature_selection", False):
            num_features = self.config["data"].get("num_selected_features", 10)
            X_train, X_test = self.feature_selection(X_train, y_train, X_test, num_features)

        return X_train, X_test, y_train, y_test

    def feature_selection(self, X_train, y_train, X_test, num_features):
        """Perform feature selection using ANOVA F-test."""
        selector = SelectKBest(score_func=f_classif, k=num_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        return X_train_selected, X_test_selected
