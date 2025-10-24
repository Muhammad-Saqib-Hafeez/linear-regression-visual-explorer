from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

class DataHandler:

    @staticmethod
    def generate_synthetic(n_samples: int = 50, noise: float = 10.0, random_state: int = 42) -> pd.DataFrame:
        if n_samples <= 0:
            raise ValueError("n_samples must be > 0")
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=int(random_state))
        df = pd.DataFrame({"X": X.ravel(), "Y": y})
        return df.sort_values("X").reset_index(drop=True)

    @staticmethod
    def validate_user_x(x_value) -> float:
        """
        Validate and convert user input for prediction.
        Accepts numeric types and numeric strings.
        """
        if x_value is None:
            raise ValueError("Input X is required.")
        try:
            xv = float(x_value)
        except Exception as e:
            raise ValueError("Invalid input for X. Must be a number.") from e
        return xv

    @staticmethod
    def get_features_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if "X" not in df.columns or "Y" not in df.columns:
            raise ValueError("DataFrame must contain 'X' and 'Y' columns")
        X = df["X"].values
        y = df["Y"].values
        return X, y
