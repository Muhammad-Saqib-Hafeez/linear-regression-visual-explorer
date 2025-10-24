from dataclasses import dataclass, field
from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

@dataclass
class RegressionModel:
    model: LinearRegression = field(default_factory=LinearRegression)
    trained: bool = False
    r2_score_value: float = 0.0

    def train(self, X: np.ndarray, y: np.ndarray):
        X = X.reshape(-1, 1)
        self.model.fit(X, y)
        preds = self.model.predict(X)
        self.r2_score_value = float(r2_score(y, preds))
        self.trained = True

    def predict(self, X_new) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")
        Xn = np.array(X_new).reshape(-1, 1)
        return self.model.predict(Xn)

    @property
    def slope(self) -> float:
        return float(self.model.coef_[0]) if self.trained else 0.0

    @property
    def intercept(self) -> float:
        return float(self.model.intercept_) if self.trained else 0.0

    @property
    def r2(self) -> float:
        return float(self.r2_score_value)
