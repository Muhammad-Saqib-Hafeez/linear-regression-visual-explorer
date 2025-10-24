from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

class Visualizer:
    @staticmethod
    def plot_data_and_regression(df: pd.DataFrame, model: LinearRegression,
                                 user_x: Optional[float] = None, user_y: Optional[float] = None,
                                 figsize=(8, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(df["X"], df["Y"], label="Data", alpha=0.75)
        x_min, x_max = df["X"].min(), df["X"].max()
        x_line = np.linspace(x_min - 1, x_max + 1, 200)
        y_line = model.predict(x_line.reshape(-1, 1))
        ax.plot(x_line, y_line, label="Regression line", linewidth=2)
        if user_x is not None and user_y is not None:
            ax.scatter([user_x], [user_y], color="red", s=90, marker="X", label="Your prediction")
            ax.annotate(f"{user_y:.2f}", xy=(user_x, user_y), xytext=(5, 10), textcoords="offset points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Scatter plot with linear regression line")
        ax.legend()
        ax.grid(alpha=0.2)
        return fig
