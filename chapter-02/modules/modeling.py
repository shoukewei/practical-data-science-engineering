# modeling.py
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_linear_model(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model