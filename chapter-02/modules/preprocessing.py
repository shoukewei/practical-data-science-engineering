# preprocessing.py
import pandas as pd

def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna().copy()

def scale_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Standardize a numeric column to zero mean and unit variance.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the target column.
    column : str
        Name of the column to scale.

    Returns
    -------
    pd.Series
        A new Series with standardized values.
    """
    values = df[column]
    return (values - values.mean()) / values.std()