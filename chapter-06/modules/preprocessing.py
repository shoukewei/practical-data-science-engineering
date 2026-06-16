# modules/preprocessing.py (missing data section)

import numpy as np
import pandas as pd


def drop_missing(
    df: pd.DataFrame,
    subset: list = None,
    threshold: float = None
) -> pd.DataFrame:
    """
    Drop rows with missing values, with explicit control over scope.

    Parameters
    ----------
    df : pd.DataFrame
    subset : list of str, optional
        Columns to consider. If None, all columns are used.
    threshold : float, optional
        Drop rows where the proportion of missing values exceeds
        this threshold (0 to 1).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if threshold is not None:
        min_valid = int((1 - threshold) * df.shape[1])
        return df.dropna(thresh=min_valid, subset=subset)
    return df.dropna(subset=subset)


def add_missing_indicator(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:
    """
    Add binary indicator columns for missing values.

    Must be called before imputation.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Columns for which to add '{column}_missing' indicators.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def compute_fill_values(
    df: pd.DataFrame,
    strategies: dict
) -> dict:
    """
    Compute imputation fill values from a DataFrame.

    Always call this on training data only.

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    strategies : dict
        Mapping of column name to 'mean', 'median', 'mode',
        or a scalar constant.

    Returns
    -------
    dict
        Mapping of column name to computed fill value.
    """
    fill_values = {}
    for col, strategy in strategies.items():
        if strategy == "mean":
            fill_values[col] = df[col].mean()
        elif strategy == "median":
            fill_values[col] = df[col].median()
        elif strategy == "mode":
            fill_values[col] = df[col].mode()[0]
        else:
            fill_values[col] = strategy
    return fill_values


def fill_missing(
    df: pd.DataFrame,
    strategies: dict
) -> pd.DataFrame:
    """
    Apply column-specific imputation strategies to a DataFrame.

    When used in a train-test pipeline, pass the output of
    compute_fill_values() as the strategies argument to ensure
    fill values are derived from training data only.

    Parameters
    ----------
    df : pd.DataFrame
    strategies : dict
        Mapping of column name to strategy string ('mean',
        'median', 'mode') or scalar fill value.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col, strategy in strategies.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(strategy)
    return df