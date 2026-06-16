# modules/stats.py
import numpy as np
import pandas as pd
from scipy import stats
from collections import namedtuple

CIResult = namedtuple("CIResult", ["mean", "ci_low", "ci_high"])


def mean_ci(values: pd.Series, confidence: float = 0.95) -> CIResult:
    """
    Compute the mean and confidence interval of a numeric series[cite: 48].

    Parameters
    ----------
    values : pd.Series
        A numeric series. Must contain at least two non-null values[cite: 48].
    confidence : float, optional
        Confidence level for the interval. Must be strictly between 0 and 1[cite: 49, 92].
        Default is 0.95[cite: 49].

    Returns
    -------
    CIResult
        A named tuple with fields[cite: 49]:
        - mean (float): sample mean [cite: 49]
        - ci_low (float): lower bound of the confidence interval [cite: 49]
        - ci_high (float): upper bound of the confidence interval [cite: 49]

    Raises
    ------
    ValueError
        If `values` contains fewer than two non-null elements, or if
        `confidence` is not strictly between 0 and 1[cite: 92].

    Examples
    --------
    >>> import pandas as pd [cite: 50]
    >>> values = pd.Series([10, 20, 30, 40, 50]) [cite: 50]
    >>> result = mean_ci(values) [cite: 50]
    >>> round(result.mean, 1) [cite: 50]
    30.0 [cite: 50]
    """
    # Validate confidence level range
    if not (0 < confidence < 1):
        raise ValueError("Confidence level must be strictly between 0 and 1.")

    # Drop null values to assess true sample size for variance
    clean_values = values.dropna()
    if len(clean_values) < 2:
        raise ValueError("The input series must contain at least two non-null values.")

    mean = clean_values.mean()
    n = len(clean_values)
    std_err = clean_values.std() / np.sqrt(n)
    z = stats.norm.ppf((1 + confidence) / 2)
    
    ci_low = mean - z * std_err
    ci_high = mean + z * std_err
    
    return CIResult(mean=mean, ci_low=ci_low, ci_high=ci_high)


def summary_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Compute mean and 95% confidence intervals for multiple columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Column names to summarize.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns: variable, mean, ci_low, ci_high.
    """
    rows = []
    for col in columns:
        result = mean_ci(df[col])
        rows.append({
            "variable": col,
            "mean": round(result.mean, 4),
            "ci_low": round(result.ci_low, 4),
            "ci_high": round(result.ci_high, 4),
        })
    return pd.DataFrame(rows)