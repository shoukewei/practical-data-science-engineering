# tests/test_stats.py

import pytest
import pandas as pd
import numpy as np
from modules.stats import mean_ci, CIResult

def test_mean_ci_returns_named_tuple():
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = mean_ci(values)
    assert isinstance(result, CIResult)


def test_mean_ci_correct_mean():
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = mean_ci(values)
    assert abs(result.mean - 30.0) < 1e-6


def test_mean_ci_interval_contains_mean():
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result = mean_ci(values)
    assert result.ci_low < result.mean < result.ci_high


def test_mean_ci_wider_at_higher_confidence():
    values = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    result_95 = mean_ci(values, confidence=0.95)
    result_99 = mean_ci(values, confidence=0.99)
    width_95 = result_95.ci_high - result_95.ci_low
    width_99 = result_99.ci_high - result_99.ci_low
    assert width_99 > width_95

def test_mean_ci_single_value_raises():
    """A series with one value has no variance — std is undefined."""
    values = pd.Series([42.0])
    with pytest.raises(ValueError, match="at least two non-null values"):
        mean_ci(values)

def test_mean_ci_with_real_data():
    """Verify the function runs correctly on a real-world column."""
    url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
    df = pd.read_csv(url, index_col=0)
    result = mean_ci(df["TV"])
    assert result.ci_low < result.mean < result.ci_high
    assert result.ci_low > 0  # TV budgets are non-negative