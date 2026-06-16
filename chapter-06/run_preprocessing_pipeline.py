# pipeline.py
from modules.data_io import load_dataset
from modules.eda import describe_df, missing_summary
from modules.preprocessing import (
    add_missing_indicator, 
    compute_fill_values, 
    fill_missing
)
import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

# Load
df = load_dataset(url, index_col=0)

# EDA
report = describe_df(df)
print(f"Dataset shape: {report['shape']}")
print(f"\nMissing summary:\n{missing_summary(df) if not missing_summary(df).empty else 'No missing values'}")

# Missing data handling (demonstrated on full data; in production split first)
X = df[["TV", "radio", "newspaper"]].copy()
y = df["sales"].copy()

# For demonstration: add indicators and fill
X = add_missing_indicator(X, ["radio", "newspaper"])

strategies = {
    "TV":        "median",
    "radio":     "mean",
    "newspaper": 0.0,
}

fill_values = compute_fill_values(X, strategies)   # compute on current data
X_filled = fill_missing(X, fill_values)

print(f"\nMissing values after handling: {X_filled.isnull().sum().sum()}")
print("Sample of filled data (with indicators):")
print(X_filled[["radio", "radio_missing", "newspaper", "newspaper_missing"]].head())
