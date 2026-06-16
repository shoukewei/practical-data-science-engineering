# pipeline.py

from modules.data_io import load_dataset
from modules.eda import describe_df, missing_summary, skewness_summary, correlation_summary

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

# Load
df = load_dataset(url, index_col=0)

# EDA
report      = describe_df(df)
missing     = missing_summary(df)
skewness    = skewness_summary(df, threshold=1.0)
correlations = correlation_summary(df, target="sales")

# Report
print(f"Dataset: {report['shape'][0]} rows × {report['shape'][1]} columns")
print(f"\nMissing values:\n{missing if not missing.empty else 'None'}")
print(f"\nHigh-skew columns:\n{skewness[skewness['high_skew']]}")
print(f"\nCorrelations with Sales:\n{correlations}")
