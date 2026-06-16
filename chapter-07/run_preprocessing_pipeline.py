# run_preprocessing_pipeline (continued from missing data step)
from modules.data_io import load_dataset
from modules.preprocessing import (
    compute_fill_values,
    fill_missing,
    outlier_summary,
    compute_iqr_bounds,
    cap_with_bounds,
    log_transform,
)

import pandas as pd

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

# Load
df = load_dataset(url, index_col=0)

# Work on a copy so the original is preserved
X = df.copy()

# --- (Optional) introduce synthetic outliers for demonstration ---
if "TV" in X.columns and "newspaper" in X.columns and "sales" in X.columns:
    X.loc[5,  "TV"]        = 980.0
    X.loc[12, "newspaper"] = 350.0
    X.loc[27, "sales"]     = 0.1

# --- Missing-value handling (derive fill values from this dataset) ---
numeric_cols = X.select_dtypes(include="number").columns.tolist()
cols_with_na = [c for c in numeric_cols if X[c].isna().any()]
if cols_with_na:
    strategies = {c: "median" for c in cols_with_na}
    fill_vals = compute_fill_values(X, strategies)
    X_filled = fill_missing(X, fill_vals)
else:
    X_filled = X

print("\nOutlier summary before treatment:")
print(outlier_summary(X_filled))

# Outlier treatment: compute bounds on training data, cap, then transform
columns_to_treat = [c for c in ["TV", "newspaper"] if c in X_filled.columns]
if columns_to_treat:
    bounds = compute_iqr_bounds(X_filled, columns=columns_to_treat)
    X_capped = cap_with_bounds(X_filled, bounds)
else:
    X_capped = X_filled

if "newspaper" in X_capped.columns:
    X_final = log_transform(X_capped, "newspaper")
else:
    X_final = X_capped

print(f"\nOutlier summary after treatment:\n{outlier_summary(X_final)}")
if "newspaper" in X_filled.columns and "newspaper" in X_final.columns:
    print(
        f"newspaper skewness reduced from {X_filled['newspaper'].skew():.3f} "
        f"to {X_final['newspaper'].skew():.3f}"
    )