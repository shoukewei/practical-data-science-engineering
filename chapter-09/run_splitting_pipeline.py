# run_splitting_pipeline.py
import pandas as pd
from modules.data_io import load_dataset
from modules.splitting import (
    create_split, create_three_way_split, create_stratified_split, create_time_split,
    save_split, load_split, cross_validate_model
)

# Load dataset and print summary
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

df = load_dataset(url, index_col=0)

print(f"Dataset shape: {df.shape}")
print(df.head())

# Create a two-way split and print summary
split = create_split(df, target="sales")

print(f"Training:   {split.X_train.shape[0]} rows")
print(f"Test:       {split.X_test.shape[0]} rows")
print(f"Split ratio: {split.X_test.shape[0] / len(df):.0%} test")

# Create a three-way split and print summary
split3 = create_three_way_split(df, target="sales", test_size=0.2, val_size=0.1)

total = len(df)
print(f"Training:   {len(split3.X_train)} rows  ({len(split3.X_train)/total:.0%})")
print(f"Validation: {len(split3.X_val)}  rows  ({len(split3.X_val)/total:.0%})")
print(f"Test:       {len(split3.X_test)}  rows  ({len(split3.X_test)/total:.0%})")

# Compare random vs. stratified splits
split_rand = create_split(df, target="sales")
split_strat = create_stratified_split(df, target="sales", bins=5)

print("Sales distribution — mean and std:")
print(f"  Full dataset: {df['sales'].mean():.2f} ± {df['sales'].std():.2f}")
print(f"  Random train: {split_rand.y_train.mean():.2f} ± {split_rand.y_train.std():.2f}")
print(f"  Strat. train: {split_strat.y_train.mean():.2f} ± {split_strat.y_train.std():.2f}")
print(f"  Random test:  {split_rand.y_test.mean():.2f} ± {split_rand.y_test.std():.2f}")
print(f"  Strat. test:  {split_strat.y_test.mean():.2f} ± {split_strat.y_test.std():.2f}")

# Create a temporal split and print summary
df_temporal = df.copy()
df_temporal["date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="W")

split_time = create_time_split(df_temporal, target="sales", date_column="date")

print(f"Training period: {df_temporal['date'].iloc[:160].min().date()} "
      f"to {df_temporal['date'].iloc[:160].max().date()}")
print(f"Test period:     {df_temporal['date'].iloc[160:].min().date()} "
      f"to {df_temporal['date'].iloc[160:].max().date()}")
print(f"\nTraining rows: {len(split_time.X_train)}")
print(f"Test rows:     {len(split_time.X_test)}")

# Save the split once
split = create_split(df, target="sales")
save_split(split, "../data/splits/advertising")

# Load it in any subsequent script — guaranteed identical
split_loaded = load_split("../data/splits/advertising")

print(f"X_train shape: {split_loaded.X_train.shape}")
print(f"Loaded split matches original: "
      f"{split_loaded.X_train.equals(split.X_train)}")

# Perform cross-validation and print results
cv_results = cross_validate_model(df, target="sales", n_splits=5)
print(cv_results.to_string(index=False))