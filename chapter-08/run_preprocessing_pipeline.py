# run_preprocessing_pipeline.py
from sklearn.model_selection import train_test_split
from modules.data_io import load_dataset
from modules.preprocessing import remove_outliers  # from Chapter 7
from modules.preprocessing import scale_with_normscaler # this chapter

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

# 1. Load
df = load_dataset(url, index_col=0)

# 2. Define features and target
feature_cols = ["TV", "radio", "newspaper"]
X = df[feature_cols]
y = df["sales"]

# 3. Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Scale — normscaler handles both splits correctly in one call
X_train_scaled, X_test_scaled = scale_with_normscaler(X_train, X_test, 'standard')

print("Pipeline complete.")
print(f"Training shape: {X_train_scaled.shape}")
print(f"Test shape:     {X_test_scaled.shape}")
print(f"\nTraining means (should be ~0):\n{X_train_scaled.mean().round(4)}")