# run_pipeline.py
from modules.data_io   import load_dataset
from modules.eda       import missing_summary, skewness_summary
from modules.splitting import create_split, save_split
from modules.pipeline   import PreprocessingPipeline, save_config, summarise_pipeline

# 1. Load data
url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"
df = load_dataset(
    url,
    index_col=0,
    required_columns=["TV", "radio", "newspaper", "sales"],
)

# 2. Quick EDA
print("Missing values:")
ms = missing_summary(df)
print(ms if not ms.empty else "None\n")

print("Skewness:")
print(skewness_summary(df)[["column", "skewness", "high_skew"]])

# 3. Split first
split = create_split(df, target="sales", test_size=0.2, random_state=42)
save_split(split, "./data/splits/advertising")

# 4. Define config
config = {
    "missing": {
        "strategies": {
            "TV":        "median",
            "radio":     "mean",
            "newspaper": 0.0,
        },
        "indicator_columns": [],
    },
    "outliers": {
        "columns":    ["TV", "newspaper"],
        "method":     "iqr",
        "multiplier": 1.5,
    },
    "scaling": {
        "columns": ["TV", "radio", "newspaper"],
        "method":  "standard",
    },
}

# 5. Fit pipeline on training data, transform both splits
pipeline = PreprocessingPipeline(config)
X_train_clean = pipeline.fit_transform(split.X_train)
X_test_clean  = pipeline.transform(split.X_test)

# 6. Save pipeline
save_config(config, "./models/pipeline_config.json")
pipeline.save("./models/preprocessing_pipeline.pkl")

# 7. Summary
summarise_pipeline(pipeline)
print(f"\nFinal training shape: {X_train_clean.shape}")
print(f"Final test shape:     {X_test_clean.shape}")