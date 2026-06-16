# run_feature_engineering_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from modules.data_io import load_dataset
# Import all standalone feature engineering utilities
from modules.feature_engineering import (
    add_ratio_feature,
    add_family_features,
    add_polynomial_features,
    fit_onehot_encoder,
    apply_onehot_encoder,
    fit_ordinal_encoder,
    apply_ordinal_encoder,
    fit_target_encoder,
    apply_target_encoder,
    log_transform_feature,
    bin_feature
)

# Import the pipeline components
from modules.pipeline import PreprocessingPipeline, run_feature_engineering


def main():
    # -------------------------------------------------------------
    # Setup & Data Loading
    # -------------------------------------------------------------
    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = load_dataset(titanic_url)
    df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    
    # Simple upfront fixes for test stability
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    target = "Survived"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Training: {X_train.shape}  |  Test: {X_test.shape}")
    
    # add family features
    X_train_fe = add_family_features(X_train)
    print(X_train_fe[["SibSp", "Parch", "FamilySize", "IsAlone"]].head())

    # add polynomial feature
    X_train_poly = add_polynomial_features(X_train_fe, columns=["Age", "Fare"])
    print(X_train_poly[["Age", "Age_pow2", "Fare", "Fare_pow2"]].describe().round(2))
    
    # one-hot encoding
    # Fit on training data only
    ohe_encoder = fit_onehot_encoder(X_train, columns=["Sex", "Embarked"])

    # Apply to both splits
    X_train_ohe = apply_onehot_encoder(X_train_fe, ohe_encoder)
    X_test_ohe  = apply_onehot_encoder(add_family_features(X_test), ohe_encoder)

    print("Columns after one-hot encoding:")
    print(X_train_ohe.columns.tolist())
    print(f"\nShape: train={X_train_ohe.shape}, test={X_test_ohe.shape}")
    
    # Target-encoded Embarked values (test set)
    te_encoder = fit_target_encoder(
    X_train, column="Embarked", target=y_train, n_folds=5
    )

    X_train_te = apply_target_encoder(X_train, "Embarked", te_encoder, is_train=True)
    X_test_te  = apply_target_encoder(X_test,  "Embarked", te_encoder, is_train=False)

    print("Target-encoded Embarked values (test set):")
    print(X_test_te[["Embarked"]].drop_duplicates().sort_values("Embarked"))
    
    # run feature engineering
    # Global configuration dict containing both preprocessing AND feature engineering
    config_fe = {
        "missing": {
            "strategies": {"Age": "median", "Fare": "median"},
            "indicator_columns": [],
        },
        "outliers": {"columns": ["Fare"], "method": "iqr", "multiplier": 1.5},
        "scaling": {"columns": ["Age", "Fare", "SibSp", "Parch", "FamilySize"], "method": "standard"},
        "feature_engineering": {
            "family_features": True,
            "log_columns": ["Fare"],
            "polynomial": {"columns": ["Age"], "degree": 2},  # <-- FIX: Added quotes around "degree"
            "onehot_columns": ["Sex", "Embarked"],
        },
    }

    print("--- Running Standalone Functional Feature Engineering ---")
    fe_config = config_fe["feature_engineering"]

    # 1. Apply feature engineering — fit encoder on training data
    X_train_fe, ohe_enc = run_feature_engineering(
        X_train, fe_config, is_train=True
    )
    X_test_fe, _ = run_feature_engineering(
        X_test, fe_config, ohe_encoder=ohe_enc, is_train=False
    )

    # 2. Apply preprocessing pipeline (missing, outliers, scaling)
    pp_config = {k: v for k, v in config_fe.items() if k != "feature_engineering"}
    pipeline  = PreprocessingPipeline(pp_config)

    X_train_ready = pipeline.fit_transform(X_train_fe)
    X_test_ready  = pipeline.transform(X_test_fe)

    print(f"Final training shape: {X_train_ready.shape}")
    print(f"Final test shape:     {X_test_ready.shape}")
    print(f"\nFinal columns:")
    print(X_train_ready.columns.tolist())


if __name__ == "__main__":
    main()