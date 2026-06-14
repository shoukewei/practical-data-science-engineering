# pipeline.py

import pandas as pd

from modules.data_io import load_dataset, save_dataset
from modules.preprocessing import remove_missing, scale_column
from modules.modeling import train_linear_model
from modules.stats import summary_stats
from modules.splitting import split_dataset


url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"


def main():
    # Load
    df = load_dataset(url)

    # Summary statistics
    print("Summary stats with 95% CI:")
    print(
        summary_stats(
            df,
            ["TV", "radio", "newspaper", "sales"]
        ).to_string(index=False)
    )

    # Preprocess
    df_clean = remove_missing(df)
    df_clean["tv_scaled"] = scale_column(df_clean, "TV")

    # Save checkpoint
    save_dataset(
        df_clean,
        "../data/processed/advertising_clean.csv"
    )

    # Split
    X_train, X_test, y_train, y_test = split_dataset(
        df_clean.assign(target=df_clean["sales"]),
        target="target",
        random_state=42
    )

    # Train
    model = train_linear_model(
        X_train[["tv_scaled"]],
        y_train,
        random_state=42
    )

    print("Pipeline completed.")

    return model


if __name__ == "__main__":
    main()