# pipeline.py

import pandas as pd
from modules.data_io import load_dataset, save_dataset

url = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

df = load_dataset(
    url,
    index_col=0,
    required_columns=["TV", "radio", "newspaper", "sales"],
    expected_dtypes={
        "TV":        "float64",
        "Radio":     "float64",
        "Newspaper": "float64",
        "Sales":     "float64",
    }
)

print(df.shape)
print(df.dtypes)

# save the dataset
save_dataset(df, "./data/Advertising.csv")