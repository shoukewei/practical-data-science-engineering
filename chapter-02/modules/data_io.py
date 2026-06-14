# data_io.py
import pandas as pd
from pathlib import Path


def load_dataset(path: str) -> pd.DataFrame:
    """Load a DataFrame from a CSV file."""
    return pd.read_csv(path)


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to a CSV file.
    If the target directory does not exist, create it.
    """
    path_obj = Path(path)

    # Create parent directory if it doesn't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(path_obj, index=False)

    print(f"Saved: {path_obj}")