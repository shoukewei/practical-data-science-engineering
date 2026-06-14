# splitting.py
import pandas as pd

def split_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Split a DataFrame into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    target : str
        Name of the target column.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)