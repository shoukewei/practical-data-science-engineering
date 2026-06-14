# ======================================================================
# Practical Data Science Engineering — Chapter 01
# The Limits of Notebook-Style Data Science
# Author: Shouke Wei | https://github.com/shoukewei/data-engineering
# Affiliation: Deepsim Intelligence Technology Inc., Canada| https://deepsim.ca
# Run: python chapter-01.py
# ======================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

URL = "https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv"

# --- 1. Notebook-style: everything inline, tightly coupled ---
df = pd.read_csv(URL, index_col=0)
df = df.dropna()
df["TV_scaled"] = (df["TV"] - df["TV"].mean()) / df["TV"].std()
print("Notebook-style output (first 3 rows):")
print(df[["TV", "TV_scaled"]].head(3))

# --- 2. System-oriented: named, purposeful functions ---
def load_dataset(url):       return pd.read_csv(url, index_col=0)
def clean_data(df):          return df.dropna().copy()
def scale_column(df, col):
    v = df[col]; return (v - v.mean()) / v.std()
def train_model(X, y):
    m = LinearRegression(); m.fit(X, y); return m
def save_model(model, path):
    import joblib; joblib.dump(model, path); print(f"Saved: {path}")

df       = load_dataset(URL)
df       = clean_data(df)
df["TV_scaled"] = scale_column(df, "TV")
X = df[["TV_scaled", "radio", "newspaper"]]
y = df["sales"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_model(X_train, y_train)
print(f"\nSystem-oriented pipeline — Test R²: {model.score(X_test, y_test):.4f}")
