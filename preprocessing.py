# preprocessing.py
"""Shared preprocessing for the House Price Prediction system.

Single source of truth for column splits, the sklearn Pipeline, and the
per-feature default values used to fill unspecified inputs at predict time.
"""
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/train.csv"
TARGET = "SalePrice"


def load_data(path: str = DATA_PATH):
    """Load the dataset, returning (X, y) with y = log(SalePrice)."""
    df = pd.read_csv(path)
    df = df.drop(["Id"], axis=1)
    y = np.log(df[TARGET])
    X = df.drop(TARGET, axis=1)
    return X, y


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) as lists of names."""
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=object).columns.tolist()
    return numeric_cols, categorical_cols


def build_pipeline(model) -> Pipeline:
    """Build a full preprocess+model pipeline for the given estimator."""
    # Columns are resolved at fit time from the training frame; we pass
    # selectors by dtype so the pipeline is independent of column order.
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,
         make_column_selector(dtype_include=np.number)),
        ("cat", categorical_transformer,
         make_column_selector(dtype_include=object)),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def compute_defaults(X: pd.DataFrame) -> dict:
    """Median for numeric columns, mode for categorical columns."""
    numeric_cols, categorical_cols = split_columns(X)
    defaults = {}
    for col in numeric_cols:
        defaults[col] = X[col].median()
    for col in categorical_cols:
        mode = X[col].mode()
        defaults[col] = str(mode.iloc[0]) if not mode.empty else "Missing"
    return defaults
