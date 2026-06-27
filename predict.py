"""Predict a house price from a partial feature dict (CLI + importable)."""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

_HERE = Path(__file__).resolve().parent
_model = joblib.load(_HERE / "house_price_model.pkl")
_columns = joblib.load(_HERE / "model_columns.pkl")
_defaults = joblib.load(_HERE / "feature_defaults.pkl")


def build_input(features: dict) -> pd.DataFrame:
    """One-row DataFrame in model-column order, defaults filled, overrides applied."""
    row = dict(_defaults)
    for key, value in features.items():
        if key not in row:
            raise KeyError(f"Unknown feature: {key}")
        row[key] = type(_defaults[key])(value)
    return pd.DataFrame([row], columns=list(_columns))


def predict(features: dict) -> float:
    """Return predicted sale price in USD for the given feature overrides."""
    df = build_input(features)
    pred_log = _model.predict(df)
    return float(np.exp(pred_log)[0])


if __name__ == "__main__":
    example = {
        "OverallQual": 7,
        "GrLivArea": 1800,
        "GarageCars": 2,
        "TotalBsmtSF": 900,
        "FullBath": 2,
        "YearBuilt": 2005,
    }
    price = predict(example)
    print("Example features:", example)
    print("Predicted house price: ${:,.0f}".format(price))
