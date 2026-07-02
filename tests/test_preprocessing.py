# tests/test_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

import preprocessing


def test_load_data_drops_id_and_logs_target():
    X, y = preprocessing.load_data("data/train.csv")
    assert "Id" not in X.columns
    assert "SalePrice" not in X.columns
    assert len(X) == len(y) == 1460
    # log target is far smaller than raw dollars
    assert y.max() < 20


def test_split_columns_covers_all_features():
    X, _ = preprocessing.load_data("data/train.csv")
    num, cat = preprocessing.split_columns(X)
    assert set(num) | set(cat) == set(X.columns)
    assert "GrLivArea" in num
    assert "MSZoning" in cat


def test_compute_defaults_covers_every_column():
    X, _ = preprocessing.load_data("data/train.csv")
    defaults = preprocessing.compute_defaults(X)
    assert set(defaults.keys()) == set(X.columns)
    # categorical default is a string, numeric is a number
    assert isinstance(defaults["MSZoning"], str)
    assert isinstance(defaults["GrLivArea"], (int, float, np.integer, np.floating))


def test_build_pipeline_fits_and_predicts():
    X, y = preprocessing.load_data("data/train.csv")
    pipe = preprocessing.build_pipeline(Ridge(alpha=1.0))
    assert isinstance(pipe, Pipeline)
    pipe.fit(X, y)
    preds = pipe.predict(X.head(5))
    assert len(preds) == 5


def test_compute_metadata_structure():
    import preprocessing
    meta = preprocessing.compute_metadata()
    assert set(meta) >= {"neighborhoods", "price_range", "field_ranges"}
    assert len(meta["neighborhoods"]) > 0
    assert meta["neighborhoods"] == sorted(meta["neighborhoods"])
    assert meta["price_range"]["min"] < meta["price_range"]["max"]
    for col in ("GrLivArea", "TotalBsmtSF", "YearBuilt"):
        r = meta["field_ranges"][col]
        assert r["min"] <= r["max"]
