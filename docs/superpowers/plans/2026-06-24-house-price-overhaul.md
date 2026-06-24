# House Price Prediction Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix correctness bugs, centralize preprocessing, upgrade the model via cross-validated selection, and add tests to the House Price Prediction system.

**Architecture:** A single `preprocessing.py` module is the source of truth for column splits, the sklearn `Pipeline`, and feature defaults. `train.py` selects the best of three models by CV and saves three artifacts (`house_price_model.pkl`, `model_columns.pkl`, `feature_defaults.pkl`). `predict.py` and `app.py` load those artifacts and fill unspecified features from defaults instead of `0`.

**Tech Stack:** Python 3.12, scikit-learn 1.5.1, pandas, numpy, joblib, Flask, pytest.

## Global Constraints

- Target is modeled in log space: train on `np.log(SalePrice)`, convert predictions back with `np.exp`.
- Drop the `Id` column before modeling.
- Numeric preprocessing: `SimpleImputer(strategy="median")` + `StandardScaler`.
- Categorical preprocessing: `SimpleImputer(strategy="constant", fill_value="Missing")` + `OneHotEncoder(handle_unknown="ignore")`.
- Artifacts live in the project root with exact names: `house_price_model.pkl`, `model_columns.pkl`, `feature_defaults.pkl`.
- Training data path: `data/train.csv`.
- All file paths in code are relative to the project root (scripts are run from root).

---

### Task 1: Shared preprocessing module

**Files:**
- Create: `preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces:
  - `load_data(path: str = "data/train.csv") -> tuple[pd.DataFrame, pd.Series]` — returns `(X, y)` where `y = np.log(SalePrice)` and `X` has `Id` and `SalePrice` dropped.
  - `split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]` — returns `(numeric_cols, categorical_cols)`.
  - `build_pipeline(model) -> sklearn.pipeline.Pipeline` — returns a `Pipeline([("preprocessor", ColumnTransformer), ("model", model)])`.
  - `compute_defaults(X: pd.DataFrame) -> dict` — median for numeric columns, mode (first value of `.mode()`) for categorical columns; one entry per column in `X`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'preprocessing'`

- [ ] **Step 3: Write the implementation**

```python
# preprocessing.py
"""Shared preprocessing for the House Price Prediction system.

Single source of truth for column splits, the sklearn Pipeline, and the
per-feature default values used to fill unspecified inputs at predict time.
"""
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
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


def split_columns(X: pd.DataFrame):
    """Return (numeric_cols, categorical_cols) as lists of names."""
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
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
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,
         make_column_selector_numeric()),
        ("cat", categorical_transformer,
         make_column_selector_categorical()),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def make_column_selector_numeric():
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_include=np.number)


def make_column_selector_categorical():
    from sklearn.compose import make_column_selector
    return make_column_selector(dtype_include=object)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_preprocessing.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add shared preprocessing module with feature defaults"
```

---

### Task 2: Rewrite training with cross-validated model selection

**Files:**
- Modify: `train.py` (full rewrite)
- Produces artifacts: `house_price_model.pkl`, `model_columns.pkl`, `feature_defaults.pkl`

**Interfaces:**
- Consumes: `preprocessing.load_data`, `preprocessing.build_pipeline`, `preprocessing.compute_defaults`.
- Produces (on disk):
  - `house_price_model.pkl` — fitted `Pipeline`.
  - `model_columns.pkl` — `pandas.Index` of feature column names in training order.
  - `feature_defaults.pkl` — `dict` from `compute_defaults`.

- [ ] **Step 1: Write the implementation**

```python
# train.py
"""Train and select the best House Price model, then save artifacts."""
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

import preprocessing

CANDIDATES = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
}


def main():
    print("Loading dataset...")
    X, y = preprocessing.load_data()
    print("Dataset shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nComparing models via 5-fold CV (neg RMSE on log target)...")
    best_name, best_score, best_pipeline = None, np.inf, None
    for name, model in CANDIDATES.items():
        pipe = preprocessing.build_pipeline(model)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=5,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        rmse_log = -scores.mean()
        print(f"  {name:22s} CV log-RMSE = {rmse_log:.4f}")
        if rmse_log < best_score:
            best_name, best_score, best_pipeline = name, rmse_log, pipe

    print(f"\nBest model: {best_name} (CV log-RMSE = {best_score:.4f})")
    print("Refitting best model on training split...")
    best_pipeline.fit(X_train, y_train)

    pred = np.exp(best_pipeline.predict(X_test))
    y_true = np.exp(y_test)
    rmse_usd = np.sqrt(mean_squared_error(y_true, pred))
    print(f"Held-out test RMSE: ${rmse_usd:,.0f}")

    print("\nSaving artifacts...")
    joblib.dump(best_pipeline, "house_price_model.pkl")
    joblib.dump(X.columns, "model_columns.pkl")
    joblib.dump(preprocessing.compute_defaults(X), "feature_defaults.pkl")
    print("Saved house_price_model.pkl, model_columns.pkl, feature_defaults.pkl")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training to generate artifacts**

Run: `python train.py`
Expected: prints CV log-RMSE for three models, a "Best model:" line, a "Held-out test RMSE: $..." line (realistic, roughly $17k–$25k), and "Saved ..." Confirms the three `.pkl` files exist.

- [ ] **Step 3: Verify artifacts exist**

Run: `python -c "import joblib; m=joblib.load('house_price_model.pkl'); c=joblib.load('model_columns.pkl'); d=joblib.load('feature_defaults.pkl'); print(len(c), 'cols', len(d), 'defaults')"`
Expected: `79 cols 79 defaults` (80 features minus none — SalePrice and Id already dropped; should print equal counts).

- [ ] **Step 4: Commit**

```bash
git add train.py
git commit -m "feat: cross-validated model selection and defaults artifact in training"
```

---

### Task 3: Rewrite predict.py with defaults-based input

**Files:**
- Modify: `predict.py` (full rewrite)
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: artifacts from Task 2; `model_columns.pkl`, `feature_defaults.pkl`.
- Produces:
  - `build_input(features: dict) -> pd.DataFrame` — one-row frame in `model_columns` order, starting from defaults, with `features` overlaid.
  - `predict(features: dict) -> float` — predicted price in USD.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
import predict


def test_predict_all_defaults_is_realistic():
    price = predict.predict({})
    assert 50_000 < price < 800_000


def test_higher_quality_raises_price():
    low = predict.predict({"OverallQual": 3})
    high = predict.predict({"OverallQual": 10})
    assert high > low


def test_build_input_has_all_columns_one_row():
    df = predict.build_input({"GrLivArea": 2000})
    assert df.shape[0] == 1
    assert df.loc[0, "GrLivArea"] == 2000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predict.py -v`
Expected: FAIL — `predict.build_input` / current `predict.py` has no such functions (AttributeError) or import error.

- [ ] **Step 3: Write the implementation**

```python
# predict.py
"""Predict a house price from a partial feature dict (CLI + importable)."""
import numpy as np
import pandas as pd
import joblib

_model = joblib.load("house_price_model.pkl")
_columns = joblib.load("model_columns.pkl")
_defaults = joblib.load("feature_defaults.pkl")


def build_input(features: dict) -> pd.DataFrame:
    """One-row DataFrame in model-column order, defaults filled, overrides applied."""
    row = dict(_defaults)
    for key, value in features.items():
        if key not in row:
            raise KeyError(f"Unknown feature: {key}")
        row[key] = value
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_predict.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the script**

Run: `python predict.py`
Expected: prints the example dict and a realistic "Predicted house price: $..." (not millions).

- [ ] **Step 6: Commit**

```bash
git add predict.py tests/test_predict.py
git commit -m "fix: rewrite predict.py to use feature defaults instead of crashing"
```

---

### Task 4: Rewrite the Flask API with validation

**Files:**
- Modify: `app.py` (full rewrite)
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `predict.build_input`, `predict.predict`, `predict._defaults`.
- Produces: Flask `app` object with routes `/`, `/health`, `/predict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
import app as app_module


def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_health_ok():
    resp = client().get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_predict_valid_returns_price():
    resp = client().post("/predict", json={"OverallQual": 8, "GrLivArea": 2000})
    assert resp.status_code == 200
    price = resp.get_json()["predicted_price"]
    assert 50_000 < price < 1_000_000


def test_predict_unknown_key_is_400():
    resp = client().post("/predict", json={"NotARealColumn": 5})
    assert resp.status_code == 400


def test_predict_non_object_body_is_400():
    resp = client().post("/predict", json=[1, 2, 3])
    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_app.py -v`
Expected: FAIL — current `app.py` has no `/health` route and returns 200 with `{"error": ...}` on bad input, so status-code assertions fail.

- [ ] **Step 3: Write the implementation**

```python
# app.py
"""Flask REST API for house price prediction."""
from flask import Flask, request, jsonify

import predict as predictor

app = Flask(__name__)
_VALID_KEYS = set(predictor._defaults.keys())


@app.route("/")
def home():
    return "House Price Prediction API is running!"


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a JSON object of features."}), 400

    unknown = [k for k in data if k not in _VALID_KEYS]
    if unknown:
        return jsonify({"error": f"Unknown feature(s): {unknown}"}), 400

    try:
        price = predictor.predict(data)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid feature value: {exc}"}), 400
    except Exception as exc:  # pragma: no cover - unexpected
        return jsonify({"error": str(exc)}), 500

    return jsonify({"predicted_price": round(price, 2)})


if __name__ == "__main__":
    print("Loading model... model loaded!")
    app.run(debug=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_app.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tests across the four test files)

- [ ] **Step 6: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: validating Flask API with /health and defaults-based prediction"
```

---

### Task 5: Cleanup, dependencies, and docs

**Files:**
- Delete: `scaler.pkl`
- Modify: `requirements.txt`, `.gitignore`, `README.md`, `test_model.py` (delete — replaced by `tests/`)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing importable.

- [ ] **Step 1: Delete dead artifacts**

```bash
git rm scaler.pkl test_model.py
```

- [ ] **Step 2: Update requirements.txt**

Replace file contents with:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
matplotlib==3.9.0
joblib==1.4.2
flask==3.0.3
pytest==8.2.0
```

- [ ] **Step 3: Populate .gitignore**

Replace file contents with:

```
__pycache__/
*.pyc
*.pkl
.pytest_cache/
```

- [ ] **Step 4: Update README.md**

Update these sections to match reality:
- **Model Performance:** replace the Ridge/$23,798 line with: "Best model selected by 5-fold cross-validation among Ridge, Random Forest, and HistGradientBoosting. See `train.py` output for the chosen model and held-out RMSE."
- **API Usage:** correct the example response to a realistic value, e.g. `{ "predicted_price": 215000.0 }`, and note that unspecified features fall back to training-set defaults (median/mode), so partial inputs still give sensible predictions.
- Add a **Health check** line: `GET /health -> {"status": "ok", "model_loaded": true}`.
- Add a **Run tests** line: `pytest`.
- Add a note that running `python train.py` regenerates the `.pkl` artifacts (now git-ignored).

- [ ] **Step 5: Verify nothing imports the removed module**

Run: `python -m pytest -v && python predict.py`
Expected: all tests pass and `predict.py` prints a realistic price.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove dead artifacts, add flask/pytest deps, update gitignore and docs"
```

---

## Notes on Execution Order

Task 2 must run (`python train.py`) before Tasks 3 and 4 tests, because those
tests load the `.pkl` artifacts. Do not reorder.
