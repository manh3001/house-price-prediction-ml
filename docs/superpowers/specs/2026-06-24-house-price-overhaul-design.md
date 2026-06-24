# House Price Prediction — Full Overhaul Design

**Date:** 2026-06-24
**Status:** Approved

## Overview

The House Price Prediction system trains a model on the Kaggle Ames Housing
dataset (1460 rows, 80 features + target) and serves it as a Flask REST API.
This overhaul fixes correctness bugs, eliminates duplicated/inconsistent
preprocessing logic, upgrades the model, and adds tests.

## Problems Being Fixed

1. **`predict.py` is broken.** It loads `scaler.pkl` and passes a 2-feature
   array to a model that is actually a full `Pipeline` expecting all 80 raw
   columns. It crashes on run.
2. **"Fill with 0" prediction bug.** `app.py` and `test_model.py` initialize
   every unspecified feature to `0` (e.g. `YearBuilt=0`, categorical columns
   set to integer `0`). This produces nonsensical model inputs and wildly wrong
   predictions (the README's ~$4.9M example is a symptom).
3. **Dead artifact.** `scaler.pkl` is unused — scaling lives inside the
   Pipeline.
4. **Missing dependency.** `flask` is not listed in `requirements.txt`.
5. **No input validation / error handling** on the API.
6. **Duplicated preprocessing logic** across `train.py`, `predict.py`, `app.py`
   with no single source of truth.
7. **Empty `.gitignore`** — committed `.pkl` artifacts and a 460KB CSV.

## Architecture

Extract a single shared module so train / predict / serve all agree:

```
preprocessing.py   <- single source of truth
        |
   +----+----+--------+
   |         |        |
train.py  predict.py  app.py
```

### `preprocessing.py` (new)

Responsibilities:
- Load the training CSV and split target (`SalePrice`, log-transformed) from
  features; drop `Id`.
- Identify numeric vs categorical columns.
- Build the `ColumnTransformer` (numeric: median impute + StandardScaler;
  categorical: constant impute + OneHotEncoder(handle_unknown="ignore")).
- Build a full `Pipeline(preprocessor, model)` given a model estimator.
- Compute **feature defaults**: median for numeric columns, mode for
  categorical columns. Returned as a dict and saved to `feature_defaults.pkl`.

Public functions (indicative):
- `load_data(path) -> (X, y)`
- `build_pipeline(model) -> Pipeline`
- `compute_defaults(X) -> dict`

### `train.py` (rewrite)

- Load data via `preprocessing`.
- Compare three models with 5-fold cross-validation on the log target,
  scoring RMSE: `Ridge`, `RandomForestRegressor`, `HistGradientBoostingRegressor`.
- Select the model with the best CV RMSE, refit on the full training split,
  evaluate on a held-out test split, and report RMSE in USD (after `expm1`/`exp`).
- Save artifacts: `house_price_model.pkl` (full pipeline),
  `model_columns.pkl` (feature column order), `feature_defaults.pkl`.
- Print which model won and its CV/test RMSE.

### `predict.py` (rewrite)

- Load model, columns, and defaults.
- Provide a `predict(features: dict) -> float` helper that starts from defaults,
  applies the caller's overrides, builds a one-row DataFrame in column order,
  predicts, and converts back to USD.
- When run as a script, predict on an example feature dict and print the result.

### `app.py` (rewrite)

- Load model, columns, defaults at startup.
- `GET /` — liveness string.
- `GET /health` — returns `{"status": "ok", "model_loaded": true}`.
- `POST /predict`:
  - Parse JSON body (must be an object/dict).
  - Start from `feature_defaults`; apply known keys with type coercion to match
    the training dtype.
  - Reject unknown feature keys with `400`.
  - Predict, `exp` back to USD, return `200 {"predicted_price": <rounded>}`.
  - Malformed/empty JSON → `400` with message; unexpected errors → `500`.

## Data Flow (API prediction)

```
JSON dict
  -> validate is object, keys are known features (else 400)
  -> copy feature_defaults
  -> overlay user-provided values (type-coerced)
  -> one-row DataFrame in model_columns order
  -> pipeline.predict() (log space)
  -> exp() -> USD
  -> 200 {"predicted_price": rounded}
```

## Error Handling

| Condition | Response |
|-----------|----------|
| Body not valid JSON / not an object | 400, `{"error": "..."}` |
| Unknown feature key | 400, lists offending key(s) |
| Value fails type coercion | 400, names the field |
| Model/internal error | 500, `{"error": "..."}` |
| Success | 200, `{"predicted_price": float}` |

## Testing

`tests/test_model.py` (pytest):
- Artifacts load (`model`, `model_columns`, `feature_defaults`).
- `predict.predict({})` (all defaults) returns a value in a sane range
  ($50k–$800k).
- Supplying a higher `OverallQual` increases the prediction vs default.
- Defaults dict covers every model column.
- API (Flask test client): `/health` returns 200; `/predict` with a valid body
  returns 200 + numeric `predicted_price`; unknown key returns 400; malformed
  body returns 400.

## Cleanup / Housekeeping

- Delete `scaler.pkl`.
- Add `flask` and `pytest` to `requirements.txt`.
- Populate `.gitignore`: `__pycache__/`, `*.pyc`, `*.pkl`.
- Update `README.md`: chosen model + real RMSE, defaults behavior, corrected
  example request/response, `/health` endpoint, test instructions.

## Out of Scope (YAGNI)

- Hyperparameter tuning beyond model selection.
- Feature engineering beyond what the current pipeline does.
- Containerization / deployment config.
- Front-end UI.

## Success Criteria

- `python train.py` runs clean, picks a model, saves three artifacts, reports
  a realistic RMSE.
- `python predict.py` runs without crashing and prints a realistic price.
- `python app.py` serves; `/predict` with partial input returns a realistic
  price (not millions), and bad input returns 4xx.
- `pytest` passes.
