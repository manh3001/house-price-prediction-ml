# House Price Prediction — End-to-End Machine Learning Project

An end-to-end machine learning system that predicts residential house prices
from the Ames Housing dataset and serves the trained model through a Flask REST
API with an interactive web UI.

The project demonstrates a complete, reproducible ML workflow: data loading and
preprocessing, model selection by cross-validation, artifact serialization, API
deployment, and automated testing.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [API Reference](#api-reference)
- [Web UI](#web-ui)
- [Testing](#testing)
- [Design Notes](#design-notes)
- [Author](#author)

---

## Features

- **Reproducible preprocessing** shared across training and serving via a single
  scikit-learn `Pipeline` (one source of truth in `preprocessing.py`).
- **Automated model selection** across Ridge, Random Forest, and
  HistGradientBoosting using 5-fold cross-validation.
- **Log-transformed target** for a more stable fit on skewed prices, with
  predictions transformed back to USD.
- **Partial-input predictions** — unspecified features fall back to training-set
  defaults (median for numeric, mode for categorical), so incomplete requests
  still return sensible estimates.
- **REST API** with prediction, health, and metadata endpoints.
- **Interactive web UI** served by the same Flask app — no separate frontend
  build required.
- **Automated test suite** covering preprocessing, prediction, and the API.

---

## Tech Stack

| Area              | Tools                                   |
| ----------------- | --------------------------------------- |
| Language          | Python 3.10+                            |
| ML / Data         | scikit-learn, pandas, NumPy             |
| Serialization     | joblib                                  |
| API / Web         | Flask                                   |
| Testing           | pytest                                  |
| Visualization     | matplotlib (exploratory notebook)       |

Exact pinned versions are listed in [`requirements.txt`](requirements.txt).

---

## Project Structure

```
house-price-prediction-ml/
├── app.py                 # Flask REST API + web UI server
├── train.py               # Trains, selects, and serializes the best model
├── predict.py             # Importable + CLI prediction interface
├── preprocessing.py       # Shared data loading, pipeline, defaults, metadata
├── house_price.ipynb      # Exploratory data analysis notebook
├── requirements.txt       # Pinned dependencies
├── data/
│   └── train.csv          # Ames Housing dataset (1460 rows, 81 features)
├── templates/
│   └── index.html         # Web UI page
├── static/
│   ├── app.js             # Front-end logic (live estimate, gauge)
│   └── style.css          # Styling
└── tests/
    ├── test_app.py        # API endpoint tests
    ├── test_predict.py    # Prediction logic tests
    └── test_preprocessing.py
```

> **Note:** The trained artifacts (`house_price_model.pkl`, `model_columns.pkl`,
> `feature_defaults.pkl`) are **git-ignored** and not committed. Run
> `python train.py` to generate them before starting the API.

---

## Getting Started

### Prerequisites

- Python 3.10 or newer
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/manh3001/house-price-prediction-ml.git
cd house-price-prediction-ml
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If activation is blocked, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (required before first run)

```bash
python train.py
```

This loads the dataset, compares the candidate models by cross-validation,
refits the best one, reports the held-out test RMSE, and writes the three
`.pkl` artifacts to the project root.

### 5. Start the server

```bash
python app.py
```

The app runs at **http://127.0.0.1:5000/**.

---

## Usage

### Web UI

Open **http://127.0.0.1:5000/** in a browser and adjust the key house features
(or load a preset) to see a live price estimate and a gauge showing where the
prediction sits within the dataset's price range.

### Command-line prediction

`predict.py` is both importable and runnable. Running it directly prints a
prediction for a built-in example:

```bash
python predict.py
```

As a library:

```python
import predict

price = predict.predict({
    "OverallQual": 7,
    "GrLivArea": 1800,
    "GarageCars": 2,
    "YearBuilt": 2005,
})
print(f"${price:,.0f}")
```

---

## Machine Learning Pipeline

**Target.** The model is trained on `log(SalePrice)` to reduce the effect of
skew in house prices; predictions are exponentiated back to USD before being
returned.

**Preprocessing** (applied uniformly to every candidate model through a shared
`ColumnTransformer`):

| Feature type | Missing values                 | Transformation      |
| ------------ | ------------------------------ | ------------------- |
| Numeric      | Imputed with the median        | `StandardScaler`    |
| Categorical  | Imputed with `"Missing"`       | One-Hot Encoding\*  |

\*`OneHotEncoder(handle_unknown="ignore")` so unseen categories at predict time
do not break inference. Columns are resolved by dtype at fit time, making the
pipeline independent of column order.

Scaling is required by the linear model (Ridge) and is harmless for the
tree-based models, which keeps preprocessing identical across all candidates.

**Model selection.** Three candidates are compared using 5-fold cross-validation
on the training split, scored by RMSE on the log target:

- `Ridge(alpha=1.0)`
- `RandomForestRegressor(n_estimators=300)`
- `HistGradientBoostingRegressor`

The lowest-RMSE model is refit on the training split and evaluated on a held-out
test set (20%). The chosen model and its dollar-scale RMSE are printed by
`train.py`.

**Artifacts** written on completion:

| File                     | Contents                                           |
| ------------------------ | -------------------------------------------------- |
| `house_price_model.pkl`  | Fitted preprocessing + model pipeline              |
| `model_columns.pkl`      | Expected input column order                        |
| `feature_defaults.pkl`   | Per-feature defaults for filling partial requests  |

---

## API Reference

Base URL: `http://127.0.0.1:5000`

### `POST /predict`

Predict a sale price from a JSON object of house features. Any subset of the
dataset's features may be supplied; the rest fall back to training-set defaults.

**Request:**

```json
{ "GrLivArea": 1800, "OverallQual": 7, "GarageCars": 2, "YearBuilt": 2005 }
```

**Response:**

```json
{ "predicted_price": 215000.0 }
```

**Errors:**

| Status | Condition                                             |
| ------ | ----------------------------------------------------- |
| `400`  | Body is not a JSON object                             |
| `400`  | One or more unknown feature names are supplied        |
| `400`  | A feature value cannot be coerced to the right type   |

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"GrLivArea": 1800, "OverallQual": 7, "GarageCars": 2, "YearBuilt": 2005}'
```

### `GET /health`

Liveness/readiness probe.

```json
{ "status": "ok", "model_loaded": true }
```

### `GET /metadata`

Dataset-derived metadata used to drive the web UI (dropdown options, input
ranges, and the price gauge).

```json
{
  "neighborhoods": ["Blmngtn", "Blueste", "..."],
  "price_range": { "min": 34900, "max": 755000 },
  "field_ranges": {
    "GrLivArea":   { "min": 334, "max": 5642 },
    "TotalBsmtSF": { "min": 0,   "max": 6110 },
    "YearBuilt":   { "min": 1872, "max": 2010 }
  }
}
```

---

## Web UI

The interactive estimator is served from `/` by the same Flask app:

- Adjust key numeric features and select a neighborhood.
- Load presets for common house profiles.
- See a live price estimate and a gauge positioning it within the dataset's
  observed price range.

Input ranges and dropdown values are populated dynamically from `/metadata`, so
the UI stays consistent with the underlying dataset without hardcoded values.

---

## Testing

Run the full suite with:

```bash
pytest
```

The tests cover:

- **`test_preprocessing.py`** — data loading, column splitting, pipeline
  construction, defaults, and metadata.
- **`test_predict.py`** — input building, default filling, and prediction.
- **`test_app.py`** — API routes, validation, and error handling.

> Tests that exercise `predict.py`/`app.py` require the trained artifacts. Run
> `python train.py` first if you have not already.

---

## Design Notes

- **Single source of truth for preprocessing.** `preprocessing.py` defines the
  column handling, pipeline, defaults, and UI metadata used by both training and
  serving, preventing train/serve skew.
- **Order-independent pipeline.** Columns are selected by dtype at fit time, so
  the model is robust to differences in input column ordering.
- **Graceful partial inputs.** Missing features are filled from stored defaults,
  which makes the API practical for real-world clients that rarely supply all 80
  features.

---

## Author

**Manh Nguyen**
