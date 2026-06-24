# House Price Prediction -- End-to-End Machine Learning Project

## Overview

This project builds a machine learning system to predict house prices
using the Kaggle Ames Housing dataset and deploys the trained model as a
REST API using Flask.

This is a full end-to-end ML pipeline including: Data preprocessing,
model training, model serialization, REST API deployment, and automated
testing.

Perfect for portfolio / graduation project.

## Objectives

-   Predict house prices based on house features
-   Build reproducible ML pipeline
-   Deploy ML model as an API
-   Demonstrate real-world ML workflow

## Dataset

Ames Housing Dataset - 1460 rows - 81 features - Mix of numeric and
categorical data

## Machine Learning Pipeline

Numeric features:
- Fill missing values with median
- StandardScaler normalization

Categorical features:
- Fill missing values with most frequent value
- One-Hot Encoding

After preprocessing: Original shape: (1460, 81), Processed shape: ~287 features

## Model Performance

Best model selected by 5-fold cross-validation among Ridge, Random Forest, and HistGradientBoosting. See `train.py` output for the chosen model and held-out RMSE.

## Deploy Model as API

Run API: `python app.py`

Server: http://127.0.0.1:5000

Note: Running `python train.py` regenerates the `.pkl` artifacts (now git-ignored).

## API Usage

**POST /predict**

Example JSON:
```json
{ "GrLivArea": 1800, "OverallQual": 7, "GarageCars": 2, "YearBuilt": 2005 }
```

Response:
```json
{ "predicted_price": 215000.0 }
```

Unspecified features fall back to training-set defaults (median/mode), so partial inputs still give sensible predictions.

**Health check**

`GET /health` -> `{"status": "ok", "model_loaded": true}`

## Run Tests

```bash
pytest
```

## Test API with Postman

1.  Open Postman
2.  POST http://127.0.0.1:5000/predict
3.  Body -> raw -> JSON
4.  Click Send

## Real-World Applications

-   Real estate websites
-   Property mobile apps
-   Bank valuation tools
-   Market analysis dashboards

## Tech Stack

-   Scikit-learn
-   Pandas, NumPy
-   Joblib
-   Flask
-   pytest

## Author

Manh Nguyen
