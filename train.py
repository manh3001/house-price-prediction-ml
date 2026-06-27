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
