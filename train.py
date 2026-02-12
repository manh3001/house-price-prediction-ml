# HOUSE PRICE PREDICTION - TRAIN

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

print("Loading dataset...")
df = pd.read_csv("data/train.csv")

print("Dataset shape:", df.shape)

# 1. DROP COLUMNS KHÔNG CẦN
df = df.drop(["Id"], axis=1)

# 2. TÁCH TARGET
# Log transform target to reduce skewness
y = np.log(df["SalePrice"])
X = df.drop("SalePrice", axis=1)
# 3. TÁCH NUMERIC & CATEGORICAL COLUMNS
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# 4. PREPROCESSING PIPELINE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# 5. TRAIN TEST SPLIT (chưa preprocess!)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. FULL PIPELINE = PREPROCESS + MODEL
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

print("\nTraining Ridge Pipeline...")

ridge_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", Ridge(alpha=1.0))
])

ridge_pipeline.fit(X_train, y_train)

# 7. EVALUATE
import numpy as np
pred_log = ridge_pipeline.predict(X_test)
pred = np.exp(pred_log)
y_true = np.exp(y_test)

rmse = np.sqrt(mean_squared_error(y_true, pred))
print("Final RMSE:", rmse)

# 8. SAVE MODEL
import joblib
joblib.dump(ridge_pipeline, "house_price_model.pkl")
joblib.dump(X.columns, "model_columns.pkl")
print("Pipeline saved!")
