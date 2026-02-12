import pandas as pd
import numpy as np
import joblib

print("Loading model...")
model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# tạo input rỗng đủ tất cả columns
input_data = pd.DataFrame(columns=model_columns)
input_data.loc[0] = 0   # fill default = 0

# ===== user input =====
input_data["OverallQual"] = 7
input_data["GrLivArea"] = 1800
input_data["GarageCars"] = 2
input_data["TotalBsmtSF"] = 900
input_data["FullBath"] = 2
input_data["YearBuilt"] = 2005
# ======================

print("Predicting...")
pred_log = model.predict(input_data)
pred_price = np.exp(pred_log)

print("Predicted house price: ${:,.0f}".format(pred_price[0]))
