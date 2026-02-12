from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# load model khi server start
print("Loading model...")
model = joblib.load("house_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")
print("Model loaded!")

@app.route("/")
def home():
    return "House Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # nhận JSON từ client

        # tạo dataframe đủ columns
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0

        # gán các field user gửi
        for key, value in data.items():
            input_df[key] = value

        # predict
        pred_log = model.predict(input_df)
        pred_price = float(np.exp(pred_log)[0])

        return jsonify({
            "predicted_price": round(pred_price, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(debug=True)
