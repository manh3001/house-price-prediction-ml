# 🏠 House Price Prediction -- End-to-End Machine Learning Project

## 📌 Overview

This project builds a machine learning system to predict house prices
using the Kaggle Ames Housing dataset and deploys the trained model as a
REST API using Flask.

This is a full end-to-end ML pipeline including: - Data preprocessing -
Model training - Model serialization - REST API deployment - API testing
with Postman

Perfect for portfolio / graduation project.

## 🎯 Objectives

-   Predict house prices based on house features
-   Build reproducible ML pipeline
-   Deploy ML model as an API
-   Demonstrate real-world ML workflow

## 📊 Dataset

Ames Housing Dataset - 1460 rows - 81 features - Mix of numeric and
categorical data

## 🧠 Machine Learning Pipeline

Numeric features: - Fill missing values with median - StandardScaler
normalization

Categorical features: - Fill missing values with most frequent value -
One-Hot Encoding

After preprocessing: Original shape: (1460, 81)\
Processed shape: \~287 features

## 📈 Model Performance

Ridge Regression\
Final RMSE ≈ 23,798 USD

## 🌐 Deploy Model as API

Run API: python app.py

Server: http://127.0.0.1:5000

## 🔌 API Usage

POST /predict

Example JSON: { "GrLivArea": 1800, "OverallQual": 7, "GarageCars": 2,
"YearBuilt": 2005 }

Response: { "predicted_price": 4937241.88 }

## 🧪 Test API with Postman

1.  Open Postman
2.  POST http://127.0.0.1:5000/predict
3.  Body → raw → JSON
4.  Click Send

## 💡 Real-World Applications

-   Real estate websites
-   Property mobile apps
-   Bank valuation tools
-   Market analysis dashboards

## 🛠 Tech Stack

-   Scikit-learn
-   Pandas, NumPy
-   Joblib
-   Flask
-   Postman

## 👨‍💻 Author

Mạnh Nguyễn
