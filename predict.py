import joblib
import numpy as np

model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Enter house features (example values)")
grliv = float(input("Above ground living area: "))
garage = float(input("Garage area: "))

sample = np.array([[grliv, garage]])
sample = scaler.transform(sample)

price = model.predict(sample)
print("Predicted house price:", price[0])
