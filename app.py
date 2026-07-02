"""Flask REST API for house price prediction."""
from flask import Flask, request, jsonify, render_template

import predict as predictor
import preprocessing

app = Flask(__name__)
_VALID_KEYS = set(predictor._defaults.keys())
_METADATA = preprocessing.compute_metadata()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/metadata")
def metadata():
    return jsonify(_METADATA)


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
    except (KeyError, ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid feature value: {exc}"}), 400
    except Exception as exc:  # pragma: no cover - unexpected
        return jsonify({"error": str(exc)}), 500

    return jsonify({"predicted_price": round(price, 2)})


if __name__ == "__main__":
    print("Starting House Price Prediction API...")
    app.run(debug=True)
