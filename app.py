import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# Must match the column order the model was trained on in Colab.
FEATURES = [
    "contribution_frequency_score",
    "payout_compliance_rate",
    "loan_repayment_ratio",
    "group_tenure_months",
    "credit_amount_normalised",
    "age",
]

MODEL_PATH = os.environ.get("MODEL_PATH", "credit_model.joblib")
API_KEY = os.environ.get("SCORING_API_KEY")  # shared secret with Spring Boot

model = joblib.load(MODEL_PATH)
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="Unauthorized"), 401

    payload = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        return jsonify(error="Missing features", missing=missing), 400

    try:
        row = pd.DataFrame([[float(payload[f]) for f in FEATURES]], columns=FEATURES)
    except (TypeError, ValueError):
        return jsonify(error="All features must be numeric"), 400

    probability = float(model.predict_proba(row)[0][1])
    return jsonify(
        probability=round(probability, 4),
        prediction=int(model.predict(row)[0]),
        modelVersion=os.environ.get("MODEL_VERSION", "v1"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
