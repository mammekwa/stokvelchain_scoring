import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.environ.get("MODEL_PATH", "credit_scoring_pipeline.pkl")
METADATA_PATH = os.environ.get("METADATA_PATH", "model_metadata.json")
API_KEY = os.environ.get("SCORING_API_KEY")  # shared secret with Spring Boot

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

# Feature order and decision threshold come straight from the notebook export.
FEATURES = metadata["features"]
THRESHOLD = float(metadata["threshold"])

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", model=metadata.get("model"), threshold=THRESHOLD)


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

    # Class 1 = approved (loan_status in the training data)
    probability = float(model.predict_proba(row)[0][1])
    return jsonify(
        probability=round(probability, 4),
        score=round(probability * 100, 1),
        threshold=THRESHOLD,
        eligible=probability >= THRESHOLD,
        modelVersion=os.environ.get("MODEL_VERSION", "v1"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
