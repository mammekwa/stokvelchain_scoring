import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
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

# Explainability: SHAP runs on the classifier, so rows are preprocessed first
# (SMOTE is training-only and is skipped here, exactly as in predict_proba).
imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
explainer = shap.TreeExplainer(model.named_steps["classifier"])

# Plain-language reasons: (raises score, lowers score)
REASONS = {
    "contribution_frequency_score": (
        "Contributes regularly",
        "Has missed contributions",
    ),
    "payout_compliance_rate": (
        "Pays contributions on time",
        "Has paid contributions late",
    ),
    "loan_repayment_ratio": (
        "Strong record of repaying what is owed",
        "Has repaid a low share of what is owed",
    ),
    "credit_amount_normalised": (
        "Amount owed is small relative to capacity",
        "Amount owed is large relative to capacity",
    ),
    "age": (
        "Age slightly increases the score",
        "Age slightly lowers the score",
    ),
}

app = Flask(__name__)


def explain(row):
    """Per-feature contributions to the approval probability, largest first."""
    x = scaler.transform(imputer.transform(row))
    values = explainer.shap_values(x)
    if isinstance(values, list):          # older shap: one array per class
        contrib = values[1][0]
    elif np.ndim(values) == 3:            # newer shap: (rows, features, classes)
        contrib = values[0, :, 1]
    else:
        contrib = values[0]

    reasons = []
    for feature, impact in sorted(zip(FEATURES, contrib), key=lambda p: -abs(p[1])):
        positive, negative = REASONS.get(feature, (feature, feature))
        reasons.append({
            "feature": feature,
            "value": float(row.iloc[0][feature]),
            "impact": round(float(impact), 4),
            "direction": "increases" if impact >= 0 else "decreases",
            "message": positive if impact >= 0 else negative,
        })
    return reasons


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

    # Class 1 = approved (creditworthy)
    probability = float(model.predict_proba(row)[0][1])
    return jsonify(
        probability=round(probability, 4),
        score=round(probability * 100, 1),
        threshold=THRESHOLD,
        eligible=probability >= THRESHOLD,
        reasons=explain(row),
        modelVersion=os.environ.get("MODEL_VERSION", "v1"),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
