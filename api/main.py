import json
from pathlib import Path

import joblib
from fastapi import FastAPI

from api.features import to_frame
from api.schemas import CustomerIn

BASE = Path(__file__).resolve().parents[1]
model = joblib.load(BASE / "models" / "best_model.joblib")
meta = json.loads((BASE / "models" / "threshold.json").read_text())
THRESHOLD = meta["threshold"]

app = FastAPI(title="Customer Churn Prediction API")


@app.get("/")
def health():
    return {
        "status": "ok",
        "model": meta["model"],
        "threshold": THRESHOLD,
        "test_auc": meta["test_auc"],
    }


@app.post("/predict")
def predict(customer: CustomerIn):
    X = to_frame(customer.model_dump())
    probability = float(model.predict_proba(X)[0, 1])

    if probability >= THRESHOLD:
        risk = "high"
    elif probability >= THRESHOLD / 2:
        risk = "medium"
    else:
        risk = "low"

    return {
        "churn_probability": round(probability, 4),
        "will_churn": probability >= THRESHOLD,
        "risk_level": risk,
    }