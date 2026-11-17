from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7, "TotalCharges": 151.65,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 60.0, "TotalCharges": 3600.0,
}


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_probability():
    r = client.post("/predict", json=HIGH_RISK)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in {"low", "medium", "high"}


def test_new_month_to_month_customer_is_riskier_than_loyal_customer():
    high = client.post("/predict", json=HIGH_RISK).json()["churn_probability"]
    low = client.post("/predict", json=LOW_RISK).json()["churn_probability"]
    assert high > low


def test_invalid_contract_is_rejected():
    bad = dict(HIGH_RISK, Contract="Weekly")
    assert client.post("/predict", json=bad).status_code == 422