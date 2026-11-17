import pandas as pd

SERVICE_COLS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + SERVICE_COLS
ADDONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def to_frame(customer: dict) -> pd.DataFrame:
    row = dict(customer)

    # "Yes" -> 1; "No", "No internet service" and "No phone service" -> 0
    for col in YES_NO_COLS:
        row[col] = 1 if row[col] == "Yes" else 0

    row["num_addons"] = sum(row[c] for c in ADDONS)
    row["is_new_customer"] = int(row["tenure"] <= 12)
    row["has_internet"] = int(row["InternetService"] != "No")

    return pd.DataFrame([row])