import pandas as pd
from pathlib import Path

RAW = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUT = Path("data/processed/churn_clean.csv")

df = pd.read_csv(RAW)

# 1. The ID column has no predictive value
df = df.drop(columns=["customerID"])

# 2. TotalCharges is text because 11 rows are blank
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
missing = df["TotalCharges"].isna()
print("Blank TotalCharges rows:", int(missing.sum()))
print("Their tenure values:", df.loc[missing, "tenure"].unique())
df.loc[missing, "TotalCharges"] = 0.0  # new customers have not been billed yet

# 3. Target: Yes/No -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# 4. Merge "No internet service" / "No phone service" into "No"
service_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
for c in service_cols:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})

# 5. Yes/No columns -> 1/0
yes_no = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"] + service_cols
for c in yes_no:
    df[c] = df[c].map({"Yes": 1, "No": 0})

# 6. Feature engineering
addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]
df["num_addons"] = df[addons].sum(axis=1)
df["is_new_customer"] = (df["tenure"] <= 12).astype(int)
df["has_internet"] = (df["InternetService"] != "No").astype(int)

# 7. Save
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print()
print("Saved:", OUT, df.shape)
print("Missing values left:", int(df.isna().sum().sum()))
print("Churn rate:", round(float(df["Churn"].mean()), 3))
print()
print("Churn rate by Contract:")
print(df.groupby("Contract")["Churn"].mean().round(3))
print()
print("Churn rate by new customer (tenure <= 12 months):")
print(df.groupby("is_new_customer")["Churn"].mean().round(3))
print()
print("Churn rate by PaymentMethod:")
print(df.groupby("PaymentMethod")["Churn"].mean().round(3))