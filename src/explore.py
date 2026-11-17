import pandas as pd

df = pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print()
print(df.dtypes)
print()
print("Churn rate:")
print(df["Churn"].value_counts(normalize=True).round(3))
print()
blank = (df["TotalCharges"].astype(str).str.strip() == "").sum()
print("Rows with blank TotalCharges:", blank)
print()
print(df.head())