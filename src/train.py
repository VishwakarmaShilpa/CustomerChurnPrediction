import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
)

RANDOM_STATE = 42
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/churn_clean.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()
print("Categorical columns:", cat_cols)
print("Numeric columns:", len(num_cols))
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Churn rate train/test:", round(y_train.mean(), 3), "/", round(y_test.mean(), 3))
print()


def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    ),
    "gradient_boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=200, class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows = []

for name, model in models.items():
    pipe = make_pipeline(model)

    
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    # Fit on the full training set, score on the untouched test set
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    rows.append({
        "model": name,
        "cv_auc": round(cv_auc.mean(), 3),
        "test_auc": round(roc_auc_score(y_test, proba), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "accuracy": round(accuracy_score(y_test, pred), 3),
    })

    joblib.dump(pipe, MODELS_DIR / f"{name}.joblib")
    print("done:", name)

results = pd.DataFrame(rows).sort_values("test_auc", ascending=False)
results.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

print()
print(results.to_string(index=False))