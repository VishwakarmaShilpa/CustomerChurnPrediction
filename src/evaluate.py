import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve, ConfusionMatrixDisplay, precision_recall_curve,
    precision_score, recall_score, f1_score, classification_report,
)

RANDOM_STATE = 42
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv("data/processed/churn_clean.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"]

# same split as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

comparison = pd.read_csv(MODELS_DIR / "model_comparison.csv")
best_name = comparison.sort_values("test_auc", ascending=False).iloc[0]["model"]
print("Best model by test AUC:", best_name)
pipe = joblib.load(MODELS_DIR / f"{best_name}.joblib")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
threshold = float(thr[np.argmax(f1s)])
print("Chosen threshold (max F1 on training folds):", round(threshold, 3))


proba = pipe.predict_proba(X_test)[:, 1]
pred_default = (proba >= 0.5).astype(int)
pred = (proba >= threshold).astype(int)

print()
print("Test AUC:", round(roc_auc_score(y_test, proba), 3))
print()
print("                 recall  precision      f1")
for label, p in [("threshold 0.50 ", pred_default), (f"threshold {threshold:.2f}", pred)]:
    print(
        f"{label}   {recall_score(y_test, p):.3f}      "
        f"{precision_score(y_test, p):.3f}   {f1_score(y_test, p):.3f}"
    )
print()
print(classification_report(y_test, pred, target_names=["Stayed", "Churned"]))

churners = int(y_test.sum())
caught = int(((pred == 1) & (y_test == 1)).sum())
flagged = int(pred.sum())
print(f"Of {churners} real churners in the test set, the model caught {caught}.")
print(f"It flagged {flagged} customers in total.")


fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=["Stayed", "Churned"], ax=ax, cmap="Blues"
)
ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curve")
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=150)
plt.close(fig)

imp = permutation_importance(
    pipe, X_test, y_test, scoring="roc_auc", n_repeats=10,
    random_state=RANDOM_STATE, n_jobs=-1,
)
importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
print()
print("Top 10 churn drivers (drop in AUC when the column is shuffled):")
print(importance.head(10).round(4).to_string())

top = importance.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.barh(top.index, top.values, color="#14b8a6")
ax.set_xlabel("Drop in ROC-AUC when shuffled")
ax.set_title("Top 10 churn drivers")
fig.tight_layout()
fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
plt.close(fig)


joblib.dump(pipe, MODELS_DIR / "best_model.joblib")
with open(MODELS_DIR / "threshold.json", "w") as f:
    json.dump(
        {
            "model": best_name,
            "threshold": round(threshold, 4),
            "test_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "test_recall": round(float(recall_score(y_test, pred)), 4),
            "test_precision": round(float(precision_score(y_test, pred)), 4),
        },
        f,
        indent=2,
    )
print()
print("Saved: models/best_model.joblib, models/threshold.json, reports/*.png")