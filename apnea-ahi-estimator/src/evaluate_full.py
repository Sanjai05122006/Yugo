import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, brier_score_loss
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "epochs_features.csv"
MODEL = PROJECT_ROOT / "models" / "apnea_lgbm.pkl"
OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# Load
df = pd.read_csv(DATA)
X = df.drop(columns=["label", "record"], errors="ignore")
y = df["label"].values

# Same split as training (for apples-to-apples)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load(MODEL)

# Predict
probs = model.predict_proba(X_val)[:, 1]
preds = (probs >= 0.5).astype(int)

# ---- Metrics ----
metrics = {}
metrics["accuracy"] = float(accuracy_score(y_val, preds))
metrics["precision"] = float(precision_score(y_val, preds))
metrics["recall"] = float(recall_score(y_val, preds))
metrics["f1"] = float(f1_score(y_val, preds))
metrics["roc_auc"] = float(roc_auc_score(y_val, probs))
metrics["avg_precision_pr_auc"] = float(average_precision_score(y_val, probs))
metrics["brier"] = float(brier_score_loss(y_val, probs))

# Confusion matrix
cm = confusion_matrix(y_val, preds)
tn, fp, fn, tp = cm.ravel()
metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
metrics["sensitivity_recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

# Save metrics JSON
with open(OUT / "metrics_full.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---- Plots (300 DPI) ----

# ROC
fpr, tpr, _ = roc_curve(y_val, probs)
plt.figure(dpi=300)
plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(OUT / "roc.png", dpi=300, bbox_inches="tight")
plt.close()

# Precision-Recall
prec, rec, _ = precision_recall_curve(y_val, probs)
plt.figure(dpi=300)
plt.plot(rec, prec, label=f"AP = {metrics['avg_precision_pr_auc']:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(OUT / "pr_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Confusion Matrix
plt.figure(dpi=300)
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), v in np.ndenumerate(cm):
    plt.text(j, i, int(v), ha="center", va="center")
plt.colorbar()
plt.savefig(OUT / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Probability histogram
plt.figure(dpi=300)
plt.hist(probs, bins=30)
plt.xlabel("Predicted probability (Apnea)")
plt.ylabel("Count")
plt.title("Prediction Confidence Histogram")
plt.savefig(OUT / "prob_hist.png", dpi=300, bbox_inches="tight")
plt.close()

# Print report
print("=== Classification Report ===")
print(classification_report(y_val, preds))
print("\n=== Metrics ===")
for k, v in metrics.items():
    if k != "confusion_matrix":
        print(f"{k}: {v}")
print("Confusion matrix:", metrics["confusion_matrix"])
print("\nSaved to outputs/: roc.png, pr_curve.png, confusion_matrix.png, prob_hist.png, metrics_full.json")
