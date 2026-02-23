import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "apnea_lgbm.pkl"
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

def run_pipeline(features_csv: Path):
    # Load
    df = pd.read_csv(features_csv)
    y_true = df["label"] if "label" in df.columns else None
    X = df.drop(columns=["label", "record"], errors="ignore")

    model = joblib.load(MODEL_PATH)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    # ---- AHI ----
    num_epochs = len(preds)
    apnea_epochs = int(preds.sum())
    hours = num_epochs / 60.0
    ahi = apnea_epochs / hours if hours > 0 else 0.0

    if ahi < 5:
        severity = "Normal"
    elif ahi < 15:
        severity = "Mild"
    elif ahi < 30:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Save timeline
    timeline = pd.DataFrame({
        "epoch": np.arange(num_epochs),
        "apnea_prob": probs,
        "pred": preds
    })
    timeline_path = OUTPUTS / "timeline.csv"
    timeline.to_csv(timeline_path, index=False)

    # ---- ROC ----
    roc_path = None
    if y_true is not None:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(dpi=300)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],"--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        roc_path = OUTPUTS / "roc.png"
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        roc_auc = None

        # ---- Confusion Matrix ----
    conf_path = None
    if y_true is not None:
        cm = confusion_matrix(y_true, preds)

        plt.figure(dpi=300)
        plt.imshow(cm)
        plt.title("Confusion Matrix (threshold = 0.5)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, int(v), ha="center", va="center")
        plt.colorbar()

        conf_path = OUTPUTS / "confusion_matrix.png"
        plt.savefig(conf_path, dpi=300, bbox_inches="tight")
        plt.close()

        
    # ---- SHAP (on a subset for speed) ----
    explainer = shap.TreeExplainer(model)
    sample_X = X.sample(min(500, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample_X)

    plt.figure()
    shap.summary_plot(shap_values, sample_X, show=False)
    shap_path = OUTPUTS / "shap.png"
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ---- Spectrogram (from one raw signal epoch if available in features CSV) ----
    # If you don't have raw signal here, we’ll just fake a spectrogram from one feature column
    sig = X.iloc[:, 0].values
    f, t, Sxx = spectrogram(sig, fs=1.0)

    plt.figure(dpi=300)
    plt.pcolormesh(t, f, Sxx, shading="gouraud")
    plt.ylabel("Frequency")
    plt.xlabel("Time")
    plt.title("Spectrogram (proxy)")
    spec_path = OUTPUTS / "spectrogram.png"
    plt.savefig(spec_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save metrics
    metrics = {
        "ahi": float(ahi),
        "severity": severity,
        "apnea_epochs": int(apnea_epochs),
        "num_epochs": int(num_epochs),
        "roc_auc": None if roc_auc is None else float(roc_auc),
    }

    metrics_path = OUTPUTS / "metrics.json"
    pd.Series(metrics).to_json(metrics_path)

    return {
        "ahi": ahi,
        "severity": severity,
        "apnea_epochs": apnea_epochs,
        "num_epochs": num_epochs,
        "roc_auc": roc_auc,
        "timeline_path": timeline_path,
        "roc_path": roc_path,
        "shap_path": shap_path,
        "spec_path": spec_path,
        "confusion_path": conf_path,
        "metrics_path": metrics_path,
    }
