import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "apnea_lgbm.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "epochs_features.csv"

# Load model
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Sleep Apnea AHI Estimator", layout="wide")
st.title("🫁 Sleep Apnea AHI Estimator (Offline Demo)")

st.write("Upload a CSV of epoch features or use the default dataset.")

# File uploader
uploaded = st.file_uploader("Upload features CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(DATA_PATH)
    st.info("Using default dataset: data/epochs_features.csv")

# Check required columns
if "label" in df.columns:
    X = df.drop(columns=["label", "record"], errors="ignore")
else:
    X = df.drop(columns=["record"], errors="ignore")

# Predict
probs = model.predict_proba(X)[:, 1]
preds = (probs > 0.5).astype(int)

# Compute AHI
num_epochs = len(preds)
apnea_epochs = preds.sum()
hours = num_epochs / 60.0
ahi = apnea_epochs / hours if hours > 0 else 0

# Severity
if ahi < 5:
    severity = "Normal"
elif ahi < 15:
    severity = "Mild"
elif ahi < 30:
    severity = "Moderate"
else:
    severity = "Severe"

# Show results
st.subheader("📊 Results")
col1, col2, col3 = st.columns(3)
col1.metric("AHI", f"{ahi:.2f}")
col2.metric("Severity", severity)
col3.metric("Apnea Epochs", int(apnea_epochs))

# Show timeline plot
st.subheader("🕒 Apnea Timeline (per 60s epoch)")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(preds, label="Apnea Prediction (1=Apnea, 0=Normal)")
ax.set_xlabel("Epoch (minutes)")
ax.set_ylabel("Prediction")
ax.set_yticks([0, 1])
ax.legend()
st.pyplot(fig)

# Show probability histogram
st.subheader("📈 Prediction Confidence")
fig2, ax2 = plt.subplots()
ax2.hist(probs, bins=30)
ax2.set_xlabel("Apnea Probability")
ax2.set_ylabel("Count")
st.pyplot(fig2)

# Show raw table (optional)
with st.expander("Show predictions table"):
    out_df = df.copy()
    out_df["apnea_prob"] = probs
    out_df["pred"] = preds
    st.dataframe(out_df.head(500))
