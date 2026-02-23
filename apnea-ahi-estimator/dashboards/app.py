import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from src.pipeline import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_ROOT / "data" / "epochs_features.csv"
OUTPUTS = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="Sleep Apnea AHI Estimator", layout="wide")
st.title("🫁 Sleep Apnea AHI Estimator (Offline)")

uploaded = st.file_uploader("Upload features CSV", type=["csv"])

if uploaded is not None:
    tmp_path = OUTPUTS / "uploaded.csv"
    df = pd.read_csv(uploaded)
    df.to_csv(tmp_path, index=False)
    csv_path = tmp_path
else:
    st.info("Using default dataset")
    csv_path = DEFAULT_CSV

if st.button("Run Analysis"):
    with st.spinner("Running pipeline..."):
        results = run_pipeline(csv_path)

    st.success("Done!")

    col1, col2, col3 = st.columns(3)
    col1.metric("AHI", f"{results['ahi']:.2f}")
    col2.metric("Severity", results["severity"])
    col3.metric("Apnea Epochs", int(results["apnea_epochs"]))

    st.subheader("ROC Curve")
    if results["roc_path"] is not None:
        st.image(str(results["roc_path"]))

    st.subheader("SHAP Feature Importance")
    st.image(str(results["shap_path"]))

    st.subheader("Spectrogram")
    st.image(str(results["spec_path"]))

    st.subheader("Timeline (first rows)")
    timeline = pd.read_csv(results["timeline_path"])
    st.dataframe(timeline.head(200))

    st.subheader("Confusion Matrix")
    if results.get("confusion_path") is not None:
        st.image(str(results["confusion_path"]))
