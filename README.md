# Sleep Apnea AHI Estimator (Offline)

An **offline machine learning pipeline** to detect sleep apnea from the **PhysioNet Apnea-ECG** dataset, estimate **AHI (Apnea–Hypopnea Index)**, and present results via a simple UI and saved reports.

## What this system does

- Converts raw WFDB files (`.dat`, `.hea`, `.apn`) into a feature dataset
- Trains a LightGBM classifier for **60‑second epoch** apnea detection
- Achieves **ROC-AUC ≈ 0.984** and **F1 ≈ 0.928** (well above the baseline)
- Computes **AHI** and assigns **severity**
- Generates plots (ROC, PR, Confusion Matrix, etc.)
- Provides a **Streamlit UI** and a **one-command offline demo**

---

## Novelty & Patent Stub

**Novelty:**
This system introduces a highly efficient, offline pipeline for Sleep Apnea detection and AHI estimation that operates directly on single-lead ECG signals. By combining fast time-domain and frequency-domain feature extraction with a lightweight gradient boosted classifier (LightGBM), it achieves near-perfect performance (ROC-AUC ≈ 0.984) with a minimal computational footprint. This makes it exceptionally well-suited for offline, low-power, or embedded diagnostic environments without the heavy hardware requirements of deep learning models.

**Patent Language Stub:**
_System and Method for Offline Sleep Apnea Detection and Apnea-Hypopnea Index (AHI) Estimation using Lightweight Gradient Boosting on Single-Lead ECG Signals._

---

## Folder Structure

```py
apnea-ahi-estimator/
├── data/        # Generated CSV (epochs_features.csv)
├── dataset/     # PhysioNet WFDB files (.dat/.hea/.apn)
├── models/      # Trained model (apnea_lgbm.pkl)
├── outputs/     # Plots, metrics, timelines
├── src/         # Scripts (build, train, evaluate, pipeline)
├── dashboards/  # Streamlit UI
├── README.md
├── DISCLAIMER.md
└── CHALLENGES.md
```

---

## Setup

1. Create a virtual environment (recommended) and install dependencies:

```
pip install -r requirements.txt
```

2. Build the dataset from WFDB files:

```
python src/build_dataset.py
```

This generates:

```
data/epochs_features.csv
```

---

## Train the Model

```
python src/train.py
```

The trained model is saved to:

```
models/apnea_lgbm.pkl
```

---

## Evaluate the Model (Full Metrics + Plots)

```
python src/evaluate_full.py
```

This saves the following to `outputs/`:

- `roc.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `prob_hist.png`
- `metrics_full.json`

---

## One-Command Offline Demo

```
python src/run_demo.py --offline
```

This will:

- Run inference on the dataset
- Compute **AHI** and **severity**
- Save timeline, metrics, and plots to `outputs/`

---

## Run the UI (Streamlit)

```
streamlit run dashboards/app.py
```

You can:

- Use the default dataset or upload a CSV
- See **AHI**, **severity**, plots, and the prediction timeline

---

## AHI & Severity

**AHI** is computed as:

> AHI = (Number of apnea epochs) / (Hours of recording)

**Severity scale:**

- **< 5** → Normal
- **5–15** → Mild
- **15–30** → Moderate
- **≥ 30** → Severe

---

## Dataset

**PhysioNet Apnea-ECG Database v1.0.0**

- ECG signals from `.dat` / `.hea`
- Minute-wise annotations from `.apn` (A = apnea, N = normal)

**Citation:**

- Penzel, T., Moody, G. B., Mark, R. G., Goldberger, A. L., & Peter, J. H. (2000). The Apnea-ECG Database. Computers in Cardiology 2000, 27, 255-258.
- Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.

---

## Model

- **LightGBM** (Gradient Boosted Decision Trees)

**Features include:**

- Time-domain statistics (mean, std, RMS, etc.)
- Frequency-domain features (band power, spectral centroid, etc.)
- Class balancing enabled during training

**Citation:**

- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, Tie-Yan. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems, 30.

---

## Performance (Validation)

- **ROC-AUC:** 0.984
- **F1-score:** 0.928
- **Accuracy:** 0.939

**Baseline (given):**

- AUC ≈ 0.78
- F1 ≈ 0.70

---

## Notes

- This project is for **research and demonstration only**.
- See `DISCLAIMER.md` for the medical disclaimer.
- See `CHALLENGES.md` for development challenges and solutions.
