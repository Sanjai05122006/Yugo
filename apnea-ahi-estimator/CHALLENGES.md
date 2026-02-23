# Challenges Encountered and How We Solved Them

This document summarizes the key technical and data-engineering challenges faced during development, and the solutions or mitigations applied.

---

## 1. No Pre-built CSV Dataset

**Problem:**
The PhysioNet Apnea-ECG dataset is provided in WFDB format (`.dat`, `.hea`, `.apn`), not as a ready-to-use CSV for machine learning.

**Solution:**
We built a preprocessing pipeline using `wfdb` to load signals and annotations, segment recordings into **60-second epochs**, extract **time- and frequency-domain features**, and export a clean `epochs_features.csv` for model training.

---

## 2. Annotation File Naming Mismatch

**Problem:**
Signal files (e.g., `a01.dat`) did not always have a matching `a01.apn`. Labels sometimes appeared as `a01er.apn`, `a01r.apn`, or `{record}.apn`.

**Solution:**
Implemented a resolver that tries the following in order:

1. `{record}er.apn`
2. `{record}r.apn`
3. `{record}.apn`

The first existing file is used.

---

## 3. Labels Stored in the Wrong Field

**Problem:**
Initial parsing read `ann.aux_note`, which is empty or non-informative for this dataset, producing all-zero labels.

**Solution:**
Inspected the annotation structure and switched to `ann.symbol`, mapping:

- `A` → 1 (apnea)
- `N` → 0 (normal)

---

## 4. Signal–Label Misalignment

**Problem:**
Truncating by `min(len(segments), len(labels))` caused loss of apnea epochs and skewed class balance.

**Solution:**
Treated labels as the **ground-truth timeline** (1 label per minute) and sliced the signal using:

- `[i * 60 * fs : (i + 1) * 60 * fs]`

Processing stops only when the signal ends, preserving correct alignment.

---

## 5. Python Import Path Issues (CLI vs UI)

**Problem:**
Running scripts from different directories (`src/`, `dashboards/`) caused `ModuleNotFoundError` due to differing import roots.

**Solution:**

- Used direct imports within `src/` (e.g., `from pipeline import ...`)
- Added the project root to `sys.path` in the Streamlit app

---

## 6. Class Imbalance

**Problem:**
The dataset is not perfectly balanced between apnea and normal epochs, risking biased training.

**Solution:**

- Enabled `class_weight="balanced"` in LightGBM
- Emphasized **ROC-AUC**, **PR-AUC**, and **recall** during evaluation

---

## 7. Risk of Optimistic Validation

**Problem:**
Random epoch-wise splits can leak subject-specific patterns between train and validation sets.

**Mitigation:**
This limitation is documented. The current split is suitable for baseline comparison; a **record-wise split** can be added if stricter evaluation is required.

---

## 8. Explainability Requirement

**Problem:**
The system needed interpretability, not just predictions.

**Solution:**
Integrated **SHAP (TreeExplainer)** to produce feature-importance summaries and explain model behavior.

---

## 9. One-Command Offline Demo

**Problem:**
The project required a single command to run the full pipeline and save all artifacts.

**Solution:**
Implemented:

```
python src/run_demo.py --offline
```

This runs inference, computes **AHI** and **severity**, generates plots, and saves all outputs to `outputs/`.

---

## Conclusion

Most challenges stemmed from raw biomedical data formats, annotation quirks, alignment issues, and packaging ML for reproducibility. Addressing these resulted in a **robust, fully offline, explainable end-to-end system** that significantly outperforms the baseline.
