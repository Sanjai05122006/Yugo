import wfdb
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

def load_record(name: str):
    path = DATASET_DIR / name
    record = wfdb.rdrecord(str(path))
    signal = record.p_signal[:, 0]  # ECG channel
    fs = record.fs
    return signal, fs

def load_labels(name: str):
    """
    Try to load apnea annotations for a record.
    Priority:
      1) name + 'er'  (e.g., a01er.apn)
      2) name + 'r'   (e.g., a01r.apn)
      3) name         (e.g., x07.apn)
    """
    candidates = [
        f"{name}er",
        f"{name}r",
        name
    ]

    ann = None
    used = None

    for c in candidates:
        try:
            ann = wfdb.rdann(str(DATASET_DIR / c), "apn")
            used = c
            break
        except FileNotFoundError:
            continue

    if ann is None:
        raise FileNotFoundError(f"No .apn annotation found for record {name}")

    print(f"Using annotation: {used}.apn for record {name}")

    labels = []
    for sym in ann.symbol:
        # In Apnea-ECG:
        # 'A' = apnea, 'N' = normal
        if sym.upper() == "A":
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)

def segment_signal(signal, fs, window_sec=60):
    win_size = int(window_sec * fs)
    n = len(signal) // win_size
    segments = []
    for i in range(n):
        seg = signal[i*win_size:(i+1)*win_size]
        segments.append(seg)
    return np.array(segments)
