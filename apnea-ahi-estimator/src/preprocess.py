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
    ann = wfdb.rdann(str(DATASET_DIR / name), "apn")
    labels = []
    for s in ann.aux_note:
        if s is not None and "A" in s:
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
