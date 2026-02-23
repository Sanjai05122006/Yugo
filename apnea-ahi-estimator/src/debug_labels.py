import wfdb
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

ann = wfdb.rdann(str(DATASET_DIR / "a01er"), "apn")

print("Unique symbols:", set(ann.symbol))
print("First 50 symbols:", ann.symbol[:50])
