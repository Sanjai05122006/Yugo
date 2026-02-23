import wfdb
from pathlib import Path

# Path to project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

record_path = DATASET_DIR / "a01"

print("Reading from:", record_path)

record = wfdb.rdrecord(str(record_path))

print("Signal shape:", record.p_signal.shape)
print("Sampling rate:", record.fs)
print("Channels:", record.sig_name)
