import wfdb
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

records = [
    "a01","a02","a03","a04","a13","a14","b03","b04","c01","c02","c03","c07","c08",
    "x06","x07","x08","x09","x10","x11","x12","x13","x21","x22","x23","x24","x25",
    "x28","x29","x30","x31","x32","x33","x34","x35"
]

def load_ann(name):
    for c in [f"{name}er", f"{name}r", name]:
        try:
            return wfdb.rdann(str(DATASET_DIR / c), "apn"), c
        except FileNotFoundError:
            continue
    return None, None

for r in records:
    ann, used = load_ann(r)
    if ann is None:
        print(r, "-> no annotation")
        continue

    syms = ann.symbol
    nA = sum(1 for s in syms if s == "A")
    nN = sum(1 for s in syms if s == "N")

    print(f"{r} using {used}.apn -> A: {nA}, N: {nN}")
