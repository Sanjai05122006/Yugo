import wfdb
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

records = ["a01","a02","a03","a04","a13","a14","b03","b04","c01","c02","c03","c07","c08",
           "x06","x07","x08","x09","x10","x11","x12","x13","x21","x22","x23","x24","x25",
           "x28","x29","x30","x31","x32","x33","x34","x35"]

def load_ann(name):
    for c in [f"{name}er", f"{name}r", name]:
        try:
            return wfdb.rdann(str(DATASET_DIR / c), "apn"), c
        except FileNotFoundError:
            continue
    return None, None

all_syms = set()

for r in records:
    ann, used = load_ann(r)
    if ann is None:
        print("No annotation for", r)
        continue

    syms = set(ann.symbol)
    all_syms |= syms
    print(f"{r} using {used}.apn -> symbols: {syms}")

print("\nALL symbols seen:", all_syms)
