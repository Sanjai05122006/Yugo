import pandas as pd
from preprocess import load_record, load_labels
from features import extract_features

RECORDS = [
    "a01","a02","a03","a04","a13","a14","b03","b04","c01","c02","c03","c07","c08",
    "x06","x07","x08","x09","x10","x11","x12","x13","x21","x22","x23","x24","x25",
    "x28","x29","x30","x31","x32","x33","x34","x35"
]

rows = []

for r in RECORDS:
    print(f"\nProcessing record: {r}")
    try:
        signal, fs = load_record(r)
        labels = load_labels(r)

        print(f"  Signal length: {len(signal)}, fs: {fs}")
        print(f"  Labels length: {len(labels)}")

        win_size = int(60 * fs)
        count = 0

        for i in range(len(labels)):
            start = i * win_size
            end = start + win_size

            if end > len(signal):
                print(f"  Break at i={i}, end={end} > signal_len={len(signal)}")
                break

            segment = signal[start:end]
            feats = extract_features(segment, fs)

            feats["label"] = labels[i]
            feats["record"] = r
            rows.append(feats)
            count += 1

        print(f"  Added {count} epochs for {r}")

    except Exception as e:
        print("  ❌ Error for", r, ":", repr(e))
        print("Skipping", r, e)

df = pd.DataFrame(rows)
df.to_csv("data/epochs_features.csv", index=False)
print("Saved data/epochs_features.csv")
print("Shape:", df.shape)
print("Label distribution:\n", df["label"].value_counts())


print("\nNumber of rows collected:", len(rows))
df = pd.DataFrame(rows)
print("DataFrame columns:", df.columns)

if len(rows) == 0:
    raise RuntimeError("No data collected! Check errors above.")

df.to_csv("data/epochs_features.csv", index=False)
print("Saved data/epochs_features.csv")
print("Label distribution:\n", df["label"].value_counts())
