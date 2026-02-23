import pandas as pd
from preprocess import load_record, load_labels, segment_signal
from features import extract_features

RECORDS = [
    "a01","a02","a03","a04","a13","a14","b03","b04","c01","c02","c03","c07","c08",
    "x06","x07","x08","x09","x10","x11","x12","x13","x21","x22","x23","x24","x25",
    "x28","x29","x30","x31","x32","x33","x34","x35"
]

rows = []

for r in RECORDS:
    try:
        signal, fs = load_record(r)
        labels = load_labels(r)
        segments = segment_signal(signal, fs, 60)

        n = min(len(segments), len(labels))

        for i in range(n):
            feats = extract_features(segments[i], fs)
            feats["label"] = labels[i]
            feats["record"] = r
            rows.append(feats)

        print(f"Processed {r} with {n} epochs")

    except Exception as e:
        print("Skipping", r, e)

df = pd.DataFrame(rows)
df.to_csv("data/epochs_features.csv", index=False)
print("Saved data/epochs_features.csv")
print("Shape:", df.shape)
