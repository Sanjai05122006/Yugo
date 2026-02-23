import pandas as pd
df = pd.read_csv("data/epochs_features.csv")
print(df["label"].value_counts())
