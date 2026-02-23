import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import lightgbm as lgb

# Load dataset
df = pd.read_csv("data/epochs_features.csv")

# Split features / labels
X = df.drop(columns=["label", "record"])
y = df["label"]

# Train/val split (stratified because classes are imbalanced)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define model (stronger than baseline)
model = lgb.LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    class_weight="balanced",
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
probs = model.predict_proba(X_val)[:, 1]
preds = (probs > 0.5).astype(int)

auc = roc_auc_score(y_val, probs)
f1 = f1_score(y_val, preds)

print("Validation AUC:", auc)
print("Validation F1:", f1)
print("\nClassification report:\n", classification_report(y_val, preds))

# Save model
joblib.dump(model, "models/apnea_lgbm.pkl")
print("Model saved to models/apnea_lgbm.pkl")
