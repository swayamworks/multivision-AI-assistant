import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================
# Paths
# =====================================

DATA_PATH = "processed/gender_features.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv(DATA_PATH)

# Remove non-feature columns
X = df.drop(columns=["gender", "actor", "file"])

# Encode labels
y = df["gender"].map({
    "male": 0,
    "female": 1
})

# =====================================
# Train/Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Feature Scaling
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# Model
# =====================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("\nTraining gender classifier...\n")

model.fit(X_train, y_train)

# =====================================
# Evaluation
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print(f"Accuracy: {accuracy:.4f}")
print("=" * 50)

print("\nClassification Report\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# =====================================
# Save
# =====================================

joblib.dump(model, os.path.join(MODEL_DIR, "gender_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "gender_scaler.pkl"))

print("\nSaved:")
print("models/gender_model.pkl")
print("models/gender_scaler.pkl")
