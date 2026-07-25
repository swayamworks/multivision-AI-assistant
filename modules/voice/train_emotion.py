import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================
# PATHS
# =====================================

DATA_PATH = "processed/emotion_features.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["emotion", "actor", "file"])
y = df["emotion"]

# Encode emotion labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# FEATURE SCALING
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# MODEL
# =====================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(encoder.classes_),

    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,

    eval_metric="mlogloss",
    tree_method="hist"
)

print("\nTraining emotion classifier...\n")

model.fit(X_train, y_train)

# =====================================
# EVALUATION
# =====================================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print(f"Accuracy: {accuracy:.4f}")
print("=" * 50)

print("\nClassification Report\n")
print(classification_report(
    y_test,
    predictions,
    target_names=encoder.classes_
))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, predictions))

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(model, os.path.join(MODEL_DIR, "emotion_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "emotion_scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "emotion_encoder.pkl"))

print("\nSaved:")
print("models/emotion_model.pkl")
print("models/emotion_scaler.pkl")
print("models/emotion_encoder.pkl")