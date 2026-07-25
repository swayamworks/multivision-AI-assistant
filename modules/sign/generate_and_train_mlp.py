"""
generate_and_train_mlp.py
-------------------------
Generates landmark training dataset for 9 signs and trains a custom MLP classifier model (sign_landmark_model.pkl).
"""

import os
import csv
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

SIGNS = ["Hello", "I Love You", "No", "Peace", "Please", "Stop", "Thanks", "Wait", "Yes"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "landmark_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sign_landmark_model.pkl")

def get_base_pose(sign):
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    pts[0] = [0.0, 0.0, 0.0]
    
    # Base MCPs
    pts[1] = [0.2, -0.2, 0.0]   # Thumb CMC
    pts[5] = [0.15, -0.5, 0.0]  # Index MCP
    pts[9] = [0.0, -0.55, 0.0]  # Middle MCP
    pts[13] = [-0.15, -0.5, 0.0] # Ring MCP
    pts[17] = [-0.3, -0.4, 0.0] # Pinky MCP

    def extend_finger(mcp_idx, dx, dy, dz):
        mcp = pts[mcp_idx]
        pts[mcp_idx+1] = mcp + [dx*0.35, dy*0.35, dz*0.35]
        pts[mcp_idx+2] = mcp + [dx*0.7, dy*0.7, dz*0.7]
        pts[mcp_idx+3] = mcp + [dx*1.0, dy*1.0, dz*1.0]

    def curl_finger(mcp_idx):
        mcp = pts[mcp_idx]
        pts[mcp_idx+1] = mcp + [0.0, 0.15, 0.1]
        pts[mcp_idx+2] = mcp + [0.0, 0.25, 0.15]
        pts[mcp_idx+3] = mcp + [0.0, 0.20, 0.05]

    if sign == "Hello":
        # All open
        extend_finger(1, 0.35, -0.35, 0.0)
        extend_finger(5, 0.08, -0.65, 0.0)
        extend_finger(9, 0.0, -0.7, 0.0)
        extend_finger(13, -0.08, -0.65, 0.0)
        extend_finger(17, -0.15, -0.55, 0.0)
        
    elif sign == "Stop":
        # Closed fist
        curl_finger(1)
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)

    elif sign == "Yes":
        # Thumb up, rest curled
        extend_finger(1, 0.1, -0.8, 0.0)
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)

    elif sign == "No":
        # Thumb down, rest curled
        extend_finger(1, 0.1, 0.8, 0.0)
        curl_finger(5)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)

    elif sign == "Peace":
        # Index & Middle extended, rest curled
        curl_finger(1)
        extend_finger(5, 0.15, -0.65, 0.0)
        extend_finger(9, -0.05, -0.65, 0.0)
        curl_finger(13)
        curl_finger(17)

    elif sign == "I Love You":
        # Thumb, Index, Pinky extended
        extend_finger(1, 0.45, -0.25, 0.0)
        extend_finger(5, 0.08, -0.65, 0.0)
        curl_finger(9)
        curl_finger(13)
        extend_finger(17, -0.2, -0.55, 0.0)

    elif sign == "Wait":
        # Only Index extended
        curl_finger(1)
        extend_finger(5, 0.0, -0.7, 0.0)
        curl_finger(9)
        curl_finger(13)
        curl_finger(17)

    elif sign == "Please":
        # Flat hand angled slightly
        extend_finger(1, 0.15, -0.3, 0.05)
        extend_finger(5, 0.05, -0.6, -0.05)
        extend_finger(9, 0.0, -0.65, -0.05)
        extend_finger(13, -0.05, -0.6, -0.05)
        extend_finger(17, -0.1, -0.5, -0.05)

    elif sign == "Thanks":
        # Hand curved forward
        extend_finger(1, 0.25, -0.2, -0.2)
        extend_finger(5, 0.05, -0.5, -0.3)
        extend_finger(9, 0.0, -0.55, -0.3)
        extend_finger(13, -0.05, -0.5, -0.3)
        extend_finger(17, -0.1, -0.4, -0.3)

    return pts

def normalise_landmarks(pts):
    origin = pts[0].copy()
    pts = pts - origin
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts = pts / scale
    return pts.flatten()

def generate_dataset(num_samples_per_class=350):
    rows = []
    np.random.seed(42)
    
    for sign in SIGNS:
        base_pts = get_base_pose(sign)
        for _ in range(num_samples_per_class):
            # Add noise, small 3D rotation, and scaling
            jitter = np.random.normal(0, 0.025, base_pts.shape).astype(np.float32)
            
            # Small random rotation matrix
            rx, ry, rz = np.random.uniform(-0.15, 0.15, 3)
            Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
            Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
            R = Rz @ Ry @ Rx

            noisy_pts = (base_pts + jitter) @ R.T
            norm_feats = normalise_landmarks(noisy_pts)
            rows.append([sign] + norm_feats.tolist())

    header = ["label"] + [f"{a}{i}" for i in range(21) for a in ("x","y","z")]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Generated {len(rows)} samples across {len(SIGNS)} signs -> {CSV_PATH}")

def train_mlp():
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c != 'label']
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df['label'].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=400,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        ))
    ])

    pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipeline.predict(X_test))
    print(f"MLP Training Complete! Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, pipeline.predict(X_test), target_names=le.classes_))

    save_obj = {'pipeline': pipeline, 'label_encoder': le}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(save_obj, f)
    print(f"Saved custom MLP model to: {MODEL_PATH}")

if __name__ == "__main__":
    generate_dataset()
    train_mlp()
