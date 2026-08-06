"""
Training Script for Custom Driver Drowsiness Detection YOLOv8 Model

This script prepares data configuration and trains a custom YOLOv8 model
to detect two classes:
  Class 0: awake
  Class 1: asleep (sleeping)
"""

import os
from ultralytics import YOLO

def create_dataset_config(dataset_dir="dataset"):
    """
    Creates the dataset.yaml file required for YOLOv8 training.
    Assumes standard YOLO dataset layout:
      dataset/
        train/images, train/labels
        val/images, val/labels
    """
    yaml_content = f"""
path: {os.path.abspath(dataset_dir)}
train: train/images
val: val/images

names:
  0: awake
  1: asleep
"""
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    os.makedirs(dataset_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"Created dataset config at: {yaml_path}")
    return yaml_path


def train_model(dataset_yaml_path, epochs=25, imgsz=640, batch=16):
    """
    Trains YOLOv8 nano model using transfer learning.
    """
    print("Loading pretrained YOLOv8n model...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training for {epochs} epochs...")
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        name="drowsiness_yolov8",
        exist_ok=True
    )
    
    best_weights = os.path.join("runs", "detect", "drowsiness_yolov8", "weights", "best.pt")
    print(f"Training Complete! Best model saved at: {best_weights}")
    return best_weights


if __name__ == "__main__":
    # Example usage:
    # 1. Place your dataset in 'dataset/' directory
    # 2. Run this script: python train_drowsiness.py
    yaml_file = create_dataset_config("dataset")
    print("\nTo start training once your dataset is placed in 'dataset/':")
    print("Execute: python train_drowsiness.py")
