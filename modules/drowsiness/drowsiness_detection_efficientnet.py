"""
Driver Drowsiness Detection - EfficientNet-B0 Training Script
Fully Refactored & Stabilized for Windows (RTX 4050 / 6GB VRAM)
Features: Automatic Checkpoint Resume / Gradient Cleanups / Dynamic Epoch Budget
"""

import os
import glob
import copy
import time
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as T
import multiprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ---------------------------------------------------------------------------
# Custom Dataset Definition
# ---------------------------------------------------------------------------
class DrowsinessDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe.loc[index, "image_path"]
        label = self.dataframe.loc[index, "label"]
        
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image, label_tensor

# ---------------------------------------------------------------------------
# Main Execution Routine
# ---------------------------------------------------------------------------
def main():
    # 1. Environment & Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")
    if device.type == "cuda":
        print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # 2. Dataset Paths & Dataframe Builder
    DATASET_DIR = r"C:\Users\swaya\Downloads\Browser\archive\Driver Drowsiness Dataset (DDD)"

    image_paths = []
    labels = []
    class_map = {"Drowsy": 0, "Non Drowsy": 1}

    for folder_name, class_id in class_map.items():
        folder_path = os.path.join(DATASET_DIR, folder_name)
        if os.path.exists(folder_path):
            files = (
                glob.glob(os.path.join(folder_path, "**", "*.png"), recursive=True) +
                glob.glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True) +
                glob.glob(os.path.join(folder_path, "**", "*.jpeg"), recursive=True)
            )
            for f in files:
                image_paths.append(f)
                labels.append(class_id)
            print(f"[Data] Found {len(files)} images in '{folder_name}' (Class ID: {class_id})")
        else:
            print(f"[Warning] Folder not found: {folder_path}")

    df_images = pd.DataFrame({"image_path": image_paths, "label": labels})
    print(f"[Data] Total dataset size: {len(df_images)} images\n")

    if len(df_images) == 0:
        raise ValueError(f"No images found in dataset directory: {DATASET_DIR}")

    # 3. Train / Val / Test Split
    df_train, df_temp = train_test_split(
        df_images, test_size=0.30, stratify=df_images["label"], random_state=42
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.50, stratify=df_temp["label"], random_state=42
    )

    print(f"[Split] Train samples : {len(df_train)}")
    print(f"[Split] Val samples   : {len(df_val)}")
    print(f"[Split] Test samples  : {len(df_test)}\n")

    # 4. Configurations & Transforms
    BATCH_SIZE = 32         # Safe for 6GB VRAM on Windows
    NUM_WORKERS = 0        # Avoids multiprocessing issues on Windows
    ADDITIONAL_EPOCHS = 10 # Number of training epochs to execute from current state
    PATIENCE = 3

    weights_config = EfficientNet_B0_Weights.DEFAULT
    eval_transform = weights_config.transforms()

    train_transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = DrowsinessDataset(df_train, train_transform)
    val_dataset = DrowsinessDataset(df_val, eval_transform)
    test_dataset = DrowsinessDataset(df_test, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda")
    )

    # 5. Base Model, Optimizer, Scheduler & Scaler Setup
    model = efficientnet_b0(weights=weights_config)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 1)
    )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 6. Checkpoint Paths & Automatic Resume Logic
    best_model_path = r"C:\Users\swaya\Desktop\Coding\Python ML\internship-project\modules\drowsiness\best_model.pt"
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    final_weights_path = os.path.join(script_dir, "final_weights.pt")

    start_epoch = 0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    print("=" * 60)
    print(" Checkpoint Inspection & Model Initialization")
    print("=" * 60)

    if os.path.exists(best_model_path):
        print("[INFO] Found previous checkpoint.")
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Determine if checkpoint is a dictionary containing metadata or raw state_dict
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Handle size mismatches (e.g., modified classifier layers)
        current_model_dict = model.state_dict()
        filtered_state_dict = {}
        mismatched_keys = []

        for k, v in state_dict.items():
            if k in current_model_dict:
                if current_model_dict[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    mismatched_keys.append(k)

        current_model_dict.update(filtered_state_dict)
        model.load_state_dict(current_model_dict)
        print("[INFO] Loaded model weights successfully.")

        if mismatched_keys:
            print(f"[WARNING] Classifier/layer shape mismatch detected for: {mismatched_keys}. Kept initialized weights for these layers.")

        # Restore optimizer state if compatible
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint and not mismatched_keys:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                print("[INFO] Restored optimizer state.")
            except Exception as e:
                print(f"[WARNING] Could not restore optimizer state ({e}). Using new optimizer.")

        # Restore starting epoch
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            completed_epoch = checkpoint["epoch"]
            start_epoch = completed_epoch + 1
            print(f"[INFO] Resuming training from epoch {start_epoch + 1}.")
        else:
            print("[INFO] No epoch metadata found in checkpoint. Starting from epoch 1 with loaded weights.")

    else:
        print("[INFO] No checkpoint found.")
        print("[INFO] Starting training from ImageNet pretrained weights.")

    best_weights = copy.deepcopy(model.state_dict())
    end_epoch = start_epoch + ADDITIONAL_EPOCHS

    print("=" * 60)
    print(f" Starting EfficientNet-B0 Training (Epochs {start_epoch + 1} to {end_epoch})")
    print("=" * 60)

    # 7. Training & Validation Loop
    for epoch in range(start_epoch, end_epoch):
        start = time.time()
        
        # --- Training Step ---
        model.train()
        running_loss, correct_preds, total_samples = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Train]")

        for images, targets in train_bar:
            optimizer.zero_grad(set_to_none=True)
            
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_preds += (preds == targets).sum().item()
            total_samples += targets.size(0)

            train_bar.set_postfix(
                loss=f"{running_loss / total_samples:.4f}",
                acc=f"{correct_preds / total_samples:.4f}"
            )

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = correct_preds / total_samples

        # --- Validation Step ---
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{end_epoch} [Val]")

        with torch.no_grad():
            for images, targets in val_bar:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True).unsqueeze(1)

                with torch.amp.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

                val_bar.set_postfix(
                    loss=f"{val_loss_sum / val_total:.4f}",
                    acc=f"{val_correct / val_total:.4f}"
                )

        epoch_val_loss = val_loss_sum / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_loss)

        print(
            f"Epoch {epoch+1:02d}/{end_epoch:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | "
            f"Time: {time.time()-start:.1f}s"
        )

        # Early Stopping & Best Checkpoint Saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save comprehensive checkpoint to target path
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, best_model_path)
            print(f"  --> Saved new best model to: {best_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\n[Info] Early stopping triggered after epoch {epoch+1}.")
                break

    # 8. Save Final Model State Dictionary
    model.load_state_dict(best_weights)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": end_epoch - 1
    }, final_weights_path)
    print(f"\n[Success] Training complete! Models saved to:\n  - {best_model_path}\n  - {final_weights_path}")

    # 9. Test Set Evaluation
    model.eval()
    test_loss_sum, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast(device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, targets)

            test_loss_sum += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            test_correct += (preds == targets).sum().item()
            test_total += targets.size(0)

    print(f"\n[Final Evaluation] Test Loss: {test_loss_sum / test_total:.4f} | Test Acc: {test_correct / test_total:.4f}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()