"""CUDA-enabled UTKFace age-regression trainer.

Usage:
  python modules/drowsiness/train_age_estimator_torch.py \
      --data-dir "C:/path/to/UTKFace"
"""

import argparse
import os
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


IMAGE_SIZE = 224


class UTKFaceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, age = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(age, dtype=torch.float32)


def load_samples(data_dir):
    samples = []
    pattern = re.compile(r"^(\d+)_")
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        match = pattern.match(path.name)
        if match:
            age = int(match.group(1))
            if 0 <= age <= 116:
                samples.append((str(path), age))
    if not samples:
        raise ValueError("No UTKFace files found. Expected filenames beginning with '<age>_'.")
    return samples


def build_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model, weights


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_absolute_error = total_items = 0
    amp_enabled = device.type == "cuda"

    for images, ages in loader:
        images, ages = images.to(device, non_blocking=True), ages.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            predicted_ages = model(images).squeeze(1)
            loss = loss_fn(predicted_ages, ages)
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        batch_size = ages.size(0)
        total_loss += loss.item() * batch_size
        total_absolute_error += (predicted_ages.detach() - ages).abs().sum().item()
        total_items += batch_size
    return total_loss / total_items, total_absolute_error / total_items


def main():
    parser = argparse.ArgumentParser(description="Train a GPU age estimator from UTKFace.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This trainer requires the NVIDIA GPU-enabled PyTorch build.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    samples = load_samples(args.data_dir)
    random.Random(42).shuffle(samples)
    split = int(len(samples) * 0.85)
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"Dataset: {len(samples)} images | train: {len(train_samples)} | validation: {len(val_samples)}")

    _, weights = build_model()
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
    ])
    eval_transform = weights.transforms()
    train_loader = DataLoader(UTKFaceDataset(train_samples, train_transform), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(UTKFaceDataset(val_samples, eval_transform), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)

    model, _ = build_model()
    model.to(device)
    loss_fn = nn.SmoothL1Loss(beta=5.0)
    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "age_estimator_best.pt"
    final_path = output_dir / "age_estimator_final.pt"
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler)
        with torch.no_grad():
            val_loss, val_mae = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}/{args.epochs} | train MAE: {train_mae:.2f} | val MAE: {val_mae:.2f}", flush=True)
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({"model": model.state_dict(), "val_mae": val_mae, "architecture": "mobilenet_v3_small"}, best_path)
            print(f"Saved best checkpoint: {best_path}", flush=True)

    torch.save({"model": model.state_dict(), "best_val_mae": best_mae, "architecture": "mobilenet_v3_small"}, final_path)
    print(f"Training complete. Best validation MAE: {best_mae:.2f} years", flush=True)
    print(f"Saved final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
