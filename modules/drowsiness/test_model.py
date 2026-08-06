"""
Quick test script for the EfficientNet drowsiness model.
Run: python test_model.py --image "path/to/your/image.jpg"
Or:  python test_model.py   (uses sample dataset images)
"""

import os, sys, glob, argparse
import cv2, torch, torch.nn as nn, numpy as np
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image

# ── Setup ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Rebuild EfficientNet-B0 exactly as trained
weights_meta = EfficientNet_B0_Weights.DEFAULT
transform = weights_meta.transforms()

model = efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 1),
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print(f"Model loaded from: {MODEL_PATH}\n")


def predict(image_path):
    """Run prediction on a single image file."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"  [ERROR] Could not read image: {image_path}")
        return None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()

    # drowsy=0 (prob < 0.5), notdrowsy=1 (prob >= 0.5)
    label = "AWAKE" if prob >= 0.5 else "SLEEPING (DROWSY)"
    confidence = prob if prob >= 0.5 else 1 - prob

    print(f"  File      : {os.path.basename(image_path)}")
    print(f"  Raw prob  : {prob:.4f}  (0=drowsy, 1=notdrowsy)")
    print(f"  Prediction: {label}  ({confidence*100:.1f}% confidence)")
    print()
    return prob


# ── Main ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, default=None,
                    help="Path to an image file to test")
args = parser.parse_args()

if args.image:
    # Test on a single user-provided image
    predict(args.image)
else:
    # Auto-test on 5 drowsy + 5 notdrowsy images from dataset
    dataset_root = os.path.join(BASE_DIR, "nthu-dataset-ddd-multi-class", "Multi class", "train")
    
    drowsy_imgs    = glob.glob(os.path.join(dataset_root, "drowsy", "**", "*.jpg"), recursive=True)[:5]
    notdrowsy_imgs = glob.glob(os.path.join(dataset_root, "notdrowsy", "**", "*.jpg"), recursive=True)[:5]

    if not drowsy_imgs and not notdrowsy_imgs:
        print("No dataset images found. Run with --image flag:")
        print('  python test_model.py --image "C:\\path\\to\\photo.jpg"')
        sys.exit(0)

    print("=" * 50)
    print("DROWSY IMAGES (expected: SLEEPING)")
    print("=" * 50)
    drowsy_correct = 0
    for path in drowsy_imgs:
        prob = predict(path)
        if prob is not None and prob < 0.5:
            drowsy_correct += 1

    print("=" * 50)
    print("NOTDROWSY IMAGES (expected: AWAKE)")
    print("=" * 50)
    notdrowsy_correct = 0
    for path in notdrowsy_imgs:
        prob = predict(path)
        if prob is not None and prob >= 0.5:
            notdrowsy_correct += 1

    total = len(drowsy_imgs) + len(notdrowsy_imgs)
    correct = drowsy_correct + notdrowsy_correct
    print(f"Accuracy on sample: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Drowsy correct   : {drowsy_correct}/{len(drowsy_imgs)}")
    print(f"  Notdrowsy correct: {notdrowsy_correct}/{len(notdrowsy_imgs)}")
