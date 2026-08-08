import os
import random
import cv2
from collections import Counter
from inference_utils import detect_face_box

# We will temporarily load the current ModelBundle which we'll rename to DiagnosticBaselineBundle later
from inference_utils import ModelBundle as DiagnosticBaselineBundle

def main():
    print("Loading models...")
    baseline = DiagnosticBaselineBundle('models/age_ethnicity_model.h5', '../emotion/weights/emotion_cnn_baseline.keras')
    
    data_dir = 'archive/UTKFace'
    if not os.path.exists(data_dir):
        print(f"Data dir {data_dir} not found.")
        return

    images = [f for f in os.listdir(data_dir) if f.endswith(".jpg")]
    random.seed(42)
    sample_images = random.sample(images, min(100, len(images)))

    baseline_predictions = []
    
    print("\n--- Running Baseline Evaluation (100 images) ---")
    for fname in sample_images:
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        box = detect_face_box(img)
        if box is None:
            continue
        try:
            res = baseline.predict(img, box)
            baseline_predictions.append(res['dominant_race'].lower())
        except:
            pass

    counts = Counter(baseline_predictions)
    total = sum(counts.values())
    white_pct = (counts.get('white', 0) / total) * 100 if total > 0 else 0
    print(f"Baseline Mode Collapse Evidence:")
    print(f"Total successful predictions: {total}")
    print(f"Distribution: {dict(counts)}")
    print(f"Predicted 'White' exactly {white_pct:.1f}% of the time.")

    print("\n--- Running Side-by-Side Comparison ---")
    comparison_images = sample_images[:3]
    for fname in comparison_images:
        path = os.path.join(data_dir, fname)
        img = cv2.imread(path)
        box = detect_face_box(img)
        if box is None:
            continue
        
        # Ground truth
        parts = fname.split('_')
        race_idx = parts[2]
        races = {"0": "white", "1": "black", "2": "asian", "3": "indian", "4": "others"}
        gt_race = races.get(race_idx, "unknown")

        # Baseline
        base_res = baseline.predict(img, box)
        base_race = base_res['dominant_race']

        print(f"Image: {fname}")
        print(f"  Ground Truth : {gt_race}")
        print(f"  Baseline     : {base_race}")
        print("-" * 40)

if __name__ == "__main__":
    main()
