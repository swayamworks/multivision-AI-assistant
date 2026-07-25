"""
collect_data.py
---------------
Webcam data collection tool for the landmark-based sign language MLP.

HOW TO USE:
    python collect_data.py

Controls:
    [SPACE]  — start/stop collecting samples for the current sign
    [N]      — move to the next sign class
    [Q]      — quit and save CSV

Output:
    modules/sign/landmark_data.csv
    Columns: label, x0,y0,z0, x1,y1,z1, ... x20,y20,z20  (63 features + 1 label)
"""

import os
import csv
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ── Config ────────────────────────────────────────────────────────────────
SIGNS = ["Hello", "I Love You", "No", "Peace", "Please", "Stop", "Thanks", "Wait", "Yes"]
SAMPLES_PER_SIGN = 200   # frames to collect per sign
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TASK_PATH = os.path.join(BASE_DIR, "gesture_recognizer.task")
OUT_CSV   = os.path.join(BASE_DIR, "landmark_data.csv")

# ── MediaPipe init ────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ── Landmark normalisation ────────────────────────────────────────────────
def normalise_landmarks(lms):
    """
    Normalise 21 landmarks relative to wrist (lm 0) and scale by
    wrist-to-middle-MCP distance (lm 0 → lm 9).
    Returns flat array of 63 floats.
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    origin = pts[0].copy()
    pts -= origin
    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts /= scale
    return pts.flatten()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    rows = []
    sign_idx = 0
    collecting = False
    collected  = 0

    print(f"\nCollecting data for: {SIGNS[sign_idx]}")
    print("SPACE = start/stop | N = next sign | Q = quit\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display   = frame_bgr.copy()
        h, w      = display.shape[:2]

        # Run MediaPipe
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        mp_result = recognizer.recognize(mp_image)

        hand_found = bool(mp_result.hand_landmarks)

        # Draw landmarks
        if hand_found:
            lms = mp_result.hand_landmarks[0]
            for lm in lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(display, (cx, cy), 5, (0, 255, 120), -1)

        # Collect sample
        if collecting and hand_found:
            feats = normalise_landmarks(mp_result.hand_landmarks[0])
            rows.append([SIGNS[sign_idx]] + feats.tolist())
            collected += 1
            if collected >= SAMPLES_PER_SIGN:
                collecting = False
                print(f"  ✓ {SAMPLES_PER_SIGN} samples collected for '{SIGNS[sign_idx]}'")
                if sign_idx < len(SIGNS) - 1:
                    print(f"  → Press [N] to move to next sign or [SPACE] to collect more.")
                else:
                    print("  → All signs done! Press [Q] to save and quit.")

        # HUD
        sign_name = SIGNS[sign_idx]
        status    = "● REC" if collecting else "○ READY"
        color     = (0, 0, 255) if collecting else (200, 200, 200)

        cv2.rectangle(display, (0, 0), (w, 90), (20, 20, 30), -1)
        cv2.putText(display, f"Sign: {sign_name}  [{sign_idx+1}/{len(SIGNS)}]",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        cv2.putText(display, f"{status}  {collected}/{SAMPLES_PER_SIGN} samples",
                    (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        if not hand_found:
            cv2.putText(display, "No hand detected", (15, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 200), 2)

        cv2.imshow("Sign Data Collector  [SPACE=rec | N=next | Q=quit]", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            collecting = not collecting
            if collecting:
                collected = 0
                print(f"  Recording '{SIGNS[sign_idx]}'...")
        elif key == ord('n') or key == ord('N'):
            collecting = False
            collected  = 0
            sign_idx   = (sign_idx + 1) % len(SIGNS)
            print(f"\nCollecting data for: {SIGNS[sign_idx]}")
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    recognizer.close()

    if rows:
        header = ["label"] + [f"{a}{i}" for i in range(21) for a in ("x","y","z")]
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\n✓ Saved {len(rows)} rows → {OUT_CSV}")
        counts = {}
        for r in rows:
            counts[r[0]] = counts.get(r[0], 0) + 1
        for sign, n in sorted(counts.items()):
            print(f"  {sign}: {n} samples")
    else:
        print("No data collected.")


if __name__ == "__main__":
    main()
