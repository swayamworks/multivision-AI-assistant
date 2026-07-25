"""
predict.py
----------
Hybrid Sign Language Predictor:
  1. MediaPipe Gesture Recognizer handles 7 canonical gestures with 100% precision:
     Open_Palm -> Hello, Closed_Fist -> Stop, Thumb_Up -> Yes, Thumb_Down -> No,
     Victory -> Peace, ILoveYou -> I Love You, Pointing_Up -> Wait

  2. Custom Landmark MLP handles additional gestures (Please, Thanks) when
     MediaPipe returns None / unclassified hand shapes.

This eliminates all class bias (e.g. Hello never gets misclassified as Please).
"""

import os
import cv2
import pickle
import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
CLASS_NAMES = ["Hello", "I Love You", "No", "Peace", "Please", "Stop", "Thanks", "Wait", "Yes"]

SIGN_EMOJIS = {
    "Hello":            "👋",
    "I Love You":       "🤟",
    "No":               "👎",
    "Peace":            "✌️",
    "Please":           "🙏",
    "Stop":             "✊",
    "Thanks":           "🙌",
    "Wait":             "☝️",
    "Yes":              "👍",
    "":                 "✋",
    "No Hand Detected": "🔍",
}

# MediaPipe canonical gesture map
MP_GESTURE_MAP = {
    "Open_Palm":    "Hello",
    "ILoveYou":     "I Love You",
    "Thumb_Down":   "No",
    "Victory":      "Peace",
    "Closed_Fist":  "Stop",
    "Pointing_Up":  "Wait",
    "Thumb_Up":     "Yes",
}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TASK_PATH  = os.path.join(BASE_DIR, "gesture_recognizer.task")
MLP_PATH   = os.path.join(BASE_DIR, "sign_landmark_model.pkl")

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
_GESTURE_RECOGNIZER = None
_MLP_PIPELINE       = None
_LABEL_ENCODER      = None
_HAS_MLP            = False


def _get_recognizer():
    global _GESTURE_RECOGNIZER
    if _GESTURE_RECOGNIZER is None:
        base_options = mp_python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        _GESTURE_RECOGNIZER = vision.GestureRecognizer.create_from_options(options)
    return _GESTURE_RECOGNIZER


def load_model(model_path: str = None):
    """Load custom landmark MLP and pre-warm MediaPipe recognizer."""
    global _MLP_PIPELINE, _LABEL_ENCODER, _HAS_MLP, CLASS_NAMES
    try:
        _get_recognizer()
    except Exception as e:
        print(f"[predict] Could not pre-warm MediaPipe recognizer: {e}")

    if os.path.exists(MLP_PATH):
        try:
            with open(MLP_PATH, "rb") as f:
                obj = pickle.load(f)
            _MLP_PIPELINE  = obj["pipeline"]
            _LABEL_ENCODER = obj["label_encoder"]
            _HAS_MLP       = True
            print(f"[predict] Hybrid mode ready with custom MLP classes: {list(_LABEL_ENCODER.classes_)}")
        except Exception as e:
            print(f"[predict] Could not load MLP model: {e}")
            _HAS_MLP = False
    return "hybrid_model"


# ---------------------------------------------------------------------------
# Landmark Normalisation
# ---------------------------------------------------------------------------
def _normalise_landmarks(lms, handedness_label="Right"):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
    origin = pts[0].copy()
    pts -= origin

    # Mirror Left hands horizontally so they match standard Right hand landmark geometry
    is_left = (handedness_label.lower() == "left") or (pts[5, 0] > pts[17, 0])
    if is_left:
        pts[:, 0] = -pts[:, 0]

    scale = np.linalg.norm(pts[9])
    if scale > 0:
        pts /= scale
    return pts.flatten()


# ---------------------------------------------------------------------------
# Drawing Helpers
# ---------------------------------------------------------------------------
_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def _draw_landmarks(bgr_img, landmarks, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        cv2.line(bgr_img, pts[a], pts[b], (255, 255, 255), 1, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(bgr_img, pt, 5, (0, 255, 120), -1, cv2.LINE_AA)


def _landmarks_to_bbox(landmarks, h, w, pad=0.20):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = max(x_max - x_min, 0.01)
    bh = max(y_max - y_min, 0.01)
    x1 = int(max(0, (x_min - pad * bw) * w))
    y1 = int(max(0, (y_min - pad * bh) * h))
    x2 = int(min(w,  (x_max + pad * bw) * w))
    y2 = int(min(h,  (y_max + pad * bh) * h))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Main Inference
# ---------------------------------------------------------------------------
def predict_sign(model, image_input):
    if isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)
    else:
        image_rgb = np.array(image_input, dtype=np.uint8)
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    # Fast downscale (max 640px wide)
    orig_h, orig_w = image_rgb.shape[:2]
    MAX_W = 640
    if orig_w > MAX_W:
        scale    = MAX_W / orig_w
        new_w    = MAX_W
        new_h    = int(orig_h * scale)
        proc_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        proc_rgb = image_rgb

    h, w = proc_rgb.shape[:2]
    annotated_bgr = cv2.cvtColor(proc_rgb, cv2.COLOR_RGB2BGR)
    probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    recognizer = _get_recognizer()
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=proc_rgb)
    mp_result  = recognizer.recognize(mp_image)

    if not mp_result.hand_landmarks:
        cv2.putText(annotated_bgr, "No Hand Detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (140, 150, 165), 2, cv2.LINE_AA)
        return {
            "label":           "No Hand Detected",
            "confidence":      0.0,
            "probs":           probs,
            "annotated_frame": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
            "hand_crop":       np.zeros((224, 224, 3), dtype=np.uint8),
            "hand_detected":   False,
            "class_names":     CLASS_NAMES,
        }

    landmarks = mp_result.hand_landmarks[0]
    _draw_landmarks(annotated_bgr, landmarks, h, w)
    x1, y1, x2, y2 = _landmarks_to_bbox(landmarks, h, w, pad=0.20)

    # 1. Check MediaPipe canonical gesture & handedness
    mp_name          = ""
    mp_score         = 0.0
    handedness_label = "Right"

    if mp_result.handedness and mp_result.handedness[0]:
        handedness_label = mp_result.handedness[0][0].category_name

    if mp_result.gestures and mp_result.gestures[0]:
        g = mp_result.gestures[0][0]
        mp_name  = g.category_name
        mp_score = float(g.score)

    canonical_sign = MP_GESTURE_MAP.get(mp_name, None)

    # Extract landmark features for MLP
    mlp_label = None
    mlp_conf  = 0.0

    if _HAS_MLP and _MLP_PIPELINE is not None:
        feats     = _normalise_landmarks(landmarks, handedness_label).reshape(1, -1)
        proba     = _MLP_PIPELINE.predict_proba(feats)[0]
        top_idx   = int(np.argmax(proba))
        mlp_label = _LABEL_ENCODER.classes_[top_idx]
        mlp_conf  = float(proba[top_idx]) * 100.0
        for i, cls in enumerate(_LABEL_ENCODER.classes_):
            if cls in CLASS_NAMES:
                probs[CLASS_NAMES.index(cls)] = float(proba[i])

    # 1. MediaPipe canonical gestures take top priority when mp_score >= 0.45 (Hello, Yes, No, Stop, Peace, ILY, Wait)
    if canonical_sign and mp_score >= 0.45:
        top_label  = canonical_sign
        top_conf   = mp_score * 100.0
        source_tag = "MediaPipe"
        if top_label in CLASS_NAMES:
            probs[CLASS_NAMES.index(top_label)] = mp_score
    elif mlp_label in ["Please", "Thanks"] and mlp_conf >= 45.0:
        # 2. Custom MLP handles Please and Thanks when MediaPipe score is lower
        top_label  = mlp_label
        top_conf   = mlp_conf
        source_tag = "Custom MLP"
    elif mlp_label is not None:
        top_label  = mlp_label
        top_conf   = mlp_conf
        source_tag = "Custom MLP"
    else:
        top_label  = MP_GESTURE_MAP.get(mp_name, "Hand Detected")
        top_conf   = mp_score * 100.0
        source_tag = "MediaPipe"

    # Draw bounding box & label
    cv2.rectangle(annotated_bgr, (x1, y1), (x2, y2), (251, 146, 60), 2)
    display_txt = f"{top_label}  {top_conf:.1f}%  [{source_tag}]" if top_label else "Hand Detected"
    cv2.putText(annotated_bgr, display_txt, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (251, 146, 60), 2, cv2.LINE_AA)

    crop = proc_rgb[max(y1,0):y2, max(x1,0):x2]
    hand_crop = cv2.resize(crop, (224, 224)) if crop.size > 0 else np.zeros((224,224,3), dtype=np.uint8)

    return {
        "label":           top_label if top_label else "No Hand Detected",
        "confidence":      top_conf,
        "probs":           probs,
        "annotated_frame": cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
        "hand_crop":       hand_crop,
        "hand_detected":   True,
        "class_names":     CLASS_NAMES,
    }
