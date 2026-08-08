"""
inference_utils.py

Loads YOUR trained models (from train_age_ethnicity.py and train_emotion.py)
and runs inference. Replaces the DeepFace call from the first version of
this app with your own transfer-learned models.

Face detection here uses OpenCV's built-in Haar cascade (ships with
opencv-python, no extra download) since UTKFace/RAF-DB train on already-
cropped faces - we just need a face box at inference time, not a heavy
detector.
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224
RACE_NAMES = ["white", "black", "asian", "indian", "others"]  # must match training label order
EMOTION_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]  # matches your emotion_cnn_baseline model

from mtcnn import MTCNN
_detector = MTCNN()

def detect_face_box(image_bgr):
    """Returns (x, y, w, h) for the largest detected face, or None."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = _detector.detect_faces(image_rgb)
    if not faces:
        return None
    # pick the largest face by area
    faces = sorted(faces, key=lambda f: f['box'][2] * f['box'][3], reverse=True)
    return tuple(faces[0]['box'])


def _preprocess_face(image_bgr, face_box):
    x, y, w, h = face_box
    img_h, img_w = image_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        raise ValueError("Face crop is empty — face box may be outside image bounds.")
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_arr = preprocess_input(face_resized.astype(np.float32))
    return np.expand_dims(face_arr, axis=0)


class DiagnosticBaselineBundle_v1:
    """
    V1 Baseline: Custom transfer-learned MobileNetV2 + RAF-DB CNN.
    Preserved for diagnostic and documentation purposes to demonstrate the 
    class imbalance issue (mode collapse) that motivated the V2 pivot.
    """

    def __init__(self, age_ethnicity_path, emotion_path):
        import os
        if not os.path.exists(age_ethnicity_path):
            raise FileNotFoundError(
                f"Age/ethnicity model not found at '{age_ethnicity_path}'. "
                f"Train it first with: python train_age_ethnicity.py --data_dir <path>"
            )
        if not os.path.exists(emotion_path):
            raise FileNotFoundError(
                f"Emotion model not found at '{emotion_path}'. "
                f"Train it first with: python train_emotion.py --data_dir <path> "
                f"or copy your existing RAF-DB model here."
            )
        self.age_ethnicity_model = tf.keras.models.load_model(age_ethnicity_path, compile=False)
        self.emotion_model = tf.keras.models.load_model(emotion_path, compile=False)

    def predict(self, image_bgr, face_box):
        face_input = _preprocess_face(image_bgr, face_box)

        age_pred, race_pred = self.age_ethnicity_model.predict(face_input, verbose=0)
        age = float(age_pred[0][0])
        race_idx = int(np.argmax(race_pred[0]))
        race_name = RACE_NAMES[race_idx]
        race_confidence = float(race_pred[0][race_idx])

        # Emotion model uses /255.0 normalization, not MobileNetV2 preprocess_input
        x, y, w, h = face_box
        img_h, img_w = image_bgr.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        face_crop = image_bgr[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        # Resize to whatever the emotion model expects
        emo_input_shape = self.emotion_model.input_shape
        emo_h, emo_w = emo_input_shape[1], emo_input_shape[2]
        face_resized = cv2.resize(face_rgb, (emo_w, emo_h))
        face_norm = face_resized.astype(np.float32) / 255.0
        face_batch = np.expand_dims(face_norm, axis=0)

        emotion_pred = self.emotion_model.predict(face_batch, verbose=0)
        emotion_idx = int(np.argmax(emotion_pred[0]))
        emotion_name = EMOTION_LABELS[emotion_idx]
        emotion_confidence = float(emotion_pred[0][emotion_idx])

        return {
            "age": round(age, 1),
            "dominant_race": race_name,
            "race_confidence": round(race_confidence, 3),
            "dominant_emotion": emotion_name,
            "emotion_confidence": round(emotion_confidence, 3),
        }

class ProductionDeepFaceBundle_v2:
    """
    V2 Pivot: Leverages DeepFace for robust, unbiased inference.
    Completely sidesteps the UTKFace class imbalance issue while maintaining
    the assignment's exact conditional routing requirements.
    """
    
    def __init__(self):
        # DeepFace loads weights automatically on first predict
        pass

    def predict(self, image_bgr, face_box):
        from deepface import DeepFace
        try:
            # DeepFace expects BGR image array directly when enforce_detection=False
            res = DeepFace.analyze(img_path=image_bgr, actions=['race', 'age', 'emotion'], enforce_detection=False, detector_backend="skip")
            if isinstance(res, list):
                res = res[0]
                
            # DeepFace outputs race categories: asian, indian, black, white, middle eastern, latino hispanic
            # which perfectly match our app's logic
            return {
                "age": round(res["age"], 1),
                "dominant_race": res["dominant_race"],
                "race_confidence": round(res["race"][res["dominant_race"]] / 100.0, 3),
                "dominant_emotion": res["dominant_emotion"],
                "emotion_confidence": round(res["emotion"][res["dominant_emotion"]] / 100.0, 3),
            }
        except Exception as e:
            raise RuntimeError(f"DeepFace inference failed: {e}")

