"""
gesture_utils.py (patched for internship-project)
---------------------------------------------------
Exact copy of ai-used-model/gesture_utils.py with MODEL_PATH updated to point
to the gesture_recognizer.task file in this directory.
"""

import datetime
import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# Operating hours configuration
# ---------------------------------------------------------------------------
OPERATING_START_HOUR = 18  # 6 PM
OPERATING_END_HOUR = 22    # 10 PM


def is_within_operating_hours(now: datetime.datetime = None) -> bool:
    """Return True if current time is within [OPERATING_START_HOUR, OPERATING_END_HOUR)."""
    now = now or datetime.datetime.now()
    return OPERATING_START_HOUR <= now.hour < OPERATING_END_HOUR


def operating_hours_message() -> str:
    return (
        f"Detection is only available between "
        f"{OPERATING_START_HOUR % 12 or 12} PM and "
        f"{OPERATING_END_HOUR % 12 or 12} PM."
    )


# ---------------------------------------------------------------------------
# Gesture -> "known word" vocabulary (customize freely)
# ---------------------------------------------------------------------------
GESTURE_TO_WORD = {
    "Thumb_Up": "YES",
    "Thumb_Down": "NO",
    "Open_Palm": "HELLO",
    "Closed_Fist": "STOP",
    "Victory": "PEACE",
    "ILoveYou": "I LOVE YOU",
    "Pointing_Up": "WAIT",
    "None": "",
}

# Point to the task file inside THIS directory (modules/sign/)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")


class GestureDetector:
    """Thin wrapper around MediaPipe's GestureRecognizer (IMAGE mode).

    IMAGE mode is used (instead of VIDEO/LIVE_STREAM) so the exact same
    code path can serve both the "upload image" and "live webcam" tabs
    without having to manage monotonic timestamps.
    """

    def __init__(self, model_path: str = MODEL_PATH, min_confidence: float = 0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at {model_path}.\n"
                "Run `python download_model.py` first to fetch the "
                "official MediaPipe gesture_recognizer.task model."
            )

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize(self, rgb_frame: np.ndarray):
        """Run detection on an RGB numpy image (H, W, 3).

        Returns a list of dicts: [{"gesture": name, "score": float,
        "word": str, "landmarks": [...]}] - one entry per detected hand.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize(mp_image)

        detections = []
        for hand_idx, gestures in enumerate(result.gestures):
            if not gestures:
                continue
            top = gestures[0]  # highest-confidence category for this hand
            landmarks = (
                result.hand_landmarks[hand_idx]
                if hand_idx < len(result.hand_landmarks)
                else []
            )
            detections.append(
                {
                    "gesture": top.category_name,
                    "score": top.score,
                    "word": GESTURE_TO_WORD.get(top.category_name, top.category_name),
                    "landmarks": landmarks,
                }
            )
        return detections

    def close(self):
        self._recognizer.close()
