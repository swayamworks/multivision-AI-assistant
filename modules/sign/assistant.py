import io
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
from modules.sign.predict import predict_sign, SIGN_EMOJIS


class SequenceAccumulator:
    """Manages continuous sign detection debouncing, text building, and special token handling."""

    def __init__(self, debounce_frames=3, min_confidence=35.0):
        self.text = ""
        self.debounce_frames = debounce_frames
        self.min_confidence = min_confidence
        self.recent_predictions = []
        self.last_added_sign = None

    def update(self, sign_label, confidence):
        """Update buffer with frame prediction. Returns True if text changed."""
        if confidence < self.min_confidence:
            return False

        self.recent_predictions.append(sign_label)
        if len(self.recent_predictions) > self.debounce_frames:
            self.recent_predictions.pop(0)

        # Check if all recent predictions match
        if len(self.recent_predictions) == self.debounce_frames and len(set(self.recent_predictions)) == 1:
            stable_sign = self.recent_predictions[0]

            # Avoid repeated insertion of the exact same letter consecutively unless reset
            if stable_sign != self.last_added_sign:
                self.last_added_sign = stable_sign
                return self.apply_sign(stable_sign)

        return False

    def apply_sign(self, sign_label):
        """Apply a validated sign to the text buffer."""
        changed = False
        if sign_label == "del":
            if len(self.text) > 0:
                self.text = self.text[:-1]
                changed = True
        elif sign_label == "space":
            if len(self.text) > 0 and not self.text.endswith(" "):
                self.text += " "
                changed = True
        elif sign_label == "nothing":
            # Idle gesture, reset last_added_sign to allow repeating same letter later
            self.last_added_sign = None
        else:
            # Letters A-Z
            self.text += sign_label
            changed = True

        return changed

    def add_character(self, char):
        self.text += char

    def add_space(self):
        if len(self.text) > 0 and not self.text.endswith(" "):
            self.text += " "

    def backspace(self):
        if len(self.text) > 0:
            self.text = self.text[:-1]

    def clear(self):
        self.text = ""
        self.recent_predictions = []
        self.last_added_sign = None

    def get_text(self):
        return self.text


def process_video_file(model, video_path, max_frames=120, sample_rate=3):
    """
    Process video file frame by frame, crop hands, predict signs, and accumulate transcript.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    accumulator = SequenceAccumulator(debounce_frames=2, min_confidence=30.0)
    annotated_frames = []
    sign_timeline = []
    frame_count = 0
    processed_count = 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or processed_count >= max_frames:
            break

        frame_count += 1
        if frame_count % sample_rate != 0:
            continue

        processed_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = predict_sign(model, frame_rgb)

        label = result["label"]
        conf = result["confidence"]
        timestamp = frame_count / fps

        changed = accumulator.update(label, conf)

        sign_timeline.append({
            "timestamp": timestamp,
            "label": label,
            "confidence": conf,
            "emoji": SIGN_EMOJIS.get(label, ""),
            "changed": changed
        })

        if len(annotated_frames) < 16:  # Keep key gallery frames
            annotated_frames.append(result["annotated_frame"])

    cap.release()

    return {
        "transcript": accumulator.get_text(),
        "timeline": sign_timeline,
        "key_frames": annotated_frames,
        "total_processed_frames": processed_count,
    }


def generate_tts_audio(text, lang="en"):
    """Generate MP3 audio bytes using gTTS."""
    clean_text = text.strip()
    if not clean_text:
        return None

    try:
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None
