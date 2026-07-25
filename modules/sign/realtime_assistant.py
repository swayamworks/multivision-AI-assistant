import os
import cv2
import time
import numpy as np
from PIL import Image
import pyttsx3
import threading

from modules.sign.predict import load_model, predict_sign, SIGN_EMOJIS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.keras")

from modules.sign.assistant import SequenceAccumulator

# Thread-safe Text-to-Speech Engine
_TTS_ENGINE = None
_TTS_LOCK = threading.Lock()


def get_tts_engine():
    global _TTS_ENGINE
    if _TTS_ENGINE is None:
        try:
            _TTS_ENGINE = pyttsx3.init()
            _TTS_ENGINE.setProperty('rate', 150)
        except Exception:
            _TTS_ENGINE = False
    return _TTS_ENGINE


def speak_text_async(text):
    """Speak text in a non-blocking background thread."""
    def _speak():
        with _TTS_LOCK:
            engine = get_tts_engine()
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print(f"TTS Error: {e}")

    threading.Thread(target=_speak, daemon=True).start()


def run_realtime_assistant():
    """
    Launch dedicated 30 FPS real-time webcam video assistant window.
    Strict MediaPipe Hand Gate:
    - Shows 'No Hand Detected' when no hand is present.
    - Draws green joint dots, yellow skeleton, orange box, and sign text when hand is present.
    """
    print("=" * 60)
    print(" 🤟 MULTIVISION AI - SIGN LANGUAGE REAL-TIME VIDEO ASSISTANT")
    print("=" * 60)
    print("Loading MobileNetV2 sign language neural network...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("Starting webcam video capture...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open local webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    accumulator = SequenceAccumulator(debounce_frames=3, min_confidence=35.0)

    window_name = "MultiVision AI - Live Sign Recognition Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    speak_text_async("Sign language video assistant activated")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run sign prediction
        res = predict_sign(model, frame_rgb)
        label = res["label"]
        conf = res["confidence"]
        annotated_bgr = cv2.cvtColor(res["annotated_frame"], cv2.COLOR_RGB2BGR)

        changed = accumulator.update(label, conf)
        current_sentence = accumulator.get_text()

        if changed and label not in ["nothing", "No Hand Detected"]:
            print(f"▶ Detected Sign: {label} ({conf:.1f}%) | Sentence: {current_sentence}")

        cv2.imshow(window_name, annotated_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            accumulator.add_space()
        elif key == 8 or key == 127:
            accumulator.backspace()
        elif key == ord('c') or key == ord('C'):
            accumulator.clear()
        elif key == ord('s') or key == ord('S'):
            if current_sentence:
                speak_text_async(current_sentence)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_assistant()
