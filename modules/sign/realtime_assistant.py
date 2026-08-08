import os
import cv2
import time
import numpy as np
from PIL import Image
try:
    import pyttsx3
except Exception:
    pyttsx3 = None
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

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(0, backend)
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

        # High-quality aspect-ratio-preserving upscale if camera natively captured lower resolution
        if frame_bgr.shape[1] < 1280:
            h, w = frame_bgr.shape[:2]
            scale = 1280.0 / w
            new_h = int(h * scale)
            frame_bgr = cv2.resize(frame_bgr, (1280, new_h), interpolation=cv2.INTER_CUBIC)
            
        cv2.resizeWindow(window_name, frame_bgr.shape[1], frame_bgr.shape[0])

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

        # --- Draw HUD Banner ---
        h, w = annotated_bgr.shape[:2]
        banner_h = 140
        
        # Draw solid dark banner background at bottom
        cv2.rectangle(annotated_bgr, (0, h - banner_h), (w, h), (20, 20, 20), -1)
        
        # Top banner separator line
        cv2.line(annotated_bgr, (0, h - banner_h), (w, h - banner_h), (200, 200, 200), 2)
        
        # Display Current Sentence
        display_text = current_sentence if current_sentence else "Waiting for signs..."
        cv2.putText(annotated_bgr, f"Sentence: {display_text}", (30, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Display Shortcuts
        shortcuts = "Shortcuts: [SPACE] Add Space | [BACKSPACE] Delete Word | [C] Clear | [S] Speak | [Q] Quit"
        cv2.putText(annotated_bgr, shortcuts, (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)

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
