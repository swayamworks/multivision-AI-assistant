import os
import cv2
import time
import numpy as np
from PIL import Image
import pyttsx3
import threading

from modules.sign.predict import load_model, predict_sign, SIGN_EMOJIS
from modules.sign.assistant import SequenceAccumulator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.keras")

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
    Launch a dedicated 30 FPS real-time webcam video assistant window
    showing ONLY the cropped hand/arm cut-out image to verify segmentation.
    """
    print("=" * 60)
    print(" 🤟 MULTIVISION AI - CUT-OUT HAND REAL-TIME ASSISTANT")
    print("=" * 60)
    print("Loading sign recognition neural network...")
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

    window_name = "MultiVision AI - Hand Cut-Out Inspector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 900, 700)

    fps_count = 0
    start_time = time.time()
    fps_display = 0.0

    show_full_frame = False  # Toggle with [T]

    speak_text_async("Hand cut-out inspector active")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Run prediction & hand white-BG cut out
        res = predict_sign(model, frame_rgb)
        label = res["label"]
        conf = res["confidence"]
        hand_crop_rgb = res["hand_crop"]

        accumulator.update(label, conf)
        current_sentence = accumulator.get_text()

        # Measure FPS
        fps_count += 1
        if time.time() - start_time >= 1.0:
            fps_display = fps_count / (time.time() - start_time)
            fps_count = 0
            start_time = time.time()

        # Prepare display frame
        if show_full_frame:
            display_rgb = res["annotated_frame"]
        else:
            # Resize cut-out hand crop to canvas size (600x600)
            display_rgb = cv2.resize(hand_crop_rgb, (600, 600))

        display_bgr = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
        h, w, _ = display_bgr.shape

        # Draw HUD Banner at TOP
        top_overlay = display_bgr.copy()
        cv2.rectangle(top_overlay, (0, 0), (w, 75), (15, 18, 26), -1)
        cv2.addWeighted(top_overlay, 0.85, display_bgr, 0.15, 0, display_bgr)

        emoji = SIGN_EMOJIS.get(label, "🤟")
        sign_text = f"Hand Cut-Out | Sign: {label} ({conf:.1f}%)"
        cv2.putText(display_bgr, sign_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 222, 100) if conf > 50 else (60, 165, 250), 2)
        cv2.putText(display_bgr, f"FPS: {fps_display:.1f}", (w - 140, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Draw Sentence HUD Banner at BOTTOM
        bottom_overlay = display_bgr.copy()
        cv2.rectangle(bottom_overlay, (0, h - 85), (w, h), (15, 18, 26), -1)
        cv2.addWeighted(bottom_overlay, 0.85, display_bgr, 0.15, 0, display_bgr)

        sentence_display = current_sentence if current_sentence else "Sign gestures in camera..."
        cv2.putText(display_bgr, "TRANSLATED TEXT:", (15, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 190, 205), 1)
        cv2.putText(display_bgr, sentence_display, (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        # Key guidance
        cv2.putText(display_bgr, "[T]: Toggle View | [S]: Speak | [C]: Clear | [Q]: Quit", (w - 380, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 150, 165), 1)

        cv2.imshow(window_name, display_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('t') or key == ord('T'):
            show_full_frame = not show_full_frame
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
