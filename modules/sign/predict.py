import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# 29 ASL Alphabet Dataset Classes (alphabetically sorted as loaded by image_dataset_from_directory)
CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
]

# Display emoji mapping for user feedback
SIGN_EMOJIS = {
    "A": "🅰️", "B": "🅱️", "C": "©️", "D": "🇩", "E": "🇪",
    "F": "🎏", "G": "🇬", "H": "🇭", "I": "ℹ️", "J": "🇯",
    "K": "🇰", "L": "🇱", "M": "Ⓜ️", "N": "🇳", "O": "⭕",
    "P": "🅿️", "Q": "🇶", "R": "🇷", "S": "🇸", "T": "🇹",
    "U": "🇺", "V": "✌️", "W": "🇼", "X": "❌", "Y": "🇾",
    "Z": "⚡", "del": "⌫", "nothing": "✋", "space": "␣"
}

_LAST_HAND_BBOX = None


def load_model(model_path):
    """Load the Keras Sign Language recognition model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Sign language model file not found at: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


def isolate_hand_only_erase_body_face(image_rgb, prev_frame_rgb=None):
    """
    Locates hand/arm region and crops tightly around it using original natural camera pixels.
    REMOVED all white pixel masking / turning white.
    Returns (natural_hand_arm_crop, annotated_frame, hand_detected, bbox).
    """
    global _LAST_HAND_BBOX
    h, w, _ = image_rgb.shape
    annotated = image_rgb.copy()
    hand_detected = False
    bbox = None
    crop = None

    # Skin color segmentation in YCrCb and HSV color spaces for location tracking only
    ycrcb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YCrCb)
    mask_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8)
    )

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask_hsv = cv2.inRange(
        hsv,
        np.array([0, 15, 60], dtype=np.uint8),
        np.array([25, 255, 255], dtype=np.uint8)
    )

    skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    motion_mask = None
    if prev_frame_rgb is not None:
        diff = cv2.absdiff(image_rgb, prev_frame_rgb)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, motion_mask = cv2.threshold(diff_gray, 18, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (h * w) * 0.005

    candidate_hand_boxes = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = float(bw) / (bh + 1e-5)
        cy = by + bh / 2.0
        cx = bx + bw / 2.0

        hull = cv2.convexHull(c)
        solidity = float(area) / (cv2.contourArea(hull) + 1e-5)

        # STRICT FACE EXCLUSION: Upper 45% of image, central X position, round shape
        is_face = (by < h * 0.45) and (0.25 * w < cx < 0.75 * w) and (0.65 <= aspect <= 1.35) and (solidity > 0.75) and (area > h * w * 0.02)

        if is_face:
            continue  # IGNORE FACE

        motion_score = 1.0
        if motion_mask is not None:
            roi_motion = motion_mask[by:by+bh, bx:bx+bw]
            if roi_motion.size > 0:
                motion_score = np.mean(roi_motion) / 255.0 + 0.1

        score = area * (1.3 - solidity) * motion_score
        if by > h * 0.25:
            score *= 1.5

        candidate_hand_boxes.append((score, (bx, by, bw, bh), c))

    if candidate_hand_boxes:
        candidate_hand_boxes.sort(key=lambda item: item[0], reverse=True)
        best_score, (bx, by, bw, bh), hand_c = candidate_hand_boxes[0]
        _LAST_HAND_BBOX = (bx, by, bw, bh)
        hand_detected = True
    elif _LAST_HAND_BBOX is not None:
        bx, by, bw, bh = _LAST_HAND_BBOX
        hand_detected = True

    if hand_detected:
        pad_x = int(bw * 0.08)
        pad_y = int(bh * 0.08)
        xmin = max(0, bx - pad_x)
        xmax = min(w, bx + bw + pad_x)
        ymin = max(0, by - pad_y)
        ymax = min(h, by + bh + pad_y)

        cw, ch = xmax - xmin, ymax - ymin
        side = max(cw, ch)
        cx, cy = (xmin + xmax) // 2, (ymin + ymax) // 2
        xmin = max(0, cx - side // 2)
        xmax = min(w, cx + side // 2)
        ymin = max(0, cy - side // 2)
        ymax = min(h, cy + side // 2)

        # NATURAL RAW CAMERA CROP OF HAND/ARM ONLY (NO WHITE MASKING)
        crop = image_rgb[ymin:ymax, xmin:xmax]
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)
        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (251, 146, 60), 2)

    if not hand_detected or crop is None or crop.size == 0:
        side = int(min(h, w) * 0.45)
        cy, cx = int(h * 0.6), w // 2
        ymin, ymax = max(0, cy - side // 2), min(h, cy + side // 2)
        xmin, xmax = max(0, cx - side // 2), min(w, cx + side // 2)

        crop = image_rgb[ymin:ymax, xmin:xmax]
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

    return crop, annotated, hand_detected, bbox


def predict_sign(model, image_input, prev_frame=None):
    """
    Predict ASL sign from image array (RGB) or PIL Image.
    Crops tightly around the hand/arm using natural original camera pixels.
    """
    if isinstance(image_input, Image.Image):
        image_rgb = np.array(image_input.convert("RGB"))
    else:
        image_rgb = np.array(image_input)
        if len(image_rgb.shape) == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    prev_rgb = None
    if prev_frame is not None:
        if isinstance(prev_frame, Image.Image):
            prev_rgb = np.array(prev_frame.convert("RGB"))
        else:
            prev_rgb = np.array(prev_frame)

    crop_natural, annotated_rgb, hand_detected, bbox = isolate_hand_only_erase_body_face(image_rgb, prev_rgb)

    resized = cv2.resize(crop_natural, (64, 64))
    input_batch = np.expand_dims(resized.astype(np.float32), axis=0)

    logits = model.predict(input_batch, verbose=0)
    probs = tf.nn.softmax(logits[0]).numpy()

    top_idx = int(np.argmax(probs))
    top_label = CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx]) * 100.0

    if bbox:
        xmin, ymin, bw, bh = bbox
        label_text = f"{top_label} ({confidence:.1f}%)"
        cv2.putText(
            annotated_rgb,
            label_text,
            (xmin, max(25, ymin - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (251, 146, 60),
            2,
            cv2.LINE_AA,
        )

    return {
        "label": top_label,
        "confidence": confidence,
        "probs": probs,
        "annotated_frame": annotated_rgb,
        "hand_crop": crop_natural,
        "hand_detected": hand_detected,
        "class_names": CLASS_NAMES,
    }
