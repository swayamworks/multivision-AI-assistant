"""
utils.py
Helper functions for:
  1. Dress/clothing dominant-color extraction (KMeans on the torso region)
  2. Mapping DeepFace's ethnicity/race output -> the 4 brief categories
     (Indian / United States / African / Other)

IMPORTANT LIMITATION (please read):
Nationality is a legal/citizenship status - it is not something a face or
body can visually encode, and no vision model can genuinely predict it.
What this app actually predicts is "appearance-based ethnicity/region",
using DeepFace's pretrained race classifier (trained on datasets like
FairFace/UTKFace-style labels: asian, indian, black, white, middle eastern,
latino hispanic). We map that onto your brief's 4 categories as the closest
available proxy. Treat the "nationality" output as an approximate,
appearance-based label for a class demo - not a real identity signal, and
never use this for real-world decisions about people.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
import webcolors


# ---------------------------------------------------------------------------
# 1. Dress / clothing dominant color
# ---------------------------------------------------------------------------

def get_torso_region(image_bgr, face_box):
    """
    Given the full image (BGR, numpy array) and a face bounding box
    (x, y, w, h), estimate a torso/clothing region just below the face
    and return it as a cropped BGR image. Falls back to the lower half
    of the image if the estimated box would be invalid.
    """
    img_h, img_w = image_bgr.shape[:2]
    x, y, w, h = face_box

    # Torso heuristic: start a bit below the chin, widen slightly,
    # extend down by ~2x the face height (clamped to image bounds).
    top = int(y + h * 1.15)
    bottom = int(y + h * 3.2)
    left = int(x - w * 0.35)
    right = int(x + w * 1.35)

    top = max(0, min(top, img_h - 1))
    bottom = max(top + 1, min(bottom, img_h))
    left = max(0, min(left, img_w - 1))
    right = max(left + 1, min(right, img_w))

    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        # fallback: lower half of the whole image
        crop = image_bgr[img_h // 2:, :]
    return crop


def _closest_color_name(rgb_tuple):
    """Map an arbitrary RGB tuple to the closest CSS3 color name."""
    try:
        return webcolors.rgb_to_name(rgb_tuple)
    except ValueError:
        # webcolors >= 4 renamed CSS3_HEX_TO_NAMES -> CSS3_NAMES_TO_HEX (inverted)
        if hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
            hex_to_name = webcolors.CSS3_HEX_TO_NAMES
        elif hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
            hex_to_name = {v: k for k, v in webcolors.CSS3_NAMES_TO_HEX.items()}
        else:
            # Absolute fallback to the basic color table below
            return closest_basic_color_name(rgb_tuple)
        min_dist = None
        closest_name = "unknown"
        for hex_code, name in hex_to_name.items():
            r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
            dist = (r_c - rgb_tuple[0]) ** 2 + (g_c - rgb_tuple[1]) ** 2 + (b_c - rgb_tuple[2]) ** 2
            if min_dist is None or dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name


# webcolors >= 4 renamed the constant map; build a manual fallback table
# so this works across versions without extra dependencies.
_BASIC_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
    "lime": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "cyan": (0, 255, 255), "magenta": (255, 0, 255), "silver": (192, 192, 192),
    "gray": (128, 128, 128), "maroon": (128, 0, 0), "olive": (128, 128, 0),
    "green": (0, 128, 0), "purple": (128, 0, 128), "teal": (0, 128, 128),
    "navy": (0, 0, 128), "orange": (255, 165, 0), "pink": (255, 192, 203),
    "brown": (165, 42, 42), "beige": (245, 245, 220), "gold": (255, 215, 0),
}


def closest_basic_color_name(rgb_tuple):
    """Simple, dependency-light nearest-color-name lookup."""
    best_name, best_dist = "unknown", None
    for name, ref_rgb in _BASIC_COLORS.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb_tuple, ref_rgb))
        if best_dist is None or dist < best_dist:
            best_dist, best_name = dist, name
    return best_name


def dominant_dress_color(image_bgr, face_box, k=3):
    """
    Returns (color_name, rgb_tuple) for the dominant clothing color
    found in the torso region below the given face box.
    """
    torso = get_torso_region(image_bgr, face_box)
    torso_rgb = cv2.cvtColor(torso, cv2.COLOR_BGR2RGB)

    # Downsample for speed, reshape to a list of pixels
    small = cv2.resize(torso_rgb, (60, 60), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    k = min(k, len(pixels))
    kmeans = KMeans(n_clusters=k, n_init=4, random_state=42)
    labels = kmeans.fit_predict(pixels)
    counts = np.bincount(labels)
    dominant_cluster = np.argmax(counts)
    dominant_rgb = tuple(int(v) for v in kmeans.cluster_centers_[dominant_cluster])

    color_name = closest_basic_color_name(dominant_rgb)
    return color_name, dominant_rgb


# ---------------------------------------------------------------------------
# 2. Ethnicity/race -> brief category mapping
# ---------------------------------------------------------------------------

# Your trained model (train_age_ethnicity.py) outputs one of:
# 'white', 'black', 'asian', 'indian', 'others'
def map_to_category(dominant_race: str) -> str:
    """
    Maps the trained model's dominant_race output to one of the four brief
    buckets. This is a coarse, imperfect proxy (see module docstring) — it
    is NOT a real nationality/citizenship determination.
    """
    race = (dominant_race or "").strip().lower()
    if race == "indian":
        return "Indian"
    if race == "black":
        return "African"
    if race == "white":
        # Closest available proxy for "United States" in the brief;
        # in reality 'white' is an appearance category, not a nationality.
        return "United States"
    # asian, others, or anything else
    return "Other"
