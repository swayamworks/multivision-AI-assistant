"""
app.py — Streamlit GUI for the "Nationality/Emotion/Age/Dress-Color" demo.

Run with:
    streamlit run app.py

Pipeline (now using YOUR own transfer-learned models instead of DeepFace):
  1. User uploads an image -> shown as a preview.
  2. OpenCV Haar cascade finds the face box.
  3. Your MobileNetV2 age+ethnicity model (train_age_ethnicity.py) predicts
     age + a raw ethnicity-proxy label; your emotion model
     (train_emotion.py, or your existing RAF-DB model) predicts emotion.
  4. dominant_race is mapped to one of 4 categories: Indian / United States
     / African / Other  (see utils.map_to_category for the honest caveat
     about why this is an appearance-proxy, not real nationality).
  5. Depending on category, only the fields required by the brief are shown:
       Indian          -> age, dress color, emotion
       United States    -> age, emotion
       African          -> dress color, emotion
       Other            -> category label, emotion
  6. Dress color is computed separately with OpenCV + KMeans on the torso
     region below the detected face box (utils.dominant_dress_color).

Before running: update MODEL_PATHS below to point at your trained .h5 files.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from utils import dominant_dress_color, map_to_category
from inference_utils import ModelBundle, detect_face_box

# --- point these at your trained model files ---
MODEL_PATHS = {
    "age_ethnicity": "models/age_ethnicity_model.h5",
    "emotion": "../emotion/weights/emotion_cnn_baseline.keras",
}


@st.cache_resource
def load_models():
    return ModelBundle(MODEL_PATHS["age_ethnicity"], MODEL_PATHS["emotion"])

st.set_page_config(page_title="Face Attribute Predictor", layout="centered")

st.title("🧑\u200d🦱 Face Attribute Predictor")
st.caption(
    "Predicts emotion for every face, plus age/dress-color/category depending "
    "on the predicted appearance-region. See the note below the results for "
    "an important caveat about what 'nationality' means here."
)

# ---------------------------------------------------------------------------
# Sidebar: model status / info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "- Age + ethnicity-proxy: **your MobileNetV2 transfer-learning "
        "model**, trained on UTKFace (`train_age_ethnicity.py`).\n"
        "- Emotion: **your MobileNetV2 transfer-learning model**, trained "
        "on your emotion dataset (`train_emotion.py`), or your existing "
        "RAF-DB model.\n"
        "- Face detection: OpenCV Haar cascade (no download needed).\n"
        "- Dress color: OpenCV + KMeans on the region below the face."
    )
    st.warning(
        "⚠️ 'Nationality' here is really an appearance/ethnicity proxy — "
        "faces do not encode citizenship. Treat results as a class-project "
        "approximation, not a real identity signal.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Upload + preview
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a photo (JPG/PNG, ideally a clear frontal face)",
    type=["jpg", "jpeg", "png"],
)

col_preview, col_results = st.columns(2)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    with col_preview:
        st.subheader("Input Preview")
        st.image(pil_image, use_container_width=True)

    run_button = st.button("🔍 Analyze", type="primary")

    if run_button:
        with st.spinner("Running face analysis..."):
            try:
                bundle = load_models()
            except FileNotFoundError as fnf:
                st.error(str(fnf))
                st.stop()

            try:
                img_rgb = np.array(pil_image)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                face_box = detect_face_box(img_bgr)
                if face_box is None:
                    raise ValueError("no face detected")

                result = bundle.predict(img_bgr, face_box)

                age = round(result["age"])
                emotion = result["dominant_emotion"]
                emotion_conf = result["emotion_confidence"]
                dominant_race = result["dominant_race"]

                category = map_to_category(dominant_race)

                dress_color_name = None
                dress_rgb = None
                if category in ("Indian", "African"):
                    dress_color_name, dress_rgb = dominant_dress_color(img_bgr, face_box)

                with col_results:
                    st.subheader("Results")
                    st.markdown(f"**Predicted category:** {category}")
                    st.markdown(
                        f"*(raw appearance/ethnicity signal: `{dominant_race}` "
                        f"— confidence {result['race_confidence']:.0%} — "
                        f"see caveat in sidebar)*"
                    )
                    st.markdown(f"**Emotion:** {emotion} ({emotion_conf:.0%})")

                    if category == "Indian":
                        st.markdown(f"**Age:** {age}")
                        st.markdown(f"**Dress color:** {dress_color_name}")
                    elif category == "United States":
                        st.markdown(f"**Age:** {age}")
                    elif category == "African":
                        st.markdown(f"**Dress color:** {dress_color_name}")
                    else:  # Other
                        st.markdown(f"**Category:** {category}")

                    if dress_color_name is not None and dress_rgb is not None:
                        swatch = f"""
                        <div style="width:40px;height:40px;border:1px solid #999;
                        background-color:rgb{dress_rgb};display:inline-block;"></div>
                        """
                        st.markdown(swatch, unsafe_allow_html=True)

                    with st.expander("Full raw output"):
                        st.json(result)

            except ValueError as ve:
                st.error(
                    "No face could be detected in this image. Try a clearer, "
                    f"more front-facing photo.\n\nDetails: {ve}"
                )
            except Exception as e:
                st.error(f"Analysis failed: {e}")
else:
    st.info("Upload an image to get started.")
