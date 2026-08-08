import os
import time
import streamlit as st
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError

from modules.nationality.inference_utils import ProductionDeepFaceBundle_v2, detect_face_box
from ui_components import (
    render_hero,
    render_empty_state,
    render_result_card,
    render_model_info,
    render_inference_time,
    render_footer,
    ACCENT_COLORS,
    upload_card,
    close_upload_card,
    render_progress_steps,
    render_workflow_summary,
)

ACCENT = ACCENT_COLORS["nationality"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = {
    "age_ethnicity": os.path.join(BASE_DIR, "models", "age_ethnicity_model.h5"),
    "emotion": os.path.join(BASE_DIR, "..", "emotion", "weights", "emotion_cnn_baseline.keras"),
}

@st.cache_resource(show_spinner="Loading Nationality & Appearance models...")
def get_bundle():
    try:
        return ProductionDeepFaceBundle_v2()
    except Exception as e:
        st.error(str(e))
        return None


def render_page():
    render_hero("🧑‍🦱", "Nationality & Appearance", "Predict appearance, age, and emotion from facial features.")

    bundle = get_bundle()
    if bundle is None:
        st.stop()

    render_workflow_summary(
        "Upload a face image to classify appearance, estimate age, recognize emotion, and detect dress color based on appearance category.",
        ["Face image", "MobileNetV2", "Multi-task prediction"]
    )
    
    st.warning(
        "⚠️ **Note on 'Nationality'**: This module uses an appearance/ethnicity proxy. "
        "Faces do not encode citizenship. Treat these results as an approximation, not a definitive identity signal.",
        icon="⚠️"
    )

    upload_card("Upload a photo", "JPG · JPEG · PNG", "📸")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed", key=f"nationality_img_uploader_{st.session_state.get('run_id', 0)}")
    close_upload_card()

    if uploaded_file is None:
        render_empty_state("📸", "Upload a face image to begin analysis", "JPG · JPEG · PNG")
        # Model info
        st.markdown("")
        with st.expander("📋 Model Information"):
            render_model_info([
                {"label": "Age/Appearance Model", "value": "MobileNetV2 (Transfer Learned)"},
                {"label": "Emotion Model", "value": "CNN (RAF-DB)"},
                {"label": "Face Detector", "value": "MTCNN"},
            ])
        render_footer()
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid image file.")
        return
    except Exception as e:
        st.error(f"Error opening image: {e}")
        return

    img_rgb = np.array(image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    start = time.time()
    try:
        face_box = detect_face_box(img_bgr)
        if face_box is None:
            st.error("No face could be detected in this image. Try a clearer, more front-facing photo.")
            return

        result = bundle.predict(img_bgr, face_box)

        age = round(result["age"])
        emotion = result["dominant_emotion"]
        emotion_conf = result["emotion_confidence"]
        dominant_race = result["dominant_race"]
        # No longer mapping ethnicity to nationality per user request

    except Exception as e:
        st.error(f"Prediction failed.\n\n{e}")
        return
        
    elapsed = time.time() - start
    render_progress_steps([("Face detected", True), ("Analysis complete", True)])

    # Results
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, use_container_width=True)

    with col_result:
        render_result_card("Ethnicity", dominant_race.capitalize(), result["race_confidence"], ACCENT)
        st.markdown("")
        render_inference_time(elapsed)

    st.markdown("")
    st.markdown("##### Detailed Attributes")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Ethnicity:** {dominant_race.capitalize()}")
        st.markdown(f"**Age:** {age}")
            
    with col2:
        st.markdown(f"**Emotion:** {emotion.capitalize()} ({emotion_conf:.1%})")

    with st.expander("🔧 Full Raw Output"):
        st.json(result)

    render_footer()
