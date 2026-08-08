import os
import tempfile
import time

import streamlit as st

from modules.voice.predictor import VoicePredictor
from ui_components import (ACCENT_COLORS, close_media_card, close_upload_card,
                           render_empty_state, render_footer, render_hero,
                           render_inference_time, render_media_card,
                           render_model_info, render_progress_steps,
                           render_result_card, render_warning_card, render_workflow_summary,
                           upload_card)

ACCENT = ACCENT_COLORS["voice"]


@st.cache_resource(show_spinner="Loading voice recognition models...")
def get_predictor():
    try:
        return VoicePredictor()
    except Exception as e:
        st.error(f"Failed to load voice models: {e}")
        return None


def render_page():
    render_hero("🎤", "Speech Emotion Recognition", "Detect speaker gender and emotional state from speech.")
    render_workflow_summary("Upload a WAV recording to extract acoustic features, identify the speaker profile, and estimate the expressed emotion.", ["WAV audio", "MFCC + Chroma + Mel", "Random Forest", "XGBoost"])
    predictor = get_predictor()
    if predictor is None:
        st.stop()

    upload_card("Upload or Record audio", "WAV", "🎤")
    run_suffix = st.session_state.get('run_id', 0)
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"], label_visibility="collapsed", key=f"voice_uploader_{run_suffix}")
    recorded_audio = st.audio_input("Record a voice message", label_visibility="collapsed", key=f"voice_recorder_{run_suffix}")
    close_upload_card()
    
    audio_source = recorded_audio if recorded_audio else uploaded_file
    
    if audio_source is None:
        render_empty_state("🎵", "Upload or record an audio file to begin analysis", "WAV")
        with st.expander("Model information"):
            render_model_info([{"label": "Gender", "value": "Random Forest"}, {"label": "Emotion", "value": "XGBoost"}, {"label": "Features", "value": "MFCC + Chroma + Mel"}, {"label": "Dataset", "value": "RAVDESS"}])
        render_footer()
        return

    render_media_card("Captured audio")
    st.audio(audio_source, format="audio/wav")
    close_media_card()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        temp_path = tmp.name
        tmp.write(audio_source.getbuffer())

    progress_area = st.empty()
    progress_area.markdown("<div class='status-card'><div class='status-title'>Analysis progress</div><div class='status-step'>• &nbsp; Extracting audio features...</div></div>", unsafe_allow_html=True)
    start = time.time()
    try:
        result = predictor.predict(temp_path)
        elapsed = time.time() - start
    except Exception as e:
        progress_area.empty()
        st.error(f"Prediction failed.\n\n{e}")
        return
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    progress_area.empty()
    render_progress_steps([("Features extracted", True), (f"{result['gender']} speaker detected", True), ("Emotion prediction complete" if result["emotion"] else "Emotion prediction unavailable for this speaker", True)])
    if result["gender"] == "Male":
        render_warning_card("Male speaker detected", "This module currently supports emotion prediction for female voices only.")
        col1, col2 = st.columns(2)
        with col1: render_result_card("Gender", "Male", result["gender_confidence"], ACCENT)
        with col2: render_result_card("Emotion", "—", accent_color=ACCENT)
        render_inference_time(elapsed)
        render_footer()
        return

    col1, col2 = st.columns(2)
    with col1: render_result_card("Gender", "Female", result["gender_confidence"], ACCENT)
    with col2: render_result_card("Emotion", result["emotion"], result["emotion_confidence"], ACCENT)
    render_inference_time(elapsed)
    with st.expander("Technical details"):
        render_model_info([{"label": "Gender", "value": result["gender"]}, {"label": "Gender confidence", "value": f"{result['gender_confidence']}%"}, {"label": "Emotion", "value": result["emotion"]}, {"label": "Feature count", "value": "262"}])
    render_footer()
