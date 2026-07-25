import os
import sys
import time
import subprocess
import tempfile
import cv2
import av
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from modules.sign.predict import (
    load_model,
    predict_sign,
    CLASS_NAMES,
    SIGN_EMOJIS,
)
from modules.sign.assistant import (
    SequenceAccumulator,
    process_video_file,
    generate_tts_audio,
)
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

ACCENT = ACCENT_COLORS["sign"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sign_model.keras")
REALTIME_SCRIPT = os.path.join(BASE_DIR, "realtime_assistant.py")


@st.cache_resource(show_spinner="Loading sign language recognition model...")
def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading sign model: {e}")
        return None


def init_session_state():
    if "sign_sentence" not in st.session_state:
        st.session_state.sign_sentence = ""
    if "accumulator" not in st.session_state:
        st.session_state.accumulator = SequenceAccumulator(debounce_frames=2)
    if "is_streaming" not in st.session_state:
        st.session_state.is_streaming = False


def render_audio_player(text, key_suffix=""):
    if not text or not text.strip():
        return
    audio_bytes = generate_tts_audio(text)
    if audio_bytes:
        st.audio(audio_bytes, format="audio/mp3")


def render_image_mode(model):
    st.markdown("##### 📸 Image Gesture Recognition")
    upload_card("Upload a hand sign image", "JPG · JPEG · PNG", "🤟")
    uploaded_file = st.file_uploader(
        "Upload sign image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="sign_img_uploader",
    )
    close_upload_card()

    if uploaded_file is None:
        render_empty_state("🖼️", "Upload a hand gesture image to begin analysis", "JPG · JPEG · PNG")
        return

    try:
        image = Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("Invalid image file.")
        return
    except Exception as e:
        st.error(f"Error reading image: {e}")
        return

    start_time = time.time()
    result = predict_sign(model, image)
    elapsed = time.time() - start_time

    render_progress_steps([
        ("MediaPipe Hand ROI Extracted", result["hand_detected"]),
        ("CNN Gesture Prediction Complete", True),
    ])

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        sub_tab1, sub_tab2 = st.tabs(["📸 Full Camera Frame", "🖐️ Natural Hand/Arm Crop"])
        with sub_tab1:
            st.image(result["annotated_frame"], caption="Detected Hand Bounding Box", use_container_width=True)
        with sub_tab2:
            st.image(result["hand_crop"], caption="Natural Hand/Arm Crop Fed to Model", use_container_width=True)

    with col_result:
        label = result["label"]
        emoji = SIGN_EMOJIS.get(label, "🤟")
        conf = result["confidence"]
        display_title = f"{emoji} Sign: {label.upper()}"

        render_result_card("Detected Gesture", display_title, conf, ACCENT)

        if not result["hand_detected"]:
            st.info("ℹ️ Hand landmarks not automatically detected. Used center frame ROI for prediction.")

        st.markdown("")
        st.markdown("**🔊 Voice Assistant Readout**")
        render_audio_player(label, key_suffix="img")

        render_inference_time(elapsed)

    st.markdown("")
    st.markdown("##### Top Probabilities across ASL Classes")
    probs = result["probs"]
    top_indices = np.argsort(probs)[::-1][:7]

    for idx in top_indices:
        cls_name = CLASS_NAMES[idx]
        prob_val = float(probs[idx])
        cls_emoji = SIGN_EMOJIS.get(cls_name, "")

        c1, c2 = st.columns([4, 1])
        with c1:
            st.progress(prob_val, text=f"{cls_emoji} Gesture {cls_name}")
        with c2:
            st.markdown(f"**{prob_val * 100:.1f}%**")


def render_video_mode(model):
    st.markdown("##### 🎥 Video File Assistant")
    upload_card("Upload a recorded sign video clip", "MP4 · MOV · AVI", "🎥")
    uploaded_video = st.file_uploader(
        "Upload video clip",
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed",
        key="sign_video_uploader",
    )
    close_upload_card()

    if uploaded_video is None:
        render_empty_state("📹", "Upload a video file to translate gestures to continuous transcript", "MP4 · MOV · AVI")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(uploaded_video.getbuffer())
        temp_video_path = tmp_vid.name

    try:
        start_time = time.time()
        with st.spinner("Processing video frames & translating sign gestures..."):
            summary = process_video_file(model, temp_video_path, max_frames=150, sample_rate=3)
        elapsed = time.time() - start_time

        transcript = summary["transcript"]
        timeline = summary["timeline"]
        key_frames = summary["key_frames"]

        render_progress_steps([
            (f"Processed {summary['total_processed_frames']} video frames", True),
            ("Transcript & Sequence Decoded", True),
        ])

        st.markdown("##### 📝 Decoded Video Transcript")
        if transcript.strip():
            st.markdown(
                f'''<div class="result-card" style="border-color:{ACCENT}50">
                    <div class="result-label">Translated Sign Text</div>
                    <div class="result-value" style="font-size:1.8rem;color:#fdfdfd;letter-spacing:.02em">{transcript}</div>
                    <div class="result-confidence">Confidence & Length: {len(transcript)} characters decoded</div>
                </div>''',
                unsafe_allow_html=True,
            )
            st.markdown("")
            st.markdown("**🔊 Listen to Translated Transcript**")
            render_audio_player(transcript, key_suffix="vid_transcript")
        else:
            st.warning("No continuous text sequence decoded from video frames. Try a video with clearer hand signs.")

        if key_frames:
            st.markdown("")
            st.markdown("##### 🖼️ Sample Video Keyframes & Hand Tracking")
            cols = st.columns(min(len(key_frames), 4))
            for i, kframe in enumerate(key_frames[:8]):
                with cols[i % 4]:
                    st.image(kframe, caption=f"Frame {i+1}", use_container_width=True)

        with st.expander("⏱️ Detailed Frame Prediction Timeline"):
            st.write(timeline)

        render_inference_time(elapsed)

    except Exception as e:
        st.error(f"Error processing video: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


class SignVideoProcessor:
    def __init__(self, model):
        self.model = model
        self.accumulator = SequenceAccumulator(debounce_frames=3, min_confidence=35.0)
        self.latest_label = "Waiting..."
        self.latest_confidence = 0.0

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        result = predict_sign(self.model, img_rgb)
        crop_rgb = cv2.resize(result["hand_crop"], (600, 600))
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        self.latest_label = result["label"]
        self.latest_confidence = result["confidence"]
        self.accumulator.update(self.latest_label, self.latest_confidence)

        return av.VideoFrame.from_ndarray(crop_bgr, format="bgr24")


def render_realtime_mode(model):
    st.markdown("##### 📹 100% Real-Time Live Camera Assistant")
    st.markdown("Zero-click 30 FPS continuous streaming with real-time hand tracking, white background noise removal, and instant text-to-speech!")

    # Prominent Launch Dedicated Window Option
    st.markdown(
        f'''<div class="result-card" style="border-color:{ACCENT};background:linear-gradient(135deg,#191e2b,#121620);margin-bottom:1.5rem">
            <div class="result-label">🔥 Recommended 30 FPS Experience</div>
            <div class="result-value" style="font-size:1.45rem;color:#fff">Dedicated Real-Time Camera Window</div>
            <div style="font-size:.85rem;color:#a0aec0;margin-top:.4rem">Launches a smooth 30 FPS live desktop window with real-time HUD, live hand bounding box tracking, automatic sentence builder, and keyboard shortcuts ([SPACE], [BACKSPACE], [C] Clear, [S] Speak).</div>
        </div>''',
        unsafe_allow_html=True,
    )

    col_launch, col_stream_opt = st.columns([1.2, 1], gap="large")

    with col_launch:
        if st.button("🚀 Launch Dedicated 30 FPS Real-Time Window", type="primary", use_container_width=True):
            python_exe = sys.executable
            subprocess.Popen([python_exe, REALTIME_SCRIPT], cwd=os.path.dirname(REALTIME_SCRIPT))
            st.success("🟢 Real-Time Camera Window launched! Look at your desktop/taskbar for the MultiVision AI live window.")

    with col_stream_opt:
        stream_option = st.radio("In-Browser Mode", ["⚡ Native OpenCV Loop", "🌐 WebRTC Stream"], horizontal=True)

    st.markdown("---")

    col_cam, col_builder = st.columns([1, 1], gap="large")

    with col_builder:
        st.markdown("##### 💬 Live Sentence Builder")
        current_sentence = st.session_state.sign_sentence

        st.markdown(
            f'''<div class="result-card" style="border-color:{ACCENT}70;min-height:110px">
                <div class="result-label">Real-Time Translated Text</div>
                <div class="result-value" style="font-size:1.6rem;color:#fff">{current_sentence if current_sentence else '<span style="color:#667085">Streaming live signs...</span>'}</div>
                <div class="result-confidence">Length: {len(current_sentence)} characters | Words: {len(current_sentence.split()) if current_sentence else 0}</div>
            </div>''',
            unsafe_allow_html=True,
        )

        st.markdown("")
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        with b_col1:
            if st.button("␣ Space", use_container_width=True):
                if current_sentence and not current_sentence.endswith(" "):
                    st.session_state.sign_sentence += " "
                    st.rerun()
        with b_col2:
            if st.button("⌫ Del", use_container_width=True):
                if len(st.session_state.sign_sentence) > 0:
                    st.session_state.sign_sentence = st.session_state.sign_sentence[:-1]
                    st.rerun()
        with b_col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.sign_sentence = ""
                st.session_state.accumulator.clear()
                st.rerun()
        with b_col4:
            if st.button("🔊 Speak", use_container_width=True, type="primary"):
                pass

        if current_sentence:
            render_audio_player(current_sentence, key_suffix="live_audio")

    with col_cam:
        if stream_option == "⚡ Native OpenCV Loop":
            c1, c2 = st.columns(2)
            with c1:
                start_btn = st.button("▶ Start In-Browser Stream", type="primary", use_container_width=True)
            with c2:
                stop_btn = st.button("⏹ Stop Stream", use_container_width=True)

            if stop_btn:
                st.session_state.is_streaming = False
                st.rerun()

            if start_btn:
                st.session_state.is_streaming = True

            if st.session_state.is_streaming:
                st.info("🟢 Live camera active - Press 'Stop Stream' above to end loop.")
                cap = cv2.VideoCapture(0)
                frame_window = st.empty()
                status_window = st.empty()

                acc = st.session_state.accumulator

                while cap.isOpened() and st.session_state.is_streaming:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        st.error("Could not access local webcam.")
                        break

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    res = predict_sign(model, frame_rgb)

                    crop_display = cv2.resize(res["hand_crop"], (500, 500))
                    frame_window.image(crop_display, caption=f"🖐️ Natural Hand/Arm Crop | Sign: {res['label']} ({res['confidence']:.1f}%)", use_container_width=True)

                    changed = acc.update(res["label"], res["confidence"])
                    if changed:
                        st.session_state.sign_sentence = acc.get_text()

                    status_window.markdown(f"**Live Sign:** {res['label']} ({res['confidence']:.1f}%) | **Cut-Out Status:** {'Hand Detected' if res['hand_detected'] else 'Center ROI Fallback'}")
                    time.sleep(0.03)

                cap.release()

        else:
            ctx = webrtc_streamer(
                key="sign-webrtc-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
                video_processor_factory=lambda: SignVideoProcessor(model),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if ctx.video_processor:
                st.caption(f"Latest Sign: **{ctx.video_processor.latest_label}** ({ctx.video_processor.latest_confidence:.1f}%)")
                live_text = ctx.video_processor.accumulator.get_text()
                if live_text:
                    st.session_state.sign_sentence = live_text


def render_page():
    init_session_state()
    render_hero("🤟", "Sign Language Video Assistant", "Recognize ASL signs in real-time camera streams, video clips, or gesture images with text-to-speech.")

    model = get_model()
    if model is None:
        st.stop()

    render_workflow_summary(
        "Translate American Sign Language (ASL) into real-time text and speech assistant using CNN classification and MediaPipe hand landmark tracking.",
        ["MediaPipe Hands", "64×64 CNN", "29 ASL Classes", "Sentence Builder", "gTTS Voice Assistant"],
    )

    mode = st.radio(
        "Select Input Mode",
        ["📹 Real-Time Live Stream", "🎥 Video File Assistant", "📸 Single Image"],
        horizontal=True,
        key="sign_mode_radio",
    )

    st.markdown("")

    if mode == "📹 Real-Time Live Stream":
        render_realtime_mode(model)
    elif mode == "🎥 Video File Assistant":
        render_video_mode(model)
    else:
        render_image_mode(model)

    st.markdown("")
    with st.expander("📋 Model Information & Technical Details"):
        render_model_info([
            {"label": "Model", "value": "Sequential CNN"},
            {"label": "Hand Tracking", "value": "MediaPipe / Skin ROI"},
            {"label": "Dataset", "value": "ASL Alphabet (87k)"},
            {"label": "Classes", "value": "29 Gestures"},
        ])

    render_footer()
