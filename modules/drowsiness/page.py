import os
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from modules.drowsiness.detector import DrowsinessDetector
from ui_components import (
    ACCENT_COLORS,
    close_upload_card,
    render_empty_state,
    render_footer,
    render_hero,
    render_inference_time,
    render_model_info,
    render_progress_steps,
    render_result_card,
    render_workflow_summary,
    upload_card,
)

ACCENT = ACCENT_COLORS["drowsiness"]

def get_detector():
    return DrowsinessDetector()

def render_page():
    render_hero(
        "😴",
        "Driver Drowsiness & Age Detection",
        "Monitor driver and passenger alertness in real-time. Marks sleeping people in RED and estimates age from a locally trained face model."
    )

    render_workflow_summary(
        "Upload an image or video to analyze occupant alertness, identify sleeping passengers with RED bounding boxes, and estimate age from a locally trained face model.",
        ["Input Media", "YOLOv8 Detection", "EfficientNet-B0", "UTKFace MobileNetV3", "Popup Alert"]
    )

    # Tab/Radio for selecting Image vs Video mode
    mode = st.radio(
        "Select Detection Mode",
        options=["📷 Image Detection", "🎥 Video Detection"],
        horizontal=True,
        key="drowsiness_mode"
    )

    detector = get_detector()

    if mode == "📷 Image Detection":
        render_image_mode(detector)
    else:
        render_video_mode(detector)

    render_footer()


def render_image_mode(detector):
    upload_card("Upload Input Image", "JPG · JPEG · PNG", "📷")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"drowsiness_img_uploader_{st.session_state.get('run_id', 0)}"
    )
    close_upload_card()

    if uploaded_file is None:
        render_empty_state("🖼️", "Upload an image of a vehicle interior to analyze alertness", "JPG · JPEG · PNG")
        with st.expander("📋 Model & Technical Info"):
            render_model_info([
                {"label": "Detector", "value": "YOLOv8 Person Model"},
                {"label": "Sleep Analysis", "value": "EfficientNet-B0 (best_model.pt)"},
                {"label": "Age Estimation", "value": "MobileNetV3 (UTKFace-trained)"},
                {"label": "Face Detection", "value": "Haar Cascade"},
                {"label": "Highlight", "value": "Red (Sleeping) / Green (Awake)"},
                {"label": "Alert System", "value": "Real-time Popup Notification"},
            ])
        return

    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    start_time = time.time()
    with st.spinner("Analyzing image for occupant drowsiness and age..."):
        annotated_bgr, summary = detector.process_image(image_bgr)
        elapsed = time.time() - start_time

    render_progress_steps([("Input Image Loaded", True), ("Sleep & Age Analysis Complete", True)])

    # Display Popup Alert Banner if sleeping occupants detected
    sleeping_count = summary["sleeping_count"]
    sleeping_details = summary["sleeping_details"]

    if sleeping_count > 0:
        ages_str = ", ".join([f"Person #{d['person_id']} (Age: ~{d['age']} years)" if d["age"] is not None else f"Person #{d['person_id']} (Age unavailable)" for d in sleeping_details])
        st.error(
            f"🚨 **DROWSINESS ALERT!** {sleeping_count} sleeping occupant(s) detected!\n\n"
            f"**Details & Estimated Ages:** {ages_str}"
        )
        st.toast(f"⚠️ Drowsiness Warning! {sleeping_count} person(s) asleep.", icon="🚨")
    else:
        st.success(f"✅ **ALL OCCUPANTS AWAKE:** {summary['total_people']} person(s) detected inside the vehicle.")
        st.toast("✅ All occupants are awake and alert.", icon="👍")

    # Display Original vs Annotated Images side by side
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="media-card"><div class="media-label">Original Preview</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="media-card"><div class="media-label">Drowsiness & Age Analysis (Red = Sleeping)</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("##### Face age estimates")
    age_rows = [
        {
            "Occupant": f"Person #{person['person_id']}",
            "Alertness": person["status"],
            "Estimated age": f"~{person['age']} years" if person["age"] is not None else "Face not detected",
            "Age model": "UTKFace MobileNetV3" if person["age"] is not None else "—",
        }
        for person in summary["occupant_details"]
    ]
    st.dataframe(age_rows, use_container_width=True, hide_index=True)

    # Metric Cards
    cols = st.columns(3, gap="medium")
    with cols[0]:
        render_result_card("Total Occupants", str(summary["total_people"]))
    with cols[1]:
        render_result_card("Awake Count", str(summary["awake_count"]))
    with cols[2]:
        render_result_card("Sleeping (Red Box)", str(summary["sleeping_count"]))

    render_inference_time(elapsed)

    with st.expander("🔧 Detailed Technical Breakdown"):
        details = [
            {"label": "Total Detected", "value": f"{summary['total_people']} people"},
            {"label": "Sleeping (Red)", "value": f"{summary['sleeping_count']} occupants"},
            {"label": "Awake (Green)", "value": f"{summary['awake_count']} occupants"},
            {"label": "Inference Time", "value": f"{elapsed:.2f}s"},
        ]
        render_model_info(details)


def render_video_mode(detector):
    upload_card("Upload Input Video", "MP4 · AVI · MOV", "🎥")
    uploaded_video = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
        key=f"drowsiness_vid_uploader_{st.session_state.get('run_id', 0)}"
    )
    close_upload_card()

    if uploaded_video is None:
        render_empty_state("🎥", "Upload a video clip of driver/passengers to process frame by frame", "MP4 · AVI · MOV")
        return

    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
        tmp_vid.write(uploaded_video.read())
        video_path = tmp_vid.name

    st.markdown('<div class="media-card"><div class="media-label">Video Input Preview</div>', unsafe_allow_html=True)
    st.video(video_path)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Process Video for Drowsiness", type="primary", use_container_width=True):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_placeholder = st.empty()

        temp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

        frame_count = 0
        max_sleeping_seen = 0
        sleeping_ages_recorded = set()

        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_frame, summary = detector.process_image(frame)
            out_writer.write(annotated_frame)

            if summary["sleeping_count"] > max_sleeping_seen:
                max_sleeping_seen = summary["sleeping_count"]

            for d in summary["sleeping_details"]:
                if d["age"]:
                    sleeping_ages_recorded.add(d["age"])

            # Update progress every 5 frames
            if frame_count % 5 == 0 or frame_count == total_frames:
                progress = min(1.0, frame_count / max(1, total_frames))
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}...")
                preview_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Processing Frame {frame_count}", use_container_width=True)

        cap.release()
        out_writer.release()
        elapsed = time.time() - start_time

        progress_bar.progress(1.0)
        status_text.success("✅ Video processing complete!")

        # Popup Alert for Video Results
        if max_sleeping_seen > 0:
            ages_str = ", ".join(f"~{age} years" for age in sorted(sleeping_ages_recorded)) or "unavailable"
            st.error(
                f"🚨 **DROWSINESS ALERT!** Detected up to {max_sleeping_seen} sleeping occupant(s) in video.\n\n"
                f"**Predicted Ages of Sleeping Occupants:** {ages_str}"
            )
            st.toast(f"⚠️ Video Alert: {max_sleeping_seen} sleeping person(s) detected!", icon="🚨")
        else:
            st.success("✅ **VIDEO CLEAN:** All occupants remained awake throughout the recording.")

        render_inference_time(elapsed)

        # Cleanup temp files
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
            if os.path.exists(temp_out_path):
                os.remove(temp_out_path)
        except Exception:
            pass
