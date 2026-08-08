import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st

from ui_components import (ACCENT_COLORS, inject_global_css, render_footer,
                           render_module_card, render_platform_metrics)

st.set_page_config(page_title="MultiVision AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

inject_global_css()

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "run_id" not in st.session_state:
    st.session_state.run_id = 0

MODULES = [
    ("1", "🚗", "Car Colour Detection", "Detect car colors, count cars and people at traffic signals.", "car"),
    ("2", "🐾", "Animal Detection", "Detect and classify animals. Highlight carnivorous animals.", "animal"),
    ("3", "🎤", "Emotion Detection (Voice)", "Detect emotions from female voice recordings or uploads.", "voice"),
    ("4", "🌐", "Nationality Detection", "Predict nationality, emotion and other attributes from face images.", "nationality"),
    ("5", "🤟", "Sign Language Detection", "Recognize sign language gestures (6 PM - 10 PM operational).", "sign"),
    ("6", "👁️", "Drowsiness Detection", "Detect drowsiness in drivers, count people and predict age.", "drowsiness"),
    ("7", "😊", "Facial Emotion", "Predict facial expressions using CNNs.", "emotion"),
]

with st.sidebar:
    st.markdown('<div class="sidebar-brand">MultiVision Workspace</div>', unsafe_allow_html=True)
    if st.button("✧  Dashboard", key="nav_Dashboard", use_container_width=True, type="primary" if st.session_state.page == "Dashboard" else "secondary"):
        st.session_state.page = "Dashboard"
        st.rerun()

    st.markdown('<div class="sidebar-kicker">Vision Capabilities</div>', unsafe_allow_html=True)
    for idx, icon, title, _, _ in MODULES:
        label = f"Task {idx}\n{title}"
        if st.button(f"{icon}  {title}", key=f"nav_{title}", use_container_width=True, type="primary" if st.session_state.page == title else "secondary"):
            st.session_state.page = title
            st.rerun()
            
    st.markdown('<div class="sidebar-kicker">Actions</div>', unsafe_allow_html=True)
    if st.button("↻  Clear Inputs", use_container_width=True):
        st.session_state.run_id += 1
        st.rerun()
    
    if st.button("👨‍💻  About Developer", key="nav_About", use_container_width=True, type="primary" if st.session_state.page == "About Developer" else "secondary"):
        st.session_state.page = "About Developer"
        st.rerun()


def render_home():
    st.markdown('<div class="product-hero"><div class="product-eyebrow">Internship Project • Swayam</div><div class="product-title">AI Vision Suite.</div><div class="product-subtitle">Select a capability from the sidebar to begin processing visual or audio data. Complete all tasks below.</div></div>', unsafe_allow_html=True)

    render_platform_metrics([
        ("7", "Core Capabilities"),
        ("4+", "AI Architectures (YOLO, CNN, SVM, MLP)"),
        ("100%", "Offline Ready")
    ])

    c1, c2 = st.columns(2)
    cols = [c1, c2]

    for i, (idx, icon, title, desc, accent_key) in enumerate(MODULES):
        with cols[i % 2]:
            render_module_card(icon, title, desc, ACCENT_COLORS[accent_key])
            st.markdown('<div class="module-action">', unsafe_allow_html=True)
            if st.button(f"Open {title}", key=f"card_{title}", use_container_width=True):
                st.session_state.page = title
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    render_footer()

def main():
    page = st.session_state.page
    if page == "Dashboard":
        render_home()
    elif page == "About Developer":
        from modules.about.page import render_page
        render_page()
    elif page == "Facial Emotion":
        from modules.emotion.page import render_page
        render_page()
    elif page == "Animal Detection":
        from modules.animal.page import render_page
        render_page()
    elif page == "Car Colour Detection":
        from modules.car.page import render_page
        render_page()
    elif page == "Emotion Detection (Voice)":
        from modules.voice.page import render_page
        render_page()
    elif page == "Sign Language Detection":
        from modules.sign.page import render_page
        render_page()
    elif page == "Drowsiness Detection":
        from modules.drowsiness.page import render_page
        render_page()
    elif page == "Nationality Detection":
        from modules.nationality.page import render_page
        render_page()
    else:
        st.info("This module is currently being prepared.")

if __name__ == "__main__":
    main()
