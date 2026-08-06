import streamlit as st

from ui_components import (ACCENT_COLORS, inject_global_css, render_footer,
                           render_module_card, render_platform_metrics)

st.set_page_config(page_title="MultiVision AI", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
inject_global_css()

MODULES = [
    ("😊", "Facial Emotion", "Predict facial expressions using CNNs.", "emotion"),
    ("🐾", "Animal Detection", "Find animals in an image with YOLOv8.", "animal"),
    ("🚗", "Vehicle Color Detection", "Detect cars, people, and blue vehicles.", "car"),
    ("🎤", "Speech Emotion Recognition", "Detect speaker gender and emotion.", "voice"),
    ("🤟", "Sign Language", "Recognize sign language gestures.", "sign"),
    ("😴", "Drowsiness & Age", "Estimate alertness and age from a face.", "drowsiness"),
]

NAV_ITEMS = [("⌂", "Home")] + [(icon, title) for icon, title, _, _ in MODULES]

if "page" not in st.session_state:
    st.session_state.page = "Home"

with st.sidebar:
    st.markdown('<div class="sidebar-brand">◉ &nbsp; MultiVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-kicker">Workspace</div>', unsafe_allow_html=True)
    for icon, label in NAV_ITEMS:
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True,
                     type="primary" if st.session_state.page == label else "secondary"):
            st.session_state.page = label
            st.rerun()


def render_home():
    st.markdown('''<div class="product-hero"><div class="product-eyebrow">Applied AI · Internship Project</div><div class="product-title">One workspace.<br>Multiple ways to understand.</div><div class="product-subtitle">MultiVision AI combines computer vision and speech intelligence into clear, purpose-built analysis tools—running locally from a single interface.</div></div>''', unsafe_allow_html=True)
    render_platform_metrics([
        ("04", "Live AI workspaces"),
        ("06", "Vision & audio capabilities"),
        ("100%", "Local model inference"),
        ("01", "Unified intelligence hub"),
    ])
    st.markdown('<div class="product-eyebrow" style="margin-bottom:.85rem">Explore capabilities</div>', unsafe_allow_html=True)
    rows = [MODULES[:3], MODULES[3:]]
    for row in rows:
        cols = st.columns(3, gap="medium")
        for col, (icon, title, desc, accent_key) in zip(cols, row):
            with col:
                render_module_card(icon, title, desc, ACCENT_COLORS[accent_key])
                if st.button(f"Open {title}  →", key=f"card_{title}", use_container_width=True):
                    st.session_state.page = title
                    st.rerun()
    render_footer()


def main():
    page = st.session_state.page
    if page == "Home":
        render_home()
    elif page == "Facial Emotion":
        from modules.emotion.page import render_page
        render_page()
    elif page == "Animal Detection":
        from modules.animal.page import render_page
        render_page()
    elif page == "Vehicle Color Detection":
        from modules.car.page import render_page
        render_page()
    elif page == "Speech Emotion Recognition":
        from modules.voice.page import render_page
        render_page()
    elif page == "Sign Language":
        from modules.sign.page import render_page
        render_page()
    elif page == "Drowsiness & Age":
        from modules.drowsiness.page import render_page
        render_page()
    else:
        details = next(item for item in MODULES if item[1] == page)
        st.markdown(f'<div class="page-hero"><div class="page-icon">{details[0]}</div><div class="page-title">{page}</div><div class="page-desc">This module is currently being prepared for your workspace.</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="empty-state"><div class="empty-icon">✦</div><div class="empty-title">Coming soon</div><div class="empty-copy">More intelligence tools are on the way.</div></div>', unsafe_allow_html=True)
        render_footer()


if __name__ == "__main__":
    main()
