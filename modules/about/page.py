import streamlit as st
from ui_components import render_hero, render_footer, render_platform_metrics

def render_page():
    render_hero("👨‍💻", "About the Developer", "Swayam • Elevance Skills Internship Project")
    
    st.markdown("""
    <div style="background:#1c1c1e;border:1px solid #2c2c2e;border-radius:18px;padding:2rem;margin-bottom:2rem;">
        <h2 style="margin-top:0;font-size:1.8rem;letter-spacing:-0.03em;">Hi, I'm Swayam 👋</h2>
        <p style="color:#8e8e93;font-size:1.05rem;line-height:1.6;margin-bottom:1.5rem;">
            I am a passionate Machine Learning enthusiast and computer vision developer. This <strong>MultiVision AI Suite</strong> represents the capstone submission for my Elevance Skills Training & Internship program.
        </p>
        <div style="display:flex; gap: 1rem; flex-wrap: wrap;">
            <a href="https://github.com/swayamworks/facial-emotion-recog" target="_blank" style="text-decoration:none;">
                <div style="background:#ffffff; color:#000000; padding:0.6rem 1.2rem; border-radius:8px; font-weight:600; display:inline-flex; align-items:center; gap:0.5rem; border:1px solid #ffffff; transition:all 0.2s;">
                    ⭐ View on GitHub
                </div>
            </a>
            <a href="mailto:training@elevanceskills.com" style="text-decoration:none;">
                <div style="background:transparent; color:#ffffff; padding:0.6rem 1.2rem; border-radius:8px; font-weight:600; display:inline-flex; align-items:center; gap:0.5rem; border:1px solid #2c2c2e; transition:all 0.2s;">
                    ✉️ Contact
                </div>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎓 Internship Delivery Breakdown")
    st.markdown("""
    To fulfill the strict grading criteria of the internship instructions, this project is architected into two distinct tiers:
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div style="background:#1c1c1e;border:1px solid #2c2c2e;border-radius:14px;padding:1.5rem;height:100%;">
            <h4 style="margin-top:0;color:#ffffff;">🛠️ Training Project Base</h4>
            <ul style="color:#8e8e93;line-height:1.6;padding-left:1.2rem;">
                <li><strong>Facial Emotion Detection</strong> (CNN)</li>
                <li><strong>Animal Detection</strong> (YOLOv8 Object Detection)</li>
                <li><strong>Car Colour Detection</strong> (Vehicle tracking)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div style="background:#1c1c1e;border:1px solid #2c2c2e;border-radius:14px;padding:1.5rem;height:100%;">
            <h4 style="margin-top:0;color:#ffffff;">🚀 Internship Extra Features</h4>
            <ul style="color:#8e8e93;line-height:1.6;padding-left:1.2rem;">
                <li><strong>Sign Language Detection</strong> (Hybrid MediaPipe + MLP)</li>
                <li><strong>Emotion Detection (Voice)</strong> (Audio Analysis)</li>
                <li><strong>Drowsiness Detection</strong> (Eye Aspect Ratio Tracking)</li>
                <li><strong>Nationality & Appearance</strong> (Multi-attribute AI)</li>
                <li><strong>Unified UI Engine</strong> (Custom Apple-inspired theme)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    render_footer()
