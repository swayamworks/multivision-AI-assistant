"""Shared presentation components for the MultiVision AI interface."""

import streamlit as st


ACCENT_COLORS = {
    "emotion": "#ffffff", "animal": "#ffffff", "car": "#ffffff",
    "voice": "#ffffff", "sign": "#ffffff", "drowsiness": "#ffffff",
    "nationality": "#ffffff",
}


def inject_global_css():
    """Set the visual language once for every page."""
    st.markdown("""
    <style>
      #MainMenu, footer {visibility:hidden;}
      header[data-testid="stHeader"] {background:transparent; visibility:visible!important; z-index:99999;}
      header[data-testid="stHeader"] button {color:#8e8e93;}
      [data-testid="stSidebarCollapsedControl"] {display:flex!important; visibility:visible!important; opacity:1!important; pointer-events:auto!important;}
      [data-testid="stSidebarCollapsedControl"] button {display:flex!important; visibility:visible!important; opacity:1!important; background:#1c1c1e!important; border:1px solid #2c2c2e!important; border-radius:10px!important; color:#ffffff!important;}
      html, body, [class*="css"] {font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;}
      .stApp {background:#000000; color:#ffffff;}
      .block-container {max-width:1120px; padding:2.5rem 2.25rem 2rem;}
      section[data-testid="stSidebar"] {background:#000000; border-right:1px solid #1c1c1e;}
      section[data-testid="stSidebar"] > div {padding:1.5rem .85rem;}
      .sidebar-brand {padding:.4rem .65rem 1.35rem; font-size:1.05rem; font-weight:700; letter-spacing:-.02em; color:#ffffff;}
      .sidebar-kicker {color:#8e8e93; font-size:.68rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; margin:.5rem .65rem .35rem;}
      section[data-testid="stSidebar"] .stButton {margin:.14rem 0;}
      section[data-testid="stSidebar"] .stButton button {justify-content:flex-start; background:transparent; color:#8e8e93; border:1px solid transparent; padding:.55rem .7rem;}
      section[data-testid="stSidebar"] .stButton button[kind="primary"] {background:#1c1c1e; border-color:#2c2c2e; color:#ffffff;}
      .stButton button {width:100%; border:1px solid #2c2c2e; border-radius:12px; background:#1c1c1e; color:#ffffff; font-weight:600; min-height:2.7rem; transition:all .18s ease;}
      .stButton button:hover {border-color:#3a3a3c; background:#2c2c2e; transform:translateY(-1px);}
      .stButton button[kind="primary"] {background:#ffffff; color:#000000; border-color:#ffffff;}
      .stButton button[kind="primary"]:hover {background:#e5e5ea;}
      .product-hero {padding:1.7rem 0 2.15rem; border-bottom:1px solid #1c1c1e; margin-bottom:1.35rem;}
      .product-eyebrow {font-size:.72rem; color:#8e8e93; font-weight:600; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.65rem;}
      .product-title {font-size:clamp(2rem,4vw,3.2rem); line-height:1.06; letter-spacing:-.055em; font-weight:700; color:#ffffff;}
      .product-subtitle {margin-top:.7rem; color:#8e8e93; font-size:1rem; max-width:520px; line-height:1.55;}
      .module-card {height:190px; background:#1c1c1e; border:1px solid #2c2c2e; border-radius:18px; padding:1.25rem; margin-bottom:.55rem; transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease; position:relative; overflow:hidden;}
      .module-card:hover {transform:translateY(-3px); border-color:#3a3a3c; box-shadow:0 14px 30px rgba(0,0,0,.22);}
      .module-icon {width:38px;height:38px;display:flex;align-items:center;justify-content:center;border-radius:11px;background:#2c2c2e;font-size:1.25rem;margin-bottom:1rem; color:#ffffff;}
      .module-title {font-size:1rem;font-weight:700;color:#ffffff;letter-spacing:-.015em;}
      .module-desc {font-size:.82rem;color:#8e8e93;line-height:1.45;margin-top:.32rem;}
      .module-meta {position:absolute;bottom:1.05rem;left:1.25rem;font-size:.69rem;letter-spacing:.07em;text-transform:uppercase;color:#8e8e93;font-weight:700;}
      .module-action .stButton button {margin-top:-3.25rem; height:168px; opacity:0; position:relative; z-index:2;}
      .page-hero {padding:.25rem 0 1.75rem; border-bottom:1px solid #1c1c1e; margin-bottom:1.8rem;}
      .page-icon {font-size:1.5rem; margin-bottom:.75rem;}.page-title {font-size:2rem;font-weight:700;letter-spacing:-.045em;color:#ffffff;}.page-desc {color:#8e8e93;margin-top:.45rem;font-size:.98rem;}
      .platform-brief {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:18px;padding:1.2rem 1.3rem;margin:0 0 1.5rem;}.brief-label {color:#8e8e93;text-transform:uppercase;letter-spacing:.09em;font-size:.68rem;font-weight:700;margin-bottom:.4rem;}.brief-text {color:#ffffff;font-size:.91rem;line-height:1.5;}.brief-chip {display:inline-block;margin:.8rem .45rem 0 0;padding:.25rem .55rem;border:1px solid #3a3a3c;border-radius:999px;color:#8e8e93;font-size:.7rem;}
      .metric-card {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:14px;padding:1rem 1.1rem;margin:.45rem 0 1.7rem;}.metric-value {font-size:1.35rem;line-height:1;font-weight:700;color:#ffffff;letter-spacing:-.04em;}.metric-label {font-size:.72rem;color:#8e8e93;margin-top:.35rem;}
      .upload-shell {max-width:670px;margin:0 auto 1.5rem;background:#1c1c1e;border:1px solid #2c2c2e;border-radius:20px;padding:1.35rem;box-shadow:0 10px 25px rgba(0,0,0,.12);}
      .upload-heading {font-weight:700;font-size:1rem;color:#ffffff;margin:.1rem 0 .3rem;}.upload-copy {font-size:.82rem;color:#8e8e93;margin-bottom:1rem;}
      div[data-testid="stFileUploader"] {background:#000000;border:1px dashed #3a3a3c;border-radius:14px;padding:.35rem;}
      div[data-testid="stFileUploader"] section {padding:.8rem;}.stFileUploader label {display:none;}
      .empty-state {text-align:center;padding:1rem .75rem .3rem;}.empty-icon {font-size:2.35rem;margin-bottom:.55rem;}.empty-title {color:#ffffff;font-weight:600;}.empty-copy {font-size:.8rem;color:#8e8e93;margin-top:.25rem;}
      .media-card {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:16px;padding:1rem;margin:0 0 1.25rem;}.media-label {font-size:.74rem;font-weight:600;color:#8e8e93;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.65rem;}
      .result-card {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:16px;padding:1.2rem;min-height:150px;}.result-label {font-size:.72rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#8e8e93;}.result-value {font-size:1.55rem;font-weight:700;letter-spacing:-.04em;color:#ffffff;margin:.55rem 0 .75rem;}.result-confidence {font-size:.78rem;color:#8e8e93;}.confidence-track {height:6px;border-radius:9px;background:#2c2c2e;margin-top:.48rem;overflow:hidden;}.confidence-fill {height:100%;border-radius:inherit;background:#ffffff;}
      .warning-card {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:16px;padding:1.15rem 1.25rem;margin-bottom:1rem;}.warning-title {font-weight:700;color:#ffffff;margin-bottom:.25rem;}.warning-copy {font-size:.88rem;color:#8e8e93;}
      .status-card {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:16px;padding:1rem 1.15rem;margin:1rem 0;}.status-title {font-size:.73rem;text-transform:uppercase;letter-spacing:.09em;color:#8e8e93;font-weight:600;margin-bottom:.5rem;}.status-step {font-size:.88rem;color:#8e8e93;line-height:1.8;}.status-complete {color:#ffffff;}
      .model-info {background:#000000;border:1px solid #2c2c2e;border-radius:12px;padding:.9rem;text-align:center;}.model-label {font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:#8e8e93;}.model-value {font-size:.86rem;font-weight:600;color:#ffffff;margin-top:.25rem;}
      .app-footer {text-align:center;border-top:1px solid #1c1c1e;margin-top:3rem;padding:1.3rem 0 .2rem;color:#8e8e93;font-size:.76rem;}.inference {font-size:.78rem;color:#8e8e93;margin:.9rem 0;text-align:center;}
      div[data-testid="stExpander"] {background:#1c1c1e;border:1px solid #2c2c2e;border-radius:13px;}
      .stProgress > div > div > div {background:#ffffff;}
    </style>
    """, unsafe_allow_html=True)


def render_hero(icon, title, description):
    st.markdown(f'<div class="page-hero"><div class="page-icon">{icon}</div><div class="page-title">{title}</div><div class="page-desc">{description}</div></div>', unsafe_allow_html=True)


def upload_card(label, formats, icon="✦"):
    st.markdown(f'<div class="upload-shell"><div class="upload-heading">{icon} &nbsp;{label}</div><div class="upload-copy">Drag and drop a file here, or browse from your device. Supported: {formats}</div>', unsafe_allow_html=True)


def close_upload_card():
    st.markdown('</div>', unsafe_allow_html=True)


def render_empty_state(icon, message, formats=""):
    supported = f'<div class="empty-copy">Supported: {formats}</div>' if formats else ''
    st.markdown(f'<div class="empty-state"><div class="empty-icon">{icon}</div><div class="empty-title">{message}</div>{supported}</div>', unsafe_allow_html=True)


def render_result_card(label, value, confidence=None, accent_color="#a78bfa"):
    confidence_html = ""
    if confidence is not None:
        confidence_html = f'<div class="result-confidence">Confidence&nbsp; {confidence:.1f}%</div><div class="confidence-track"><div class="confidence-fill" style="width:{min(confidence,100)}%;background:{accent_color}"></div></div>'
    st.markdown(f'<div class="result-card"><div class="result-label">{label}</div><div class="result-value">{value}</div>{confidence_html}</div>', unsafe_allow_html=True)


def render_warning_card(title, message):
    st.markdown(f'<div class="warning-card"><div class="warning-title">⚠ {title}</div><div class="warning-copy">{message}</div></div>', unsafe_allow_html=True)


def render_progress_steps(steps):
    """Render lightweight visible progress feedback; steps is [(text, done), ...]."""
    rows = ''.join(f'<div class="status-step {"status-complete" if done else ""}">{"✓" if done else "•"} &nbsp;{text}</div>' for text, done in steps)
    st.markdown(f'<div class="status-card"><div class="status-title">Analysis progress</div>{rows}</div>', unsafe_allow_html=True)


def render_media_card(label):
    st.markdown(f'<div class="media-card"><div class="media-label">{label}</div>', unsafe_allow_html=True)


def close_media_card():
    st.markdown('</div>', unsafe_allow_html=True)


def render_model_info(models):
    cols = st.columns(len(models))
    for col, info in zip(cols, models):
        with col:
            st.markdown(f'<div class="model-info"><div class="model-label">{info["label"]}</div><div class="model-value">{info["value"]}</div></div>', unsafe_allow_html=True)


def render_inference_time(seconds):
    st.markdown(f'<div class="inference">⚡ Analysis completed in {seconds:.2f}s</div>', unsafe_allow_html=True)


def render_tech_stack(techs):
    st.markdown('<div style="text-align:center;color:#798395;font-size:.78rem">' + ' &nbsp;•&nbsp; '.join(techs) + '</div>', unsafe_allow_html=True)


def render_footer():
    st.markdown('<div class="app-footer">Built with YOLOv8 &nbsp;•&nbsp; TensorFlow &nbsp;•&nbsp; XGBoost &nbsp;•&nbsp; Streamlit</div>', unsafe_allow_html=True)


def render_module_card(icon, title, description, accent):
    st.markdown(f'<div class="module-card" style="--accent-soft:{accent}20"><div class="module-icon">{icon}</div><div class="module-title">{title}</div><div class="module-desc">{description}</div><div class="module-meta">Ready for analysis &nbsp;→</div></div>', unsafe_allow_html=True)


def render_platform_metrics(metrics):
    """Render compact, high-value project facts without duplicating page layout."""
    cols = st.columns(len(metrics), gap="small")
    for col, (value, label) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)


def render_workflow_summary(summary, chips):
    """Present a module's purpose, input/output and model stack as a concise product brief."""
    chip_html = ''.join(f'<span class="brief-chip">{chip}</span>' for chip in chips)
    st.markdown(f'<div class="platform-brief"><div class="brief-label">Analysis workspace</div><div class="brief-text">{summary}</div>{chip_html}</div>', unsafe_allow_html=True)
