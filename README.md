<div align="center">
  <h1>🧠 MultiVision AI Suite</h1>
  <p><strong>Advanced Computer Vision & AI Workspace</strong></p>
  <p><i>A capstone internship project demonstrating real-time machine learning deployment.</i></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  </p>
</div>

---

## 📖 Overview

This repository contains my final project submission for the **Elevance Skills Training & Internship Program**. 

The **MultiVision AI Suite** is a unified, offline-ready desktop application that integrates 7 distinct Machine Learning and Computer Vision models into a single, cohesive, Apple-inspired interface. It demonstrates end-to-end expertise in model inference, real-time video processing, and modern UI/UX design.

### 🎯 Project Separation (Grading Criteria)

To fulfill the internship requirements, this repository builds upon the foundational training project and expands it with advanced internship modules:

**1. Training Project (Base Foundation)**
*   **Facial Emotion Detection:** Real-time CNN-based facial expression analysis.
*   **Car Colour Detection:** Vehicle tracking and color classification.
*   **Animal Detection:** YOLOv8 object detection tailored for animals, with carnivorous highlighting.

**2. Internship Expansion (Extra Features & Models)**
*   **Sign Language Detection (HD 720p):** A hybrid MediaPipe + Custom MLP architecture with a live sequence builder, majority-voting debounce logic, and Text-to-Speech (gTTS) integration. Includes transcript exporting.
*   **Emotion Detection (Voice):** Audio analysis for detecting sentiment from vocal files.
*   **Drowsiness Detection:** Real-time eye aspect ratio tracking to prevent driver fatigue.
*   **Nationality & Appearance:** Multi-attribute prediction (age, emotion, nationality) from faces.
*   **Unified UI Engine:** A custom-built, Apple-inspired monochromatic Streamlit theme with dynamic routing, seamless CSS injection, and state preservation.

---

## 🚀 Features

*   **100% Offline Capable:** Core computer vision models run locally without requiring internet access.
*   **Real-Time Processing:** Supports live webcam feeds (OpenCV & WebRTC) and recorded video files.
*   **Heads-Up Display (HUD):** Custom OpenCV rendering for crisp, non-distorted bounding boxes and UI overlays.
*   **Premium UI:** Apple-inspired monochromatic design with custom CSS components.

---

## 🛠️ Tech Stack

*   **Core:** Python 3
*   **Computer Vision:** OpenCV (cv2), MediaPipe
*   **Machine Learning:** TensorFlow, Keras, Scikit-Learn, YOLOv8 (Ultralytics)
*   **Frontend:** Streamlit, Custom HTML/CSS
*   **Audio/TTS:** gTTS (Google Text-to-Speech)

---

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/multivision-ai.git
   cd multivision-ai
   ```

2. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## 👨‍💻 About the Developer

**Swayam**  
*Machine Learning Intern @ Elevance Skills*  

This project represents the culmination of a rigorous training and internship track focused on practical AI deployment, computer vision pipelines, and full-stack integration. 

* **Domain:** AI & Computer Vision
* **Contact:** [Your Email]
* **GitHub:** [Your Profile Link]

---
<div align="center">
  <i>Submitted for the Elevance Skills Internship Program • 2026</i>
</div>
