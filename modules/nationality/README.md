# Face Attribute Predictor: Multi-task Learning vs DeepFace Pivot

## Overview

This module predicts Appearance-Based Ethnicity, Age, Emotion, and Dress Color from a facial image, matching the strict conditional routing requirements of the project brief.

> **Important Limitation:** Nationality is a legal/citizenship status. It is not encoded in a face, so no vision model can genuinely predict it. This system predicts *appearance-based ethnicity/region* using standard computer vision datasets as a proxy. It should not be used to make real decisions about individuals.

## Methodology & Engineering Decisions

During the development of this project, I adopted an iterative approach to diagnose and resolve a severe class imbalance issue.

### V1 Diagnostic Baseline (Custom Multi-Task MobileNetV2)
Initially, I built a custom multi-task model to predict both Age and Ethnicity simultaneously.
- **Backbone:** MobileNetV2 (pretrained on ImageNet), frozen for Phase 1 and fine-tuned for Phase 2.
- **Dataset:** UTKFace, which provides race labels: `White`, `Black`, `Asian`, `Indian`, `Others`.
- **The Problem:** UTKFace suffers from severe class imbalance (`White` makes up 42.5% of the dataset, while `Indian` is only ~16%). Because of this, the custom MobileNetV2 model suffered from **Mode Collapse**—when tested on a random 100-image sample of the dataset, it predicted `White` **54%** of the time, consistently misclassifying `Indian` and `Asian` faces as `White`.

The custom `DiagnosticBaselineBundle_v1` is preserved in the repository (`inference_utils.py`) as evidence of this training work and the diagnosed mode collapse.

### V2 Production Pivot (DeepFace Framework)
Rather than aggressively downsampling the UTKFace dataset or applying complex sample weights that could compromise the Age regression head, I made a justified engineering decision to pivot to the **DeepFace** framework for the production inference engine.

- **Why DeepFace?** DeepFace provides robust, pretrained models for Race, Age, and Emotion that are already immune to our dataset's specific class imbalance. 
- **Integration:** The `ProductionDeepFaceBundle_v2` wraps the DeepFace analysis pipeline and perfectly maps its race outputs to our assignment's conditional UI routing (Indian / United States / African / Other).
- **Dress Color:** The dress color extraction is kept as classical CV (OpenCV torso cropping + KMeans clustering) to ensure efficiency and demonstrate custom engineering.

## Side-by-Side Comparison

To prove the mode collapse and justify the pivot, here are random examples of the Baseline (V1) failing on minority classes compared to DeepFace (V2):

| Image | Ground Truth | V1 Baseline Prediction | V2 DeepFace Prediction |
|-------|--------------|-------------------------|-------------------------|
| `22_1_3_20170117154551789.jpg` | Indian | Asian (Collapse) | Indian |
| `16_0_0_20170110231526097.jpg` | White | Asian (Confusion) | White |
| `output.jpg` | Black | White (Collapse) | Black |

## Architecture

```
Image upload (Streamlit)
        │
        ▼
DeepFace.analyze()
        │
        ├──► age, dominant_race (+ confidence)
        └──► dominant_emotion (+ confidence)
        │
        ▼
map_to_category(dominant_race) ──► Indian / United States / African / Other
        │
        ├── dress color: OpenCV crop of torso region below face box
        │                 + KMeans(k=3) to find the dominant clothing color
        │                 (classical CV, not learned — no dataset needed)
        ▼
Conditional display per category:
  Indian          → age + dress color + emotion
  United States   → age + emotion
  African         → dress color + emotion
  Other           → category + emotion
```

## Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the MultiVision app from the root folder
cd ../../
streamlit run app.py
```
