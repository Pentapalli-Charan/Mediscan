---
title: MediScan — Dermatoscopic Skin Lesion Classifier
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Educational AI skin lesion classifier with Grad-CAM interpretability
---

# MediScan — Skin Lesion Classification Prototype

> ⚠️ **DISCLAIMER:** This application is an **educational and research prototype** for automated dermatoscopic image classification. It is **not a medical diagnostic tool** and must **never be used for clinical decision-making or patient triage**. Predictions may be incorrect and performance varies across populations, devices, and illumination settings.

## Model Overview
- **Backbone Architecture:** EfficientNet-B0 (Pretrained on ImageNet-1K, frozen feature extractor)
- **Dataset:** HAM10000 (10,015 dermatoscopic images across 7 classes)
- **Input Resolution:** 224 × 224 RGB (ImageNet normalized)
- **Explainability:** Grad-CAM visual activation targeting `model.features[8]`
- **Locked Test Accuracy:** 75.79% Top-1, 89.55% Top-2
- **Macro ROC-AUC:** 0.9211

## 7 Diagnostic Categories
1. **AKIEC:** Actinic keratoses / Intraepithelial carcinoma
2. **BCC:** Basal cell carcinoma
3. **BKL:** Benign keratosis-like lesions
4. **DF:** Dermatofibroma
5. **MEL:** Melanoma
6. **NV:** Melanocytic nevi
7. **VASC:** Vascular lesions

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
