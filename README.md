---
title: MediScan
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: AI skin lesion classifier with Grad-CAM explainability
---

# MediScan

A deep learning system that classifies dermatoscopic skin lesion images into 7 categories using EfficientNet-B0, with Grad-CAM visual explanations showing which parts of the image the model focused on.

> **⚠️ Not a medical tool.** This is a research/educational prototype. It has not been clinically validated and must not be used for diagnosis or treatment decisions. Predictions can be wrong.

---

## What it does

- Classifies dermoscopy images into 7 HAM10000 skin lesion categories
- Shows model confidence and full probability distribution across all classes
- Generates Grad-CAM heatmaps so you can see where the model is "looking"
- Accepts image uploads or lets you try pre-loaded sample images
- Exports predictions as downloadable JSON

## The 7 categories

| Code | Category |
|------|----------|
| akiec | Actinic keratoses / Intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

---

## Model

**Architecture:** EfficientNet-B0 with a frozen ImageNet backbone and a trainable classification head (Linear 1280 → 7).

Only 8,967 out of 4 million parameters are actually trained — the rest come from ImageNet pretraining.

**Training:** Adam optimizer, lr=1e-3, batch size 32, cross-entropy loss, trained for 15 epochs with ReduceLROnPlateau scheduling and early stopping. Best checkpoint saved at epoch 14.

**Dataset:** [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — 10,015 dermatoscopic images across 7 categories, collected at the Medical University of Vienna and Cliff Rosendahl Clinic in Queensland.

Data splits are done at the *lesion level* (not image level) to prevent leakage — same lesion never appears in both train and test.

## Results

Evaluated on 1,512 held-out test images:

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 75.79% |
| Top-2 Accuracy | 89.55% |
| Macro F1 | 0.5157 |
| Macro ROC-AUC | 0.9211 |

Per-class breakdown:

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| akiec | 0.60 | 0.44 | 0.51 | 55 |
| bcc | 0.54 | 0.55 | 0.55 | 71 |
| bkl | 0.52 | 0.43 | 0.47 | 167 |
| df | 0.36 | 0.20 | 0.26 | 20 |
| mel | 0.48 | 0.40 | 0.44 | 168 |
| nv | 0.85 | 0.92 | 0.88 | 1,007 |
| vasc | 0.67 | 0.42 | 0.51 | 24 |

The model is heavily biased toward `nv` (melanocytic nevi) since that class makes up 67% of the dataset. Minority classes like `df` and `vasc` have weak recall.

---

## Setup

```bash
git clone https://github.com/Pentapalli-Charan/Mediscan.git
cd Mediscan

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

You'll also need the HAM10000 dataset if you want to use the dataset sample browser. Download it from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) and place the images in `data/raw/`.

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. You can upload your own dermoscopy images or pick from the built-in samples.

---

## Project structure

```
MediScan/
├── app.py                  # Entrypoint (run this with streamlit)
├── app/
│   ├── app.py              # Streamlit UI
│   ├── utils.py            # Model loading, preprocessing, inference, Grad-CAM
│   └── assets/samples/     # 7 demo images (one per class)
├── src/
│   ├── data/               # Dataset loading and preprocessing
│   ├── models/             # EfficientNet-B0 and ResNet-50 definitions
│   ├── training/           # Training loop, checkpointing, MLflow tracking
│   ├── evaluation/         # Metrics, inference, analysis
│   ├── explainability/     # Grad-CAM implementation
│   └── utils/              # Device and seed utilities
├── models/
│   └── checkpoints/
│       └── efficientnet_b0_best.pth   # Trained weights
├── config/
│   └── config.yaml         # All hyperparameters and pipeline config
├── requirements.txt
└── .gitignore
```

---

## Known limitations

- **Dataset bias:** Trained exclusively on HAM10000 — mostly light-skinned European patients. Performance on darker skin tones is unknown and likely worse.
- **Class imbalance:** 67% of training data is melanocytic nevi, creating a strong bias toward predicting benign.
- **False negatives matter:** In testing, ~42% of melanomas were misclassified as benign nevi. A "benign" prediction from this model means nothing clinically.
- **Domain shift:** Trained on dermatoscope images only. Phone camera photos will perform significantly worse.
- **Closed world:** Can only predict the 7 categories it was trained on. Rare conditions or non-skin-lesion images will still get classified into one of the 7.
- **Confidence ≠ certainty:** The model can be highly confident and still wrong. 13 melanomas were classified as nevi with >80% confidence.

## License

CC BY-NC-SA 4.0 (following HAM10000 dataset license).
