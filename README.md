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

# MediScan

> ⚠️ **IMPORTANT DISCLAIMER:** MediScan is an **educational and research prototype** for automated dermatoscopic image classification. It is **not a medical diagnostic tool** and must **never be used for clinical decision-making, patient triage, or direct care**. Predictions may be incorrect and performance varies across populations, devices, acquisition settings, and illumination environments.

---

## Project Overview

**MediScan** is an end-to-end deep learning medical image analysis system engineered as a 15-day capstone project. It classifies dermatoscopic skin lesion images into 7 diagnostic categories from the widely benchmarked HAM10000 dataset using transfer learning with **EfficientNet-B0**, provides visual model interpretability via **Grad-CAM** (`model.features[8]`), logs experiments reproducibly via **MLflow**, and delivers an interactive user interface via **Streamlit** deployable on **Hugging Face Spaces**.

| Milestone | Deliverable | Status |
|:---|:---|:---:|
| **Day 1** | Project Setup, Exploratory Data Analysis & Integrity Verification | ✅ Completed |
| **Day 2** | Data Preprocessing, Deduplication & Patient-Aware Split | ✅ Completed |
| **Day 3** | Medically Grounded Data Augmentation Pipeline | ✅ Completed |
| **Day 4** | PyTorch Dataset & High-Throughput DataLoader Pipeline | ✅ Completed |
| **Day 5** | EfficientNet-B0 Transfer Learning Architecture Setup | ✅ Completed |
| **Day 6** | Training Engine, Validation & Checkpoint Management | ✅ Completed |
| **Day 7** | Baseline Model Training Execution (15 Epochs) | ✅ Completed |
| **Day 8** | Controlled Hyperparameter Tuning & Learning Rate Sweep | ✅ Completed |
| **Day 9** | Comprehensive Model Evaluation & Held-Out Test Audit | ✅ Completed |
| **Day 10** | Grad-CAM Visual Explainability & Weight Immutability | ✅ Completed |
| **Day 11** | Architectural Comparison: EfficientNet-B0 vs ResNet-50 | ✅ Completed |
| **Day 12** | Interactive Streamlit Web Application Development | ✅ Completed |
| **Day 13** | MLflow Experiment Tracking, Artifact Logging & Benchmarking | ✅ Completed |
| **Day 14** | Edge-Case, Subgroup Bias, Calibration & Robustness Analysis | ✅ Completed |
| **Day 15** | Finalization, Deployment Packaging, Model Card & Sign-Off | ✅ Completed |

---

## Problem Statement

Skin cancer is among the most prevalent human malignancies worldwide, with melanoma representing the deadliest form. While early detection substantially improves clinical prognosis, dermatoscopic evaluation requires specialized training and exhibits inter-observer variability. Developing computer-aided algorithmic prototypes that categorize lesions and provide visual evidence maps offers valuable pedagogical and research tools, provided that data leakage, class imbalance, calibration errors, and clinical boundaries are rigorously understood.

---

## Objectives

1. **Patient-Aware Integrity**: Construct stratified, lesion-grouped dataset partitions preventing data leakage between train, validation, and test splits.
2. **Transfer Learning**: Adapt pretrained convolutional architectures (EfficientNet-B0 and ResNet-50) using frozen feature extractors and customized heads.
3. **Rigorous Evaluation**: Audit model performance across overall accuracy, top-2 accuracy, per-class sensitivity/precision/F1, and ROC/PR curves.
4. **Visual Interpretability**: Integrate Grad-CAM explanations targeting penultimate convolutional activations (`model.features[8]`).
5. **Experiment Reproducibility**: Record hyperparameters, step-wise training curves, artifacts, and metrics in a local MLflow file store.
6. **Algorithmic Auditing**: Quantify demographic disparities, expected calibration error (ECE), synthetic perturbation sensitivities, and high-confidence failure modes.
7. **Production Deployment**: Package a lightweight, secure web application with curated demonstration samples for Hugging Face Spaces.

---

## Dataset

**HAM10000 ("Human Against Machine with 10000 training images")**
- **Total Images:** 10,015 dermatoscopic photographs
- **Unique Lesions:** 7,470 distinct lesions (many lesions photographed multiple times)
- **Clinical Centers:** Department of Dermatology, Medical University of Vienna (Austria) & Cliff Rosendahl Clinic (Queensland, Australia)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- **Diagnostic Categories (7 Classes):**
  1. `akiec`: Actinic keratoses / Intraepithelial carcinoma (Bowen's disease)
  2. `bcc`: Basal cell carcinoma
  3. `bkl`: Benign keratosis-like lesions (solar lentigines / seborrheic keratoses)
  4. `df`: Dermatofibroma
  5. `mel`: Melanoma
  6. `nv`: Melanocytic nevi
  7. `vasc`: Vascular lesions (angiomas / pyogenic granulomas)

---

## Data Leakage Prevention

A critical risk in dermatoscopic machine learning is **patient/lesion data leakage**. In HAM10000, 10,015 images map to 7,470 unique `lesion_id` identifiers; identical lesions often appear under multiple magnifications or time points. Random splitting would cause identical lesions to appear in both training and test sets, artificially inflating test scores.

- **Mitigation Strategy:** Stratified Group Splitting (`GroupKFold` / lesion-level stratification).
- **Enforcement:** All images belonging to a specific `lesion_id` are strictly assigned to either Train, Validation, or Test.
- **Verification:**
  - Train: 6,982 images (5,263 unique lesions)
  - Validation: 1,521 images (1,086 unique lesions)
  - Test: 1,512 images (1,121 unique lesions)
  - **Lesion Overlap (Train ∩ Val, Train ∩ Test, Val ∩ Test): Exactly 0 lesions (0.0%).**

---

## Preprocessing

All images are preprocessed through a deterministic, standardized pipeline matching ImageNet normalization standards:
1. **Source Loading:** RGB conversion from raw JPEG format.
2. **Standard Resolution:** Resized to $224 \times 224$ pixels using bilinear interpolation.
3. **Tensor Conversion:** Conversion from $[0, 255]$ uint8 to $[0.0, 1.0]$ float32 tensor $[3, 224, 224]$.
4. **ImageNet Normalization:** Channel-wise mean subtraction and standard deviation scaling:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## Data Augmentation

To prevent overfitting on the majority class and enhance rotational and illumination invariance, training images undergo Albumentations transforms:
- **Spatial Transforms:** Random Horizontal Flip ($p=0.5$), Random Vertical Flip ($p=0.5$), Random 90° Rotation ($p=0.5$), Affine Shift/Scale/Rotate ($\pm 15\%$, $p=0.3$).
- **Color & Lighting Transforms:** ColorJitter (brightness $\pm 0.15$, contrast $\pm 0.15$, saturation $\pm 0.15$, hue $\pm 0.05$, $p=0.4$), CLAHE ($p=0.2$).
- **Medical Validation:** Transforms are designed to simulate realistic clinical dermoscopy variations while preserving diagnostic morphology.

---

## Model Architecture

The primary deployment model is **EfficientNet-B0**:
- **Backbone:** ImageNet-1K pretrained convolutional feature extractor.
- **Feature Layer:** Output of `features[8]` ($1280$ channels, $7 \times 7$ feature maps).
- **Pooling & Head:**
  - `AdaptiveAvgPool2d(output_size=1)`
  - `Dropout(p=0.20)`
  - `Linear(in_features=1280, out_features=7)`
- **Parameter Count:**
  - Total Parameters: **4,016,515**
  - Trainable Parameters (Head only): **8,967** (0.22%)
  - Frozen Backbone Parameters: **4,007,548** (99.78%)

---

## Training Strategy

- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$)
- **Baseline Learning Rate:** $\eta = 1.0 \times 10^{-3}$
- **Batch Size:** 32
- **Loss Function:** Cross-Entropy Loss with Softmax
- **Learning Rate Scheduler:** `ReduceLROnPlateau` (mode='min', factor=0.5, patience=2, min_lr=1e-6)
- **Early Stopping:** Patience of 3 epochs monitoring validation loss
- **Epoch Budget:** 15 epochs (Checkpoint saved at Epoch 14 with best validation loss)

---

## Hyperparameter Experiments

In Day 8 and Day 13, hyperparameter configurations were benchmarked and logged to MLflow:

| Experiment Name | Learning Rate | Dropout | Epochs | Best Val Loss | Val Acc (%) | Early Stopped | Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`EfficientNet-B0-Baseline`** | **0.0010** | **0.20** | **15** | **0.6422** | **77.38%** | No | **Canonical Deployment Model** |
| `EfficientNet-B0-LR-5e-4` | 0.0005 | 0.20 | 5 | 0.7283 | 75.21% | Yes (Patience) | Candidate Tuning Run |
| `ResNet50-Benchmark` | 0.0010 | 0.20 | 1 | 0.7646 | 72.52% | No | Architecture Comparison |

---

## Evaluation Results

Evaluation on the locked, held-out test split (1,512 samples, MD5: `6a5ae1b65c25d504f78bc1da2361d82c`):

### Overall Test Metrics
| Metric | Value |
|:---|:---:|
| **Top-1 Accuracy** | **75.79%** (1,146 / 1,512 correct) |
| **Top-2 Accuracy** | **89.55%** (1,354 / 1,512 correct) |
| **Macro Precision** | **0.5744** |
| **Macro Recall (Sensitivity)** | **0.4787** |
| **Macro F1-Score** | **0.5157** |
| **Weighted Precision** | **0.7382** |
| **Weighted Recall** | **0.7579** |
| **Weighted F1-Score** | **0.7448** |
| **Macro ROC-AUC** | **0.9211** |
| **Weighted ROC-AUC** | **0.9145** |
| **Macro PR-AUC** | **0.5646** |
| **Test Cross-Entropy Loss** | **0.6638** |

### Per-Class Test Breakdown
| Class | Support | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`akiec`** | 55 | 0.6000 | 0.4364 | 0.5053 | 0.9315 | 0.4996 |
| **`bcc`** | 71 | 0.5417 | 0.5493 | 0.5455 | 0.9396 | 0.5704 |
| **`bkl`** | 167 | 0.5221 | 0.4251 | 0.4686 | 0.8726 | 0.5097 |
| **`df`** | 20 | 0.3636 | 0.2000 | 0.2581 | 0.9612 | 0.3455 |
| **`mel`** | 168 | 0.4786 | **0.3988** | 0.4351 | 0.8766 | 0.5046 |
| **`nv`** | 1,007 | 0.8479 | **0.9245** | 0.8846 | 0.9235 | 0.9605 |
| **`vasc`** | 24 | 0.6667 | 0.4167 | 0.5128 | 0.9430 | 0.5619 |

---

## Grad-CAM Explainability

To provide visual transparency for model predictions, Grad-CAM backpropagates class-specific gradients to compute a weighted activation map across the penultimate convolutional feature map:
- **Target Layer:** Strictly `model.features[8]` (Conv2dNormActivation with 1280 channels).
- **Safety Guarantee:** Backpropagation is isolated; model weights are snapshot and verified bitwise identical before and after explainability calls.
- **Visual Presentation:** Colored Jet heatmaps superimposed on original lesion inputs with user-adjustable blending opacity $\alpha$.

---

## ResNet-50 Comparison

In Day 11, EfficientNet-B0 was benchmarked against ResNet-50 under identical training conditions:
- **Parameters:** EfficientNet-B0 (4.02M) vs. ResNet-50 (23.52M) — EfficientNet is **5.85× lighter**.
- **Validation Loss:** EfficientNet-B0 (**0.6422**) vs. ResNet-50 (0.7646) — EfficientNet achieved **16.0% lower loss**.
- **Validation Accuracy:** EfficientNet-B0 (**77.38%**) vs. ResNet-50 (72.52%) — EfficientNet outperformed by **+4.86%**.
- **Inference Latency:** EfficientNet-B0 executes in **~25 ms** on CPU vs. ~58 ms for ResNet-50.

---

## MLflow Experiment Tracking

Full experiment tracking was integrated on Day 13 using a local file-based MLflow backend:
- **Experiment Name:** `MediScan` (ID: `944157940989004540`)
- **Tracked Runs:** 3 genuine historical runs (`EfficientNet-B0-Baseline`, `EfficientNet-B0-LR-5e-4`, `ResNet50-Benchmark`).
- **Logged Parameters:** 30 parameters per run (architecture, hyperparameters, commit hash, dataset sizes).
- **Logged Artifacts:** Step-wise epoch history CSVs, comparison charts, and classification reports.
- **UI Integration:** The Streamlit app dynamically displays historical experiment summaries from MLflow with offline fallback.

---

## Bias, Robustness & Limitations

Audited extensively during Day 14 across 18 analytical phases:
1. **Class Imbalance:** Extreme **58.3:1** imbalance between `nv` (67.0%) and `df` (1.15%) biases the network toward predicting benign nevi under diagnostic uncertainty.
2. **Clinical False Negatives:** **41.67% of test melanomas** (70 cases) were misclassified as benign nevi (`nv`).
3. **Overconfident Errors:** 45 test errors exhibited $\ge 80\%$ softmax confidence (including 13 melanomas classified as nevus with $>80\%$ confidence).
4. **Calibration:** Overall Expected Calibration Error (ECE) is **0.0350**; high confidence bin ($[0.9, 1.0]$) is exceptionally well calibrated (97.3% conf vs 97.8% acc), but mid-range bins ($[0.4, 0.6]$) suffer moderate overconfidence.
5. **Synthetic Robustness:** Gaussian blur induced the largest fragility (consistency dropped to **76.0%**, class flip rate **24.0%**), demonstrating critical reliance on high-frequency pigment networks.
6. **Demographic Disparities:** Accuracy in geriatric patients ($\ge 60$ yrs) dropped to **61.04%** (vs 86.17% in $<40$ yrs). Facial lesions had the lowest anatomical site accuracy (**44.94%**).

---

## Streamlit Web Application

The interactive web application (`app/app.py` / `app.py`) provides:
- **Dual Input Modes:** User file upload (JPEG, PNG) or selection from curated HAM10000 demonstration samples.
- **Deterministic Pipeline:** Standardized 224×224 preprocessing with ImageNet normalization.
- **Model Predictions:** Displays top predicted category, model confidence, and full 7-class probability distributions.
- **Grad-CAM Visualizer:** Interactive target class selection and adjustable overlay blending ($\alpha$).
- **Structured Export:** Downloadable JSON report capturing prediction metadata and probability vectors.
- **Safety Disclaimers:** Non-diagnostic educational taxonomy and prominent limitation notices.

---

## Deployment

The application is packaged for deployment on **Hugging Face Spaces**:
- **SDK:** Streamlit 1.54.0
- **App Entrypoint:** `app.py` (delegates to `app/app.py`)
- **Curated Samples:** 7 lightweight demonstration samples in `app/assets/samples/` (1.9 MB) avoiding the multi-GB raw dataset.
- **Total Deployment Size:** **18.17 MB** (including model weights of 15.68 MB).

To deploy to Hugging Face Spaces:
```bash
# 1. Create a new Streamlit Space on huggingface.co/new-space
# 2. Clone the empty space repository
git clone https://huggingface.co/spaces/<username>/mediscan-demo
cd mediscan-demo

# 3. Copy the prepared deployment bundle
cp -r "path/to/MediScan Project/deployment/hf_space/*" .

# 4. Push to Hugging Face
git add .
git commit -m "Deploy MediScan Streamlit Application"
git push
```

---

## Installation

### Prerequisites
- Python 3.11+ (verified up to Python 3.14)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/Pentapalli-Charan/Mediscan.git
cd Mediscan

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux / macOS

# Install development dependencies
pip install -r requirements.txt
```

---

## Running Locally

To run the interactive Streamlit application locally:
```bash
streamlit run app.py
```
Or directly:
```bash
streamlit run app/app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## MLflow Usage

To inspect historical training runs, hyperparameters, metrics, and logged artifacts:
```bash
mlflow ui --backend-store-uri mlruns/
```
Open `http://localhost:5000` to access the MLflow dashboard.

---

## Project Structure

```
MediScan Project/
├── README.md                          # Comprehensive project documentation & Space metadata
├── MODEL_CARD.md                      # Formal 24-section model card
├── requirements.txt                   # Full development & training dependencies
├── requirements-deployment.txt        # Lightweight runtime deployment dependencies
├── .gitignore                         # Git ignore specifications
├── app.py                             # Root application entrypoint (Hugging Face Spaces)
├── config/
│   └── config.yaml                    # Centralized experiment & pipeline configuration
├── app/
│   ├── app.py                         # Streamlit interactive application interface
│   ├── utils.py                       # App utilities: loading, preprocessing, inference, Grad-CAM
│   └── assets/
│       └── samples/                   # Curated demonstration images (1 per class, 1.9 MB total)
│           ├── samples_metadata.json  # Provenance and metadata for demo samples
│           └── *.jpg                  # 7 curated representative images (from training split)
├── models/
│   └── checkpoints/
│       └── efficientnet_b0_best.pth   # Canonical frozen weights (MD5: 7c1c6fcbe02e93f0ff8b4a20f62e29e3)
├── src/
│   ├── data/                          # Preprocessing, dataset, and DataLoader modules
│   ├── models/                        # EfficientNet-B0 and ResNet-50 architectures
│   ├── training/                      # Trainer engine, metrics logging, MLflow tracker
│   ├── evaluation/                    # Metrics calculation, inference, Day 14 analysis
│   └── explainability/                # Grad-CAM engine and visualization utilities
├── scripts/
│   ├── run_day14_analysis.py          # Day 14 edge-case & robustness runner
│   ├── prepare_hf_space.py            # Deployment bundle assembly script
│   ├── benchmark_smoke_test.py        # Local latency and smoke test benchmark
│   └── curate_deployment_samples.py   # Sample curation utility
├── reports/
│   ├── DAY1_DAY7_FULL_AUDIT.md        # Comprehensive Days 1–7 technical audit
│   ├── DAY8_COMPLETION_REPORT.md      # Day 8 hyperparameter tuning report
│   ├── DAY9_COMPLETION_REPORT.md      # Day 9 test evaluation report
│   ├── DAY10_COMPLETION_REPORT.md     # Day 10 Grad-CAM explainability report
│   ├── DAY11_COMPLETION_REPORT.md     # Day 11 ResNet-50 comparison report
│   ├── DAY12_COMPLETION_REPORT.md     # Day 12 Streamlit application report
│   ├── DAY13_COMPLETION_REPORT.md     # Day 13 MLflow experiment tracking report
│   ├── DAY14_COMPLETION_REPORT.md     # Day 14 bias & robustness completion report
│   ├── DAY15_COMPLETION_REPORT.md     # Day 15 finalization & deployment report
│   ├── FINAL_PROJECT_AUDIT.md         # Technical sign-off and milestone audit
│   ├── data/                          # CSV and JSON metrics exports across all milestones
│   └── figures/                       # Publication-grade figures & Grad-CAM heatmaps
├── tests/                             # Comprehensive automated test suite (336+ tests)
└── mlruns/                            # Local MLflow file tracking backend
```

---

## Ethical / Clinical Disclaimer

1. **Non-Diagnostic Tool:** MediScan is an exploratory academic prototype. It has not been validated in prospective clinical environments and possesses no regulatory clearance (FDA 510(k), CE mark, or equivalent).
2. **Never Supersede Clinical Judgment:** Healthcare professionals and individuals must never use predictions from this tool for triage, diagnosis, biopsy scheduling, or treatment alterations.
3. **False Reassurance Danger:** In empirical testing, 41.7% of biopsied melanomas were classified as benign nevi. A negative or benign prediction by this model must never be used to rule out malignancy.

---

## Known Limitations

- **Dataset Demographics:** HAM10000 images were captured in central Europe and Australia, representing almost exclusively light skin types (Fitzpatrick I–III). Efficacy on darker skin tones (Fitzpatrick IV–VI) is unverified and anticipated to be substantially lower.
- **Closed-World Assumption:** The model can only categorize into the 7 training diagnoses. Rare neoplasms (Merkel cell carcinoma, sebaceous carcinoma) or non-neoplastic inflammatory dermatoses cannot be detected.
- **Hardware & Image Specificity:** Trained exclusively on dermatoscope images; clinical photography from smartphones suffers severe domain shift.
- **Superficial Feature Attention:** Grad-CAM analysis demonstrates that foreign artifacts (skin markings, hair, ruler markings) can occasionally distort attention away from true lesion margins.
