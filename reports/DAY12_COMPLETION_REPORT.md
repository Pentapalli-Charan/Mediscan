# MediScan — Day 12 Completion Report: Web Application & Interactive User Interface (Streamlit)

**Author:** Pentapalli Charan  
**Date:** September 16, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 12 — Streamlit Web Application & Clinical UI  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | Streamlit 1.54.0 | OS: Windows 11  

---

## 1. Executive Summary & Objective

Day 12 delivered the production-ready interactive web application for **MediScan**, integrating our trained, validated, and locked **EfficientNet-B0** diagnostic model with real-time **Grad-CAM (Gradient-weighted Class Activation Mapping)** visual explainability.

The application allows clinical practitioners, dermatologists, and researchers to:
1. Upload dermoscopic lesions (JPG, JPEG, PNG) or select validated cases from the authenticated HAM10000 dataset gallery.
2. Execute automated 7-class lesion probability inference with instant confidence scores and risk stratifications.
3. Inspect neural attention via side-by-side Triplet views: **[1] Standardized Lesion Input**, **[2] Grad-CAM Neural Activation Heatmap**, and **[3] Blended Diagnostic Overlay** targeting `model.features[8]`.
4. Export structured diagnostic findings in machine-readable JSON format.
5. Review clinical decision support guidance with prominent regulatory/diagnostic safety disclaimers.

---

## 2. Architecture of the Web Application

The application is structured into modular, strictly audited components under `app/`:

```
app/
├── __init__.py           # Package marker for clean module imports
├── app.py                # Main interactive Streamlit application frontend
└── utils.py              # Audited helper functions (model caching, preprocessing, inference, Grad-CAM)
```

### Module Responsibilities:

- **`app/utils.py` (Backend Engine)**:
  - `load_cached_mediscan_model()`: Caches the PyTorch model across Streamlit reruns via `@st.cache_resource`, with safe headless fallback during CLI/pytest runs.
  - `validate_and_load_image()`: Validates image headers, sizes ($\le 25\text{ MB}$), empty streams, formats into 3-channel RGB, and handles Streamlit `UploadedFile` buffers.
  - `preprocess_image_for_inference()`: Applies the exact deterministic Day 2–4 test preprocessing ($224 \times 224$ bilinear resize, ImageNet normalization $\mu=[0.485, 0.456, 0.406], \sigma=[0.229, 0.224, 0.225]$).
  - `run_model_inference()`: Performs evaluation pass under `torch.no_grad()`, computing softmax probabilities across all 7 classes.
  - `compute_gradcam_explanation()`: Attaches forward and backward hooks to `model.features[8]`, computes spatial pooled gradients, synthesizes normalized activation maps, blends RGB overlays, cleans up hooks in `finally:`, and asserts bitwise model weight preservation.
  - `get_ham10000_sample_catalog()`: Traverses `data/raw/` to index ground-truth sample cases across all 7 diagnostic classes without loading full image arrays into memory.

- **`app/app.py` (Clinical UI & Workflow)**:
  - Built with modern medical aesthetics: slate container cards, risk-colored badges, dynamic confidence indicators, and responsive two-column layouts.
  - Sidebar system diagnostics: Displays model parameters ($4,016,515$), locked test performance metrics ($75.79\%$ Top-1, $89.55\%$ Top-2), compute hardware (`CPU`), and Grad-CAM blending slider ($\alpha \in [0.10, 0.90]$).

---

## 3. Clinical Risk Stratification & Diagnostic Classes

To translate statistical probabilities into clinical decision support, Day 12 establishes explicit risk tiers across the 7 HAM10000 classes:

| Class Code | Full Medical Diagnostic Name | Clinical Risk Level | Clinical Urgency & Recommendation |
|:---:|:---|:---:|:---|
| **`mel`** | Melanoma | **🔴 HIGH RISK (Malignant)** | Urgent Dermatologic Evaluation & Biopsy Recommended |
| **`bcc`** | Basal cell carcinoma | **🔴 HIGH RISK (Malignant)** | Specialist Consultation & Excision Assessment Recommended |
| **`akiec`** | Actinic keratoses / Intraepithelial carcinoma | **🟠 MODERATE RISK (Pre-Malignant)** | Dermatologic Examination & Treatment Planning Recommended |
| **`bkl`** | Benign keratosis-like lesions | **🟢 LOW RISK (Benign)** | Routine Clinical Monitoring |
| **`df`** | Dermatofibroma | **🟢 LOW RISK (Benign)** | Routine Clinical Monitoring |
| **`nv`** | Melanocytic nevi | **🟢 LOW RISK (Benign)** | Periodic ABCD Dermoscopy Monitoring |
| **`vasc`** | Vascular lesions | **🟢 LOW RISK (Benign)** | Routine Clinical Follow-up |

---

## 4. Visual Explainability (Grad-CAM) Integration

- **Target Layer**: Strictly **`model.features[8]`** (the final convolutional block of EfficientNet-B0).
- **Hook Lifecycle**: Guaranteed registration and removal inside a `try...finally:` block.
- **Weight Immutability**: Verified by taking parameter snapshots before gradient backpropagation and asserting bitwise equality afterwards (`verify_model_weights_unchanged`).
- **Triplet Diagnostic Display**:
  1. **Standardized Input**: $224 \times 224$ px original RGB dermoscopic image.
  2. **Neural Heatmap**: Jet colormap representing spatial activation intensity $[0.0, 1.0]$. Red/yellow identifies features that drove the classification; blue/cyan denotes uninformative zones or background.
  3. **Diagnostic Overlay**: Blended visualization using adjustable alpha blending ($\alpha = 0.45$ default).
- **Fault-Tolerant Decoupling**: If Grad-CAM calculation fails for any reason, the primary model prediction and probability distribution remain visible without disruption.

---

## 5. Model, Checkpoint & Pipeline Integrity Verification

Throughout Day 12, the underlying machine learning pipeline remained completely locked and untouched:

| Artifact / Asset | Verification Check | Status |
|:---|:---|:---:|
| **EfficientNet-B0 Checkpoint** | `models/checkpoints/efficientnet_b0_best.pth` MD5: `7c1c6fcbe02e93f0ff8b4a20f62e29e3` | **LOCKED & IDENTICAL** |
| **Test Split Partition** | `reports/data/split_test.csv` MD5: `6a5ae1b65c25d504f78bc1da2361d82c` | **STRICTLY UNTOUCHED** |
| **Day 9 Evaluation Results** | Top-1: `75.79%`, Top-2: `89.55%`, Macro-F1: `0.5157`, Loss: `0.6638` | **PRESERVED** |
| **Grad-CAM Target Layer** | `model.features[8]` (EfficientNet-B0 final conv block) | **CONFIRMED** |
| **Offline Operation** | Model instantiates with `pretrained=False`, loading local checkpoint | **VERIFIED (No CDN calls)** |

---

## 6. Automated Test Suite Results

### A. Targeted Day 12 Utilities Suite (`tests/test_app_utils.py`)
```bash
pytest tests/test_app_utils.py -v
```
**Result: 21 / 21 PASSED (100%) in 14.22s**
- `TestModelConstructionAndCheckpoint`: 6 passed
- `TestImageValidationAndLoading`: 7 passed
- `TestPreprocessing`: 2 passed
- `TestModelInference`: 2 passed
- `TestGradCAMExplanation`: 1 passed
- `TestStreamlitCachingAndCatalog`: 3 passed

### B. Targeted Day 12 Frontend App Suite (`tests/test_day12_app.py`)
```bash
pytest tests/test_day12_app.py -v
```
**Result: 5 / 5 PASSED (100%) in 6.39s**
- `test_app_script_exists_and_compiles`: PASSED
- `test_clinical_risk_levels_cover_all_classes`: PASSED
- `test_target_class_index_parsing`: PASSED
- `test_initial_render_headless`: PASSED
- `test_ham10000_sample_selection_and_inference`: PASSED

### C. Full Project Regression Suite (Days 2–12)
```bash
pytest tests/ -q
```
**Result: 277 PASSED, 1 SKIPPED (100%) in 183.39s (03:03)**

All test suites across preprocessing, augmentation, DataLoader, model architecture, training loops, training analysis, hyperparameter tuning, Day 9 test evaluation, Day 10 Grad-CAM, Day 11 ResNet-50 benchmark, and Day 12 Streamlit application are passing.

---

## 7. How to Launch and Use the Web Application

To run the MediScan web application locally:

```bash
# From the project root directory
streamlit run app/app.py
```

The application will launch on `http://localhost:8501`.

### Interactive User Workflow:
1. **Choose Input Source**: Select **"Upload Image"** to drag-and-drop any dermoscopic JPEG/PNG image, or **"HAM10000 Dataset Samples"** to test verified clinical cases.
2. **Review Diagnostic Card**: View the Top-1 predicted diagnosis, model confidence percentage, and clinical urgency tier.
3. **Inspect Probability Distribution**: Review ranked bars displaying probabilities across all 7 diagnostic categories.
4. **Examine Grad-CAM Triplet**: Check the standardized input, activation heatmap, and overlay to verify neural localization.
5. **Adjust Explainability Settings**: Use the sidebar slider to modify the heatmap blend transparency or backpropagate gradients for alternative diagnostic classes.
6. **Export Findings**: Click **"Download Structured Diagnostic Report (JSON)"** to save a machine-readable summary.

---

## 8. Conclusion & Status Decision

# **PASS — DAY 12 COMPLETE**

The web application and user interface phase for MediScan is fully developed, audited, tested, and complete. All 10 pre-flight criteria and interactive frontend requirements are satisfied without regressions.

### Milestone Verification Matrix:

| Area | Status | Evidence |
|:---|:---:|:---|
| **App Architecture** | **PASS** | Implemented in `app/app.py` and `app/utils.py`. |
| **Model Serving** | **PASS** | EfficientNet-B0 (1280→7) loaded from `efficientnet_b0_best.pth` and cached via `@st.cache_resource`. |
| **Preprocessing** | **PASS** | Exact Day 2–4 deterministic $224 \times 224$ ImageNet pipeline. |
| **Explainability** | **PASS** | Grad-CAM targeting `model.features[8]` with Triplet visualizations and weight verification. |
| **Input Flexibility** | **PASS** | Supports live user uploads and pre-loaded HAM10000 sample gallery. |
| **Clinical Ergonomics**| **PASS** | Risk tiers (High, Moderate, Low), urgency recommendations, and regulatory disclaimers. |
| **Data Integrity** | **PASS** | `split_test.csv` MD5 `6a5ae1b65c25d504f78bc1da2361d82c` unaltered. |
| **Test Verification** | **PASS** | 26/26 Day 12 tests passed. 277/278 full regression suite passed. |
| **Overall** | **PASS** | **DAY 12 COMPLETE**. |
