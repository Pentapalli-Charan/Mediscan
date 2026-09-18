# MediScan — Final Technical Sign-Off & Project Audit Report

**Author / Lead Engineer:** Pentapalli Charan  
**Date:** September 18, 2026  
**Project:** MediScan — End-to-End Dermatoscopic Image Classification & Explainability System  
**Milestone:** Day 15 — Finalization, Deployment, Verification & Formal Capstone Sign-Off  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | MLflow 3.16.0 | Streamlit 1.54.0 | OS: Windows 11  

---

## 1. Project Objective
MediScan was engineered as a 15-day technical internship capstone to build an end-to-end, reproducible, medically grounded deep learning classification system for dermatoscopic skin lesions. The project spans the complete machine learning lifecycle: data integrity verification, leakage-free patient grouping, transfer learning with EfficientNet-B0, staged evaluation, Grad-CAM visual interpretability, MLflow experiment tracking, bias and calibration audits, and a production-ready Streamlit web deployment.

---

## 2. Dataset & Provenance
- **Dataset:** HAM10000 ("Human Against Machine with 10000 training images").
- **Size:** 10,015 dermatoscopic images (600×450 JPEG).
- **Clinical Sources:** ViDIR Group, Medical University of Vienna (Austria) and Cliff Rosendahl Clinic (Queensland, Australia).
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
- **7 Diagnostic Classes:**
  `akiec` (Actinic keratoses), `bcc` (Basal cell carcinoma), `bkl` (Benign keratosis), `df` (Dermatofibroma), `mel` (Melanoma), `nv` (Melanocytic nevi), `vasc` (Vascular lesions).

---

## 3. Data Integrity Verification
- **Duplicate Images:** Verified zero duplicate file hashes.
- **Corrupt Images:** 100% of images verified valid and readable by PIL and OpenCV.
- **Metadata Alignment:** 10,015 CSV rows perfectly map 1-to-1 with existing image files on disk.

---

## 4. Data Leakage Prevention
- **Challenge:** Multiple images frequently originate from the same physical lesion (`lesion_id`).
- **Solution:** Group-aware stratified splitting (`GroupKFold` on `lesion_id`).
- **Validation:**
  - Train: 6,982 images (5,263 lesions)
  - Validation: 1,521 images (1,086 lesions)
  - Test: 1,512 images (1,121 lesions)
  - **Lesion Overlap Across Splits:** **0 lesions (0.00% leakage).**

---

## 5. Model Architecture & Weights Lock
- **Canonical Model:** EfficientNet-B0 (`torchvision.models.efficientnet_b0`, ImageNet pretrained).
- **Classifier Head:** `AdaptiveAvgPool2d(1) -> Dropout(0.20) -> Linear(1280, 7)`.
- **Checkpoint Location:** `models/checkpoints/efficientnet_b0_best.pth`.
- **Cryptographic Hash (MD5):** `7c1c6fcbe02e93f0ff8b4a20f62e29e3` (100% verified match).
- **Parameters:** Total: 4,016,515 | Trainable: 8,967 (0.22%) | Frozen: 4,007,548 (99.78%).

---

## 6. Training Pipeline Execution
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=0.001$).
- **Batch Size:** 32 (ImageNet normalized $224 \times 224$).
- **Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=2, min_lr=1e-6).
- **Early Stopping:** Patience 3 epochs.
- **Completed Epochs:** 15 epochs (Day 7).

---

## 7. Hyperparameter Evaluation
Benchmarked in Day 8 and logged in MLflow:
- `Baseline (LR 1e-3, Drop 0.20)`: Best Val Loss `0.6422`, Val Acc `77.38%` (Optimal).
- `Candidate Exp 1 (LR 5e-4, Drop 0.20)`: Best Val Loss `0.7283`, Val Acc `75.21%` (Early stopped at epoch 5).

---

## 8. Final Validation Performance (Model Selection)
- **Best Validation Loss:** **0.6422** (Epoch 14)
- **Validation Accuracy at Best Loss:** **77.38%**
- **Peak Validation Accuracy:** **77.65%** (Epoch 13)

---

## 9. Final Held-Out Test Performance (Locked Day 9 Evaluation)
Evaluated on the locked test partition (1,512 samples, MD5: `6a5ae1b65c25d504f78bc1da2361d82c`):
- **Top-1 Accuracy:** **75.79%** (1,146 / 1,512 correct)
- **Top-2 Accuracy:** **89.55%** (1,354 / 1,512 correct)
- **Macro Precision:** **0.5744**
- **Macro Recall:** **0.4787**
- **Macro F1-Score:** **0.5157**
- **Weighted F1-Score:** **0.7448**
- **Macro ROC-AUC:** **0.9211**
- **Weighted ROC-AUC:** **0.9145**
- **Macro PR-AUC:** **0.5646**
- **Cross-Entropy Loss:** **0.6638**

---

## 10. Explainability (Grad-CAM)
- **Target Layer:** `model.features[8]` (penultimate convolutional feature layer).
- **Weight Safety:** Gradients are isolated; model weights are snapshot and verified bitwise identical after execution.
- **Integration:** Real-time visual overlay in Streamlit with adjustable blend opacity $\alpha \in [0.10, 0.90]$.

---

## 11. Model Comparison (Day 11)
EfficientNet-B0 was benchmarked against ResNet-50:
- EfficientNet-B0 has **5.85× fewer parameters** (4.02M vs 23.52M).
- EfficientNet-B0 achieved **+4.86% higher validation accuracy** (77.38% vs 72.52%) and **16.0% lower validation loss** (0.6422 vs 0.7646).
- EfficientNet-B0 CPU inference latency is **2.3× faster** (25.6 ms vs 58.2 ms).

---

## 12. MLflow Experiment Tracking (Day 13)
- **Backend:** Local file-based store at `mlruns/` under experiment **`MediScan`** (ID: `944157940989004540`).
- **Synchronized Runs:** 3 actual historical runs (`EfficientNet-B0-Baseline`, `EfficientNet-B0-LR-5e-4`, `ResNet50-Benchmark`).
- **Parameters Logged:** 30 parameters per run.
- **Artifacts Stored:** Epoch history CSVs, comparison charts, and classification reports.

---

## 13. Bias Analysis (Day 14)
- **Class Imbalance:** 58.3:1 ratio between `nv` (66.95%) and `df` (1.15%) creates a heavy prior toward predicting benign nevi.
- **Clinical Failure Rate:** 41.67% of melanomas (70 cases) were misclassified as benign nevi (`nv`).
- **Subgroup Disparities:** Accuracy on geriatric patients ($\ge 60$ yrs) is **61.04%** vs. **86.17%** in $<40$ yrs. Facial lesions exhibited **44.94%** accuracy due to complex actinic/lentigo pseudonetworks.

---

## 14. Robustness Analysis (Day 14)
Tested across 6 synthetic image perturbations on 150 stratified samples:
- **Gaussian Blur:** Caused highest sensitivity (consistency fell to 76.0%, class flip rate 24.0%).
- **Crop & Resize:** 82.7% consistency, 17.3% flip rate.
- **JPEG Compression (Q=50):** 84.0% consistency, 16.0% flip rate.
- **Mild Brightness / Color Shifts:** 85.3% – 86.0% consistency.
- **Contrast Shifts:** 89.3% consistency (highest stability).

---

## 15. Calibration Analysis (Day 14)
- **Expected Calibration Error (ECE):** **0.0350** across 10 equal-width bins.
- **Maximum Calibration Error (MCE):** **0.1489** (in bin $[0.2, 0.3]$).
- **High-Confidence Accuracy:** In bin $[0.9, 1.0]$ (41.87% of predictions), empirical accuracy is **97.79%** against 97.29% mean confidence.
- **Overconfident Errors:** 45 test errors occurred with confidence $\ge 0.80$.

---

## 16. Streamlit Web Application (Day 12 & Day 15)
- **Interface:** Modern medical aesthetic with responsive two-column layout.
- **Input Modes:** Real-time file uploader (JPEG/PNG) and curated sample selector.
- **Safety Terminology:** Replaced alarmist clinical triage terminology with non-diagnostic educational categorization and prominent limitation banners.
- **Export Feature:** Structured JSON export of predictions and probability distributions.

---

## 17. Deployment Status (Day 15)
- **Platform Target:** Hugging Face Spaces (Streamlit SDK 1.54.0).
- **Bundle Prepared:** Standalone deployment package created at `deployment/hf_space/`.
- **Bundle Size:** **18.17 MB** total (including frozen weights of 15.68 MB and 7 demo images of 1.9 MB).
- **Full Dataset Avoidance:** The multi-GB raw dataset is strictly excluded.
- **Live Verification Status:** Deployment bundle is fully prepared and locally verified via automated smoke tests; live browser deployment to Hugging Face Spaces is ready for one-command git push with user credentials.

---

## 18. Security & Package Audit
- **Credentials / Tokens:** Zero API keys, personal access tokens, or credentials committed (verified via regex scanner).
- **Git Ignore:** Configured to exclude raw data, MLflow run artifacts, large models, and cache files.
- **Dependencies:** Audited and segregated into development (`requirements.txt`) and runtime (`requirements-deployment.txt`).

---

## 19. Automated Test Results
- **Day 15 Finalization Tests:** **12 passed** in 7.02s (`tests/test_day15_finalization.py`).
- **Day 14 Robustness Tests:** **37 passed** in 19.13s (`tests/test_day14_robustness.py`).
- **Day 12 Streamlit App Tests:** **5 passed** in 13.71s (`tests/test_day12_app.py`).
- **App Utils Unit Tests:** **21 passed** in 13.53s (`tests/test_app_utils.py`).
- **Full Project Regression:** **336 passed, 1 skipped** (zero failures across all project milestones).

---

## 20. Documentation Status
- [x] [README.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/README.md): All 22 required sections complete.
- [x] [MODEL_CARD.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/MODEL_CARD.md): All 24 required sections complete.
- [x] [FINAL_PROJECT_AUDIT.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/FINAL_PROJECT_AUDIT.md): Technical sign-off complete.
- [x] [DAY15_COMPLETION_REPORT.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/DAY15_COMPLETION_REPORT.md): Final completion report generated.

---

## 21. Known Limitations
1. Overconfident error modes: 45 test errors with $\ge 80\%$ confidence.
2. High false-negative rate on melanoma (41.67% misdiagnosed as benign).
3. Demographic bias: Lack of darker skin phototypes (Fitzpatrick IV–VI).
4. Closed vocabulary: Strictly limited to the 7 HAM10000 classes.

---

## 22. Ethical Considerations
1. Educational / Research Scope: Must never be deployed as a clinical diagnostic tool.
2. Clinical Safety: Overreliance on automated predictions could lead to delayed diagnosis of malignant melanoma.
3. Algorithmic Transparency: Grad-CAM and confidence calibration must be provided alongside any classification.

---

## 23. Final Acceptance Checklist
| Item | Verification Criteria | Status |
|:---|:---|:---:|
| 1 | Final model checkpoint verified unchanged (`7c1c6fcbe02e93f0ff8b4a20f62e29e3`) | ✅ PASS |
| 2 | Test split verified unchanged (`6a5ae1b65c25d504f78bc1da2361d82c`) | ✅ PASS |
| 3 | Day 9 metrics verified unchanged (75.79% Top-1, 0.5157 Macro F1, 0.9211 ROC-AUC) | ✅ PASS |
| 4 | Grad-CAM target verified as `model.features[8]` | ✅ PASS |
| 5 | Streamlit UI finalized and safety wording non-diagnostic | ✅ PASS |
| 6 | Prominent educational/research disclaimer present | ✅ PASS |
| 7 | Local application smoke tests pass (141.4 ms total end-to-end latency) | ✅ PASS |
| 8 | Deployment dependencies audited (`requirements-deployment.txt`) | ✅ PASS |
| 9 | Deployment package created at `deployment/hf_space/` (18.17 MB) | ✅ PASS |
| 10 | Full HAM10000 dataset is NOT packaged for public deployment | ✅ PASS |
| 11 | No secrets or credentials committed | ✅ PASS |
| 12 | Hugging Face Spaces configuration created (`README.md` frontmatter + `app.py`) | ✅ PASS |
| 13 | Live deployment status documented (Bundle ready; auth required for remote push) | ✅ PASS |
| 14 | Final Day 15 tests pass (12 / 12 passed) | ✅ PASS |
| 15 | Full regression suite passes with 0 failures (336 passed, 1 skipped) | ✅ PASS |
| 16 | Final README.md completed (all 22 sections) | ✅ PASS |
| 17 | Formal MODEL_CARD.md completed (all 24 sections) | ✅ PASS |
| 18 | Final audit report completed (all 23 sections) | ✅ PASS |
| 19 | Git repository clean and review-friendly | ✅ PASS |
| 20 | All known limitations and ethical considerations documented | ✅ PASS |

**OVERALL SIGN-OFF STATUS: ✅ COMPLETE & APPROVED FOR CAPSTONE ARCHIVAL**
