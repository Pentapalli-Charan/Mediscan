# MediScan — Day 15 Completion Report: Finalization, Deployment & Project Sign-Off

**Author:** Pentapalli Charan  
**Date:** September 18, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 15 — Finalization, Deployment, Model Card & Final Project Sign-Off  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | MLflow 3.16.0 | Streamlit 1.54.0 | OS: Windows 11  

---

## 1. Day 15 Objective

Day 15 is the **final capstone milestone** of the MediScan project. Operating strictly under a **ZERO RETRAINING / ZERO MODEL MODIFICATION** contract, this milestone accomplished:
- Complete cryptographic verification of model weights and dataset splits.
- Elimination of misleading clinical diagnostic/triage terminology from the Streamlit web application in favor of non-diagnostic educational taxonomy.
- Deployment data strategy formulation: curated 7 lightweight training-split demonstration samples (`app/assets/samples/`, 1.9 MB total) eliminating the need to package the multi-GB raw dataset for public deployment.
- Assembly of a self-contained, lightweight deployment package for Hugging Face Spaces (`deployment/hf_space/`, 18.17 MB total).
- Comprehensive dependency segregation (`requirements-deployment.txt`).
- Local end-to-end CPU latency benchmarking (averaging 141.4 ms total end-to-end latency).
- Creation of formal capstone documentation: 22-section [README.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/README.md), 24-section [MODEL_CARD.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/MODEL_CARD.md), and 23-section [reports/FINAL_PROJECT_AUDIT.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/FINAL_PROJECT_AUDIT.md).
- Execution and verification of automated test suites: **12 / 12 passed** in Day 15 tests, and **336 passed, 1 skipped, 0 failed** across the complete repository regression suite.

---

## 2. Files Created
1. `app.py`: Root application entrypoint delegating to `app/app.py` for Hugging Face Spaces.
2. `requirements-deployment.txt`: Minimal runtime dependencies for CPU deployment.
3. `app/assets/samples/samples_metadata.json`: Metadata manifest for the 7 demonstration samples.
4. `app/assets/samples/*.jpg`: 7 curated representative images (one per class, all from `split_train.csv`).
5. `scripts/curate_deployment_samples.py`: Script used to curate sample images from the training split.
6. `scripts/benchmark_smoke_test.py`: Benchmark script measuring real CPU inference latencies.
7. `scripts/prepare_hf_space.py`: Packaging utility assembling the standalone Hugging Face Space bundle.
8. `deployment/hf_space/`: Complete standalone deployment directory (67 files, 18.17 MB).
9. `tests/test_day15_finalization.py`: Comprehensive 12-test suite for Day 15 verification.
10. `MODEL_CARD.md`: Formal 24-section model card document.
11. `reports/FINAL_PROJECT_AUDIT.md`: Complete 23-section technical audit and capstone sign-off report.
12. `reports/DAY15_COMPLETION_REPORT.md`: This completion report.

---

## 3. Files Modified
1. `app/app.py`:
   - Updated `CLINICAL_RISK_LEVELS` to map to histological class types (e.g. `Malignant Neoplasm (Melanoma)`, `Benign Lesion (Melanocytic Nevus)`) rather than alarmist risk/triage labels.
   - Replaced user-facing terms: "Diagnostic Assessment" $\to$ "Model Prediction", "Diagnostic Overlay" $\to$ "Grad-CAM Explanation Overlay", "Clinical Recommendation" $\to$ "Histological Classification (Research Reference Only)".
   - Updated headers and captions to emphasize research/educational prototype status.
   - Inserted prominent, comprehensive educational and clinical limitation disclaimer box.
2. `app/utils.py`:
   - Enhanced `get_ham10000_sample_catalog()` with an automatic fallback to `app/assets/samples/samples_metadata.json` when raw dataset directories are absent (ensuring offline and public deployment resilience).
3. `README.md`:
   - Expanded into comprehensive 22-section final documentation including Hugging Face YAML frontmatter, architecture diagrams, benchmark tables, and installation instructions.

---

## 4. UI Finalization & Non-Diagnostic Terminology Cleanup

In accordance with Phase 3 instructions, all user-facing language was audited:
- **Removed Misleading Terms:** "HIGH RISK", "MODERATE RISK", "LOW RISK", "clinical urgency", "diagnostic assessment", "diagnostic overlay", "Urgent Dermatologic Evaluation & Biopsy Recommended".
- **Adopted Objective Terminology:** "Model Prediction", "Predicted Class", "Model Confidence", "Class Probability", "Grad-CAM Explanation Overlay", "Research / Educational Prototype".
- **Added Prominent Disclaimer:**
  > *"This application is an educational and research prototype for automated dermatoscopic image classification. It is not a medical diagnostic tool and must not be used to make clinical decisions. Predictions may be incorrect and performance may vary across populations, devices, acquisition settings, and clinical environments."*
- **Disclosed Key Biases:** Documented training on HAM10000, lack of clinical trials, domain shift risk, 58:1 class imbalance, and that softmax confidence $\ne$ clinical certainty.

---

## 5. Deployment Preparation & Data Strategy

- **Challenge:** The original HAM10000 dataset contains 10,015 images spanning multiple gigabytes, which should not be bundled into a lightweight public web deployment.
- **Solution:** Curated exactly 1 representative image per diagnostic class (7 images total) directly from the **training partition** (`split_train.csv`).
- **Safety Verification:** Verified that **zero** demonstration images originate from the held-out test split (`split_test.csv`).
- **Metadata Documentation:** Generated `app/assets/samples/samples_metadata.json` documenting image ID, class code, full description, patient age/sex/localization, and CC BY-NC-SA 4.0 license attribution.
- **Total Samples Size:** **1.9 MB** (down from several GBs).

---

## 6. Dependency Audit

Dependencies were audited and separated into two distinct manifests:
- **Development Manifest (`requirements.txt`):** Includes all training, evaluation, Jupyter, and experiment tracking dependencies (47 packages).
- **Deployment Manifest (`requirements-deployment.txt`):** Minimal runtime set for CPU execution:
  - `torch>=2.0.0`
  - `torchvision>=0.15.0`
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
  - `Pillow>=9.5.0`
  - `opencv-python-headless>=4.7.0`
  - `albumentations>=1.3.0`
  - `streamlit>=1.30.0`
  - `grad-cam>=1.4.8`
  - `pyyaml>=6.0`
- **Optional MLflow Handling:** Confirmed that the Streamlit application safely catches `ImportError` / `Exception` when MLflow is absent, rendering static benchmark fallbacks without disruption.

---

## 7. Package-Size Audit

- **Canonical Model Checkpoint:** `models/checkpoints/efficientnet_b0_best.pth` $\to$ **15.68 MB**.
- **Curated Demonstration Samples:** `app/assets/samples/` $\to$ **1.91 MB**.
- **Source Code, Configurations & Metadata:** $\to$ **0.58 MB**.
- **Total Deployment Bundle (`deployment/hf_space/`):** **18.17 MB** across 67 files.
- **Budget Compliance:** Comfortably below Hugging Face Spaces' standard 50 MB git file limit and well within standard free tier allocations.

---

## 8. Security Audit

- **Credential & Token Scanning:** Scanned all `.py`, `.yaml`, and `.json` files across `app/`, `src/`, `config/`, and `scripts/` using regex patterns for:
  - Hugging Face user access tokens (`hf_*`)
  - GitHub personal access tokens (`ghp_*`)
  - OpenAI / third-party API keys (`sk-*`)
  - AWS access keys (`AKIA*`)
- **Result:** **0 credentials detected.**
- **Git Ignore Protection:** Verified that `.gitignore` correctly ignores `kaggle.json`, `models/checkpoints/*.pth`, `data/raw/`, `mlruns/`, and local virtual environments.

---

## 9. Local Production Smoke Test & Latency Benchmarks

The full inference and Grad-CAM pipeline was benchmarked on CPU across all 7 curated demonstration images using [scripts/benchmark_smoke_test.py](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/scripts/benchmark_smoke_test.py):

| Sample Image ID | True Class | Predicted Class | Softmax Confidence | Preprocessing | Forward Inference | Grad-CAM Overlay | Total End-to-End |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ISIC_0029417` | `akiec` | `bkl` | 44.2% | 19.4 ms | 39.7 ms | 224.6 ms | 283.6 ms |
| `ISIC_0034093` | `bcc` | `nv` | 43.2% | 14.2 ms | 23.0 ms | 85.9 ms | 123.1 ms |
| `ISIC_0027419` | `bkl` | `akiec` | 28.0% | 14.6 ms | 23.8 ms | 72.9 ms | 111.3 ms |
| `ISIC_0027008` | `df` | `df` | 34.1% | 16.2 ms | 24.8 ms | 90.0 ms | 131.0 ms |
| `ISIC_0025964` | `mel` | `nv` | 69.3% | 13.3 ms | 23.3 ms | 72.0 ms | 108.6 ms |
| `ISIC_0024698` | `nv` | `nv` | 81.3% | 12.9 ms | 23.3 ms | 89.8 ms | 126.0 ms |
| `ISIC_0031197` | `vasc` | `vasc` | 81.8% | 12.3 ms | 21.2 ms | 72.9 ms | 106.4 ms |
| **Averages** | — | — | — | **14.7 ms** | **25.6 ms** | **101.2 ms** | **141.4 ms (0.141 s)** |

**Measurement Parameters:**
- **Hardware Device:** CPU (Intel/AMD x86_64, Windows 11).
- **Sample Count:** 7 representative HAM10000 images.
- **Timing Methodology:** High-resolution monotonic timers (`time.perf_counter()`).
- **Key Takeaway:** Forward neural network evaluation requires only **25.6 ms** on CPU; entire end-to-end preprocessing, inference, and Grad-CAM generation completes in **~0.14 seconds**.

---

## 10. Deployment Result

- **Packaging Status:** **COMPLETE & LOCALLY VALIDATED.**
- **Deployment Artifact Location:** `deployment/hf_space/`.
- **Target Platform:** Hugging Face Spaces (`sdk: streamlit`, `sdk_version: 1.54.0`).
- **Entrypoint:** `app.py` $\to$ `app/app.py`.

---

## 11. Live URL Verification Status

- **Live URL:** **PARTIAL / PENDING USER AUTHENTICATION.**
- **Statement:** The deployment package was fully prepared, tested, and assembled at `deployment/hf_space/`. Because live external Hugging Face account authentication tokens were not provided in the local environment, external deployment was not pushed to a public remote server. **No fictitious deployment URL was generated.**
- **Remaining Manual Step:**
  To publish the prepared bundle to your personal Hugging Face Space:
  ```bash
  # 1. Log in to Hugging Face
  huggingface-cli login
  # 2. Clone your target Space repository
  git clone https://huggingface.co/spaces/<your-username>/mediscan
  # 3. Copy files from deployment/hf_space/
  cp -r "deployment/hf_space/*" mediscan/
  # 4. Commit and push
  cd mediscan
  git add .
  git commit -m "Deploy MediScan v1.0"
  git push
  ```

---

## 12. Model Integrity Hashes

- **Canonical Checkpoint:** `models/checkpoints/efficientnet_b0_best.pth`
- **MD5 Checksum:** **`7c1c6fcbe02e93f0ff8b4a20f62e29e3`**
- **Status:** ✅ Verified bitwise identical to Day 5/6/9/13/14 checkpoints. Zero drift.

---

## 13. Test Split Hash

- **Canonical Test Split:** `reports/data/split_test.csv` (1,512 samples)
- **MD5 Checksum:** **`6a5ae1b65c25d504f78bc1da2361d82c`**
- **Status:** ✅ Verified bitwise identical. Zero data drift.

---

## 14. Day 9 Metric Integrity

All Day 9 locked held-out test evaluation metrics remain unchanged and verified:
- **Top-1 Accuracy:** `0.757937` (75.79%)
- **Top-2 Accuracy:** `0.895503` (89.55%)
- **Macro F1-Score:** `0.515696`
- **Weighted F1-Score:** `0.744771`
- **Macro Precision:** `0.574358`
- **Macro Recall:** `0.478688`
- **Macro ROC-AUC:** `0.921136`
- **Weighted ROC-AUC:** `0.914520`
- **Macro PR-AUC:** `0.564585`
- **Test Loss:** `0.663784`

---

## 15. Final Test Results

- **Test Suite:** `tests/test_day15_finalization.py`
- **Result:** **12 passed in 7.02s** (100% pass rate).
- **Coverage:** Checkpoint hash, test split hash, Day 9 metrics, class mapping, Grad-CAM layer (`features[8]`), app compilation, inference pipeline, curated demonstration samples, non-diagnostic UI phrasing, deployment bundle size budget, dependency coverage, and secret scanning.

---

## 16. Full Regression Results

- **Command Executed:** `pytest -q`
- **Test Scope:** Full project test suite across all 15 milestones.
- **Pass / Fail Count:** **336 passed, 1 skipped, 0 failed** in 1,267.10s (21 min 7 s).
- **Regression Status:** **ZERO REGRESSIONS.** 100% of historical milestones remain intact.

---

## 17. Documentation Completed

- [x] [README.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/README.md): Finalized with all 22 required sections.
- [x] [MODEL_CARD.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/MODEL_CARD.md): Formal 24-section model card.
- [x] [reports/FINAL_PROJECT_AUDIT.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/FINAL_PROJECT_AUDIT.md): Final technical sign-off and audit.
- [x] [reports/DAY15_COMPLETION_REPORT.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/DAY15_COMPLETION_REPORT.md): This report.

---

## 18. Model Card Status

**COMPLETE & VALIDATED.** Formally archived in project root as [MODEL_CARD.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/MODEL_CARD.md).

---

## 19. Final Audit Status

**COMPLETE & SIGNED OFF.** Formally archived in reports directory as [reports/FINAL_PROJECT_AUDIT.md](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/FINAL_PROJECT_AUDIT.md).

---

## 20. Known Limitations Summary

1. **False Negatives in Melanoma:** 41.67% of melanomas are misclassified as benign nevi under standard argmax selection.
2. **Class Imbalance:** Extreme 58.3:1 dataset imbalance creates strong baseline predictive priors toward benign nevi.
3. **High-Confidence Errors:** 45 test errors occur with softmax confidence $\ge 0.80$, proving confidence does not equal certainty.
4. **Demographic Homogeneity:** Dataset consists almost exclusively of fair-skinned European populations; darker skin phototypes are unrepresented.
5. **Blur Fragility:** Out-of-focus blur causes a 24.0% class flip rate.
6. **Domain Specificity:** Untested on macroscopic smartphone images.

---

## 21. Acceptance Checklist

| Acceptance Criterion | Result | Evidence |
|:---|:---:|:---|
| Final model checkpoint verified unchanged | ✅ PASS | MD5 `7c1c6fcbe02e93f0ff8b4a20f62e29e3` verified |
| Test split verified unchanged | ✅ PASS | MD5 `6a5ae1b65c25d504f78bc1da2361d82c` verified |
| Day 9 metrics verified unchanged | ✅ PASS | Verified against `reports/data/day9_metrics.json` |
| Grad-CAM target verified as `model.features[8]` | ✅ PASS | Verified in `tests/test_day15_finalization.py` |
| Streamlit UI finalized with non-diagnostic wording | ✅ PASS | "Model Prediction", educational classification adopted |
| Prominent clinical & educational disclaimer present | ✅ PASS | Verified rendered in app and checked via test suite |
| Local application smoke test verified | ✅ PASS | End-to-end CPU latency: 141.4 ms |
| Deployment dependencies audited | ✅ PASS | Segregated into `requirements-deployment.txt` |
| Deployment package created | ✅ PASS | 18.17 MB package assembled at `deployment/hf_space/` |
| Full HAM10000 dataset excluded from deployment | ✅ PASS | Curated 7 demonstration images (1.9 MB) |
| No secrets or credentials committed | ✅ PASS | Automated regex security audit passed |
| Hugging Face Spaces configuration created | ✅ PASS | Frontmatter in `README.md` + root `app.py` |
| Live deployment status documented honestly | ✅ PASS | Deployment package ready; auth step documented |
| Final Day 15 test suite passes | ✅ PASS | 12 / 12 passed in `tests/test_day15_finalization.py` |
| Full regression suite has 0 failures | ✅ PASS | 336 passed, 1 skipped, 0 failed in 1,267.10s |
| Final README.md completed | ✅ PASS | All 22 required sections implemented |
| MODEL_CARD.md created | ✅ PASS | All 24 required sections implemented |
| FINAL_PROJECT_AUDIT.md created | ✅ PASS | All 23 required sections implemented |
| Git repository clean & review-friendly | ✅ PASS | Verified via git status & test assertions |
| Known limitations documented | ✅ PASS | Documented across all reports and UI |

---

## 22. Final Project Status

**DAY 15 IS OFFICIALLY COMPLETE.**  
All 15 days of the MediScan internship capstone project have been fully implemented, rigorously validated, documented, and signed off.
