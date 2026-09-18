# Model Card for MediScan — EfficientNet-B0 Skin Lesion Classifier

## 1. Model Name
**MediScan EfficientNet-B0 (Day 9 Locked Baseline)**

- **Model Version:** 1.0.0
- **Model Checkpoint:** `models/checkpoints/efficientnet_b0_best.pth`
- **Checkpoint MD5 Checksum:** `7c1c6fcbe02e93f0ff8b4a20f62e29e3`
- **Release Date:** September 18, 2026
- **License:** CC BY-NC-SA 4.0 (derived from HAM10000)

---

## 2. Model Architecture
- **Base Architecture:** EfficientNet-B0 (`torchvision.models.efficientnet_b0`)
- **Pretrained Weights:** ImageNet-1K (`IMAGENET1K_V1`)
- **Feature Extractor:** Frozen convolutional backbone (`model.features`), ending at `model.features[8]` (1280 output channels)
- **Classification Head:**
  - `AdaptiveAvgPool2d(output_size=1)`
  - `Dropout(p=0.20)`
  - `Linear(in_features=1280, out_features=7)`
- **Parameters:**
  - Total Parameters: **4,016,515**
  - Trainable Parameters: **8,967** (0.22%)
  - Non-Trainable Parameters: **4,007,548** (99.78%)
- **Input Dimensions:** $[B, 3, 224, 224]$ RGB float32 tensor
- **Output Dimensions:** $[B, 7]$ unnormalized logits (softmax applied for probabilities)

---

## 3. Intended Use
- **Primary Domain:** Educational demonstrations, academic computer-aided diagnosis research, and algorithmic auditing.
- **Intended Users:** Computer vision students, ML researchers, dermatological data scientists, and educators.
- **Intended Setting:** Offline exploration, classroom demonstrations, and reproducible benchmark evaluations.

---

## 4. Out-of-Scope Use
- **Clinical Diagnosis:** MUST NOT be used for clinical triage, primary screening, diagnostic decision-making, or treatment planning in clinical environments.
- **Consumer Medical Guidance:** MUST NOT be used by patients or consumers as a smartphone self-diagnostic application.
- **Non-Dermatoscope Images:** MUST NOT be deployed on macroscopic digital camera photos, smartphone snapshots, or histological biopsy slices.
- **Emergency Triage:** MUST NOT be used to prioritize biopsy or specialist referrals.

---

## 5. Dataset
- **Name:** HAM10000 ("Human Against Machine with 10000 training images")
- **Sources:**
  - Department of Dermatology, Medical University of Vienna, Austria
  - Cliff Rosendahl Clinic, Queensland, Australia
- **Modalities:** Epiluminescence microscopy (contact dermoscopy, polarized and non-polarized immersion fluid).
- **Ground Truth Confirmation:** Biopsy histopathology (53.3%), confocal microscopy / clinical follow-up (33.3%), expert panel consensus (13.4%).

---

## 6. Dataset Size
- **Total Images:** 10,015 images
- **Unique Lesions:** 7,470 distinct lesions
- **Image Resolution (Raw):** $600 \times 450$ pixels, 24-bit RGB JPEG

---

## 7. Seven Diagnostic Classes
The model categorizes images into 7 mutually exclusive dermatological conditions:
1. **`akiec`** (Actinic Keratoses and Intraepithelial Carcinoma / Bowen's Disease): Pre-malignant squamous proliferation.
2. **`bcc`** (Basal Cell Carcinoma): Common, slow-growing malignant epidermal neoplasm.
3. **`bkl`** (Benign Keratosis-like Lesions): Solar lentigines, seborrheic keratoses, and lichen-planus like keratoses.
4. **`df`** (Dermatofibroma): Benign fibrous histiocytoma.
5. **`mel`** (Melanoma): Aggressive malignant neoplasm of melanocytes.
6. **`nv`** (Melanocytic Nevi): Common benign melanocytic proliferation (moles).
7. **`vasc`** (Vascular Lesions): Angiomas, angiokeratomas, and pyogenic granulomas.

---

## 8. Data Splitting Methodology
- **Split Ratio:** ~70% Train, ~15% Validation, ~15% Test
- **Sample Counts:**
  - Train Split: **6,982 images** (69.72%)
  - Validation Split: **1,521 images** (15.19%)
  - Test Split: **1,512 images** (15.10%)
- **Stratification:** Class frequency distributions are strictly conserved across all three splits.

---

## 9. Data Leakage Prevention
- **Group-Aware Splitting:** Patient lesion clustering (`lesion_id`) was strictly enforced. All images belonging to a particular `lesion_id` were placed exclusively within a single partition.
- **Verification:**
  - Unique Lesions: Train (5,263), Val (1,086), Test (1,121).
  - Train ∩ Val Lesions: **0**
  - Train ∩ Test Lesions: **0**
  - Val ∩ Test Lesions: **0**
  - **Zero Patient/Lesion Leakage (0.0%).**

---

## 10. Preprocessing Pipeline
1. Input image loaded as RGB uint8 array.
2. Bilinear resize to $224 \times 224$ spatial dimensions.
3. Pixel scaling to $[0.0, 1.0]$.
4. ImageNet channel normalization:
   $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$

---

## 11. Training Procedure
- **Hardware:** CPU execution (Intel/AMD x86_64, Windows 11)
- **Framework:** PyTorch 2.10.0+cpu, torchvision 0.28.0+cpu
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\text{lr}=0.001$, $\epsilon=10^{-8}$)
- **Batch Size:** 32
- **Epochs:** 15 epochs
- **Loss:** CrossEntropyLoss
- **Scheduler:** `ReduceLROnPlateau` (factor=0.5, patience=2, min_lr=1e-6)
- **Early Stopping:** Patience of 3 epochs monitoring validation loss

---

## 12. Validation Results (Development & Model Selection)
> **Note:** The validation set was used during model development, hyperparameter selection, and checkpoint saving. It was NOT used for final test reporting.

- **Best Epoch:** 14
- **Best Validation Loss:** **0.6422**
- **Validation Accuracy at Best Loss:** **77.38%**
- **Peak Validation Accuracy:** **77.65%** (Epoch 13)

---

## 13. Locked Held-Out Test Results
> **Note:** The held-out test split (1,512 samples) was evaluated on Day 9 after model selection was locked. It was NEVER used for hyperparameter tuning, checkpoint selection, or feature engineering.

- **Test Split MD5:** `6a5ae1b65c25d504f78bc1da2361d82c`
- **Top-1 Accuracy:** **75.79%** (1,146 / 1,512 correct)
- **Top-2 Accuracy:** **89.55%** (1,354 / 1,512 correct)
- **Macro Precision:** **0.5744**
- **Macro Recall:** **0.4787**
- **Macro F1-Score:** **0.5157**
- **Weighted F1-Score:** **0.7448**
- **Macro ROC-AUC:** **0.9211**
- **Weighted ROC-AUC:** **0.9145**
- **Macro PR-AUC:** **0.5646**
- **Test Loss:** **0.6638**

---

## 14. Per-Class Test Performance
| Class Code | Diagnostic Category | Test Support | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`akiec`** | Actinic Keratoses | 55 | 0.6000 | 0.4364 | 0.5053 | 0.9315 | 0.4996 |
| **`bcc`** | Basal Cell Carcinoma | 71 | 0.5417 | 0.5493 | 0.5455 | 0.9396 | 0.5704 |
| **`bkl`** | Benign Keratosis | 167 | 0.5221 | 0.4251 | 0.4686 | 0.8726 | 0.5097 |
| **`df`** | Dermatofibroma | 20 | 0.3636 | 0.2000 | 0.2581 | 0.9612 | 0.3455 |
| **`mel`** | Melanoma | 168 | 0.4786 | **0.3988** | 0.4351 | 0.8766 | 0.5046 |
| **`nv`** | Melanocytic Nevi | 1,007 | 0.8479 | **0.9245** | 0.8846 | 0.9235 | 0.9605 |
| **`vasc`** | Vascular Lesions | 24 | 0.6667 | 0.4167 | 0.5128 | 0.9430 | 0.5619 |

---

## 15. Explainability (Grad-CAM)
- **Methodology:** Gradient-weighted Class Activation Mapping computes the gradient of the target class score with respect to feature maps of `model.features[8]`.
- **Target Verification:** Formally verified targeting `model.features[8]` (Conv2dNormActivation, 1280 channels).
- **Weight Safety:** Gradients are isolated; model parameters remain bitwise identical after visualization.
- **Observations:** Correct predictions show centered attention on lesion pigment patterns; high-confidence errors show attention dispersed onto surrounding normal skin or localized to non-diagnostic central pigment globules.

---

## 16. Robustness Analysis
Evaluated across 6 controlled synthetic perturbations on a stratified test subset (150 images):
- **Baseline Accuracy:** 74.00%
- **Gaussian Blur:** Consistency fell to **76.00%**; class flip rate **24.00%**; accuracy dropped to 70.67%.
- **Crop & Resize (90%):** 82.67% consistency; 17.33% flip rate.
- **JPEG Compression (Q=50):** 84.00% consistency; 16.00% flip rate.
- **Mild Brightness Shift ($\pm 15\%$):** 85.33% consistency; 14.67% flip rate.
- **Mild Color Shift ($\pm 10$):** 86.00% consistency; 14.00% flip rate.
- **Mild Contrast Shift ($\pm 10\%$):** 89.33% consistency; 10.67% flip rate.

---

## 17. Calibration Analysis
- **Expected Calibration Error (ECE):** **0.0350** (3.50%)
- **Maximum Calibration Error (MCE):** **0.1489** (14.89%)
- **High-Confidence Bin ($[0.9, 1.0]$):** Exceptionally well-calibrated (mean conf 97.29%, accuracy 97.79%, 41.87% of test set).
- **Mid-Range Bins ($[0.4, 0.6]$):** Exhibit moderate overconfidence (mean confidence ~50%, accuracy ~40%).

---

## 18. Known Limitations
1. Overconfident error modes: 45 test errors occur with softmax confidence $\ge 0.80$.
2. Asymmetric diagnostic failure: 41.67% of melanomas are misclassified as benign nevi.
3. Blur sensitivity: High vulnerability to out-of-focus acquisition.
4. Closed vocabulary: Cannot identify any of the thousands of dermatologic conditions outside the 7 HAM10000 classes.

---

## 19. Class Imbalance
- The HAM10000 dataset has a **58.30:1** class imbalance ratio between `nv` (6,705 samples, 66.95%) and `df` (115 samples, 1.15%).
- Resulting Model Bias: High prior toward predicting `nv`, yielding 92.45% sensitivity for `nv` but only 39.88% sensitivity for `mel` and 20.00% for `df`.

---

## 20. Domain Shift Vulnerability
- **Optical Standard:** HAM10000 was collected with dermatoscope lenses; performance on non-dermatoscope images, smartphone cameras, or uneven ambient lighting is untested and will degrade substantially.
- **Clinical Setting:** Collected in high-resource dermatology referral centers; generalization to primary care clinics or self-examination photos is unproven.

---

## 21. Metadata & Demographic Limitations
- **Skin Phototypes:** Fitzpatrick skin type is completely missing from HAM10000 metadata. The dataset predominantly represents light-skinned European phenotypes (Fitzpatrick I–III). Efficacy on darker skin tones is unmeasured.
- **Missing Demographics:** In the test set, age is missing in 9 cases, sex in 12 cases, and anatomical site in 44 cases.
- **Subgroup Disparities:** Accuracy in patients $\ge 60$ years drops to **61.04%**; facial lesions have an accuracy of only **44.94%**.

---

## 22. Lack of Clinical Validation
- The model has NOT been tested in prospective, blinded, randomized clinical trials.
- The model is NOT cleared, approved, or registered with the US FDA, European CE, or any other healthcare regulatory authority as a medical device.

---

## 23. Ethical Considerations
- **Risk of Delayed Treatment:** A false-negative prediction on melanoma can delay life-saving biopsy and excision.
- **Risk of False Reassurance:** Presenting confidence scores to patients without clinical context can cause harmful false reassurance.
- **Fairness & Access:** Algorithmic bias resulting from dataset demographic homogeneity risks exacerbating diagnostic disparities across racial and ethnic groups.

---

## 24. Final Disclaimer
> **⚠️ REGULATORY AND CLINICAL DISCLAIMER:**  
> This model card and the MediScan software are intended exclusively for academic, research, and educational purposes. Under no circumstances should predictions from this model be construed as medical diagnosis, prognosis, triage, or clinical advice. Always consult a qualified dermatologist for skin lesion assessment.
