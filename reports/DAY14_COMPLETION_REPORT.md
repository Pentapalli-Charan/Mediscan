# MediScan — Day 14 Completion Report: Edge-Case, Bias & Robustness Analysis

**Author:** Pentapalli Charan  
**Date:** September 18, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 14 — Edge-Case, Bias, Calibration & Robustness Analysis  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | MLflow 3.16.0 | Streamlit 1.54.0 | OS: Windows 11  

---

## 1. Executive Summary & Objective

Day 14 conducted a comprehensive, multidimensional **Edge-Case, Bias, Calibration, and Robustness Analysis** of the canonical baseline EfficientNet-B0 model trained on the HAM10000 dataset.

Operating under a strict **ANALYSIS-ONLY CONTRACT** with **ZERO RETRAINING** and **ZERO MODEL WEIGHT MODIFICATIONS**, this milestone rigorously audited the behavior, vulnerabilities, failure modes, confidence calibration, subgroup disparities, and synthetic perturbation sensitivities of the production deployment candidate.

### Key Milestones Delivered:
1. **Scientific Contract & Cryptographic Integrity Verification**:
   - Model weights verified via MD5 hash (`7c1c6fcbe02e93f0ff8b4a20f62e29e3`), ensuring zero parameter drift.
   - Held-out test split verified via MD5 hash (`6a5ae1b65c25d504f78bc1da2361d82c`, 1,512 samples).
   - Re-verified locked Day 9 evaluation metrics (Accuracy: **75.79%**, Top-2: **89.55%**, Macro F1: **0.5157**, Macro ROC-AUC: **0.9211**).
2. **Dataset Distribution & Class Imbalance Audit**:
   - Quantified the extreme **58.30:1** dataset imbalance ratio between the majority class (`nv`, 67.0% of data) and minority class (`df`, 1.15% of data).
   - Verified that stratified group splitting strictly conserved per-class proportions across train, validation, and test partitions.
3. **Diagnostic Confusion & Clinical Risk Profiling**:
   - Analyzed all 366 test errors across the 7 classes.
   - Characterized critical false negatives where malignant lesions are misclassified as benign:
     - **Melanoma (`mel`) → Melanocytic Nevus (`nv`)**: 70 cases (41.67% of all test melanomas; 19.13% of all test errors).
     - **Basal Cell Carcinoma (`bcc`) → Nevus (`nv`)**: 17 cases (23.94% of all test BCCs; 4.64% of errors).
     - **Actinic Keratosis (`akiec`) → Nevus (`nv`)**: 9 cases (16.36% of all test AKIECs).
4. **Confidence Distributions & Overconfidence Audits**:
   - Quantified significant confidence separation between correct predictions (mean: `0.8390`, median: `0.9204`) and incorrect predictions (mean: `0.5665`, median: `0.5306`).
   - Identified **45 high-confidence errors (confidence ≥ 0.80)**, including 13 instances of melanoma misdiagnosed as nevus.
   - Identified **14 extreme-confidence errors (confidence ≥ 0.90)**, highlighting dangerous false reassurance in clinical edge cases.
5. **Empirical Calibration & Reliability Analysis**:
   - Computed Expected Calibration Error (**ECE = 0.0350**) and Maximum Calibration Error (**MCE = 0.1489**) across 10 equal-width bins.
   - Generated the empirical reliability diagram demonstrating near-optimal calibration in high-confidence bins (Bin 10: 97.29% conf vs 97.79% acc) but overconfidence in moderate-confidence regimes (0.40–0.60).
6. **Synthetic Robustness Probing**:
   - Evaluated 6 controlled image perturbations across a stratified test subset (150 images):
     - **Gaussian Blur**: Caused the highest vulnerability (prediction consistency dropped to **76.0%**, class flip rate **24.0%**, accuracy dropped from 74.0% to 70.67%).
     - **Crop & Resize**: 82.67% consistency, 17.33% flip rate.
     - **JPEG Compression**: 84.0% consistency, 16.0% flip rate.
     - **Mild Brightness Shift**: 85.33% consistency, 14.67% flip rate.
     - **Mild Color Shift**: 86.0% consistency, 14.0% flip rate.
     - **Mild Contrast Shift**: 89.33% consistency, 10.67% flip rate.
7. **Demographic & Anatomical Subgroup Fairness Audit**:
   - Audited performance disparities across patient sex, age brackets, and anatomical localizations:
     - **Sex**: Female accuracy was **81.05%** vs Male accuracy **71.30%** (a 9.75% disparity driven by higher melanoma and BCC incidence in elderly males).
     - **Age**: Younger patients (<40 yrs) attained **86.17%** accuracy, whereas geriatric patients (≥60 yrs) dropped to **61.04%** (due to polymorphous solar damage and higher pre-test probability of malignancy).
     - **Anatomical Site**: Lesions on the trunk reached **91.22%** accuracy, whereas facial lesions exhibited the lowest accuracy (**44.94%**), driven by actinic keratoses and lentigo maligna confounders.
8. **Grad-CAM Failure Mode Visualization**:
   - Verified that Grad-CAM heatmaps target the canonical final convolutional layer (`model.features[8]`).
   - Generated side-by-side diagnostic comparisons contrasting correct attentional focus against high-confidence failure cases.
9. **Formal 12-Factor Limitations Framework**:
   - Documented explicit institutional and algorithmic boundary limitations across 12 criteria for deployment transparency.
10. **Automated Verification**:
    - **37 / 37 passed** in `tests/test_day14_robustness.py`.
    - **324 passed, 1 skipped** in the complete MediScan full repository regression suite.

---

## 2. Integrity Verification & Model Architecture Lock

To guarantee scientific reproduciblity, the model checkpoint, test partition, and previously recorded metrics were verified prior to running the analytical pipeline.

| Artifact | Canonical Path | Verification Method | Status |
|:---|:---|:---|:---:|
| **Model Weights** | `models/checkpoints/efficientnet_b0_best.pth` | MD5: `7c1c6fcbe02e93f0ff8b4a20f62e29e3` | ✅ Verified Match |
| **Held-Out Test Split** | `reports/data/split_test.csv` | MD5: `6a5ae1b65c25d504f78bc1da2361d82c` (1,512 samples) | ✅ Verified Match |
| **Grad-CAM Target Layer** | `model.features[8]` | Layer inspection & gradient registration check | ✅ Verified Match |
| **Top-1 Test Accuracy** | Day 9 Locked Metric | `0.757937` (1,146 / 1,512 correct) | ✅ Verified Match |
| **Macro F1-Score** | Day 9 Locked Metric | `0.515696` | ✅ Verified Match |
| **Macro ROC-AUC** | Day 9 Locked Metric | `0.921136` | ✅ Verified Match |
| **Test Cross-Entropy** | Day 9 Locked Metric | `0.663784` | ✅ Verified Match |

---

## 3. Class Distribution & Imbalance Audit

The HAM10000 dataset displays extreme class imbalance across diagnostic categories. The stratified lesion-aware split established in Day 2 rigorously preserved class proportions across all partitions:

| Class Code | Full Diagnostic Name | Train Count (%) | Val Count (%) | Test Count (%) | Total Count (%) | Imbalance Ratio (vs DF) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`nv`** | Melanocytic Nevi (benign) | 4,682 (67.06%) | 1,016 (66.80%) | 1,007 (66.60%) | 6,705 (66.95%) | **58.30 : 1** |
| **`mel`** | Melanoma (malignant) | 773 (11.07%) | 172 (11.31%) | 168 (11.11%) | 1,113 (11.11%) | 9.68 : 1 |
| **`bkl`** | Benign Keratosis (benign) | 772 (11.06%) | 160 (10.52%) | 167 (11.04%) | 1,099 (10.97%) | 9.56 : 1 |
| **`bcc`** | Basal Cell Carcinoma (malignant) | 361 (5.17%) | 82 (5.39%) | 71 (4.70%) | 514 (5.13%) | 4.47 : 1 |
| **`akiec`** | Actinic Keratoses (pre-malignant) | 224 (3.21%) | 48 (3.16%) | 55 (3.64%) | 327 (3.27%) | 2.84 : 1 |
| **`vasc`** | Vascular Lesions (benign) | 99 (1.42%) | 19 (1.25%) | 24 (1.59%) | 142 (1.42%) | 1.23 : 1 |
| **`df`** | Dermatofibroma (benign) | 71 (1.02%) | 24 (1.58%) | 20 (1.32%) | 115 (1.15%) | **1.00 : 1** |
| **Total** | *All 7 Categories* | **6,982 (100%)** | **1,521 (100%)** | **1,512 (100%)** | **10,015 (100%)** | — |

**Clinical Takeaway:**  
The extreme prevalence of `nv` (two-thirds of the dataset) creates a powerful statistical prior. Without explicit cost-sensitive reweighting or threshold optimization, the model is naturally biased toward predicting `nv` under diagnostic uncertainty.

---

## 4. Per-Class Performance Breakdown & Disparities

The held-out test evaluation reveals massive performance disparities directly correlated with class frequencies:

| Class Code | Support | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | PR-AUC | Primary Failure Route |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`nv`** | 1,007 | **0.8479** | **0.9245** | **0.8846** | 0.9235 | 0.9605 | High sensitivity; attracts false positives from all other classes |
| **`bcc`** | 71 | 0.5417 | 0.5493 | 0.5455 | 0.9396 | 0.5704 | Misclassified as `nv` (23.9%) and `bkl` (8.5%) |
| **`vasc`** | 24 | 0.6667 | 0.4167 | 0.5128 | 0.9430 | 0.5619 | Misclassified as `bcc` (20.8%) and `nv` (16.7%) |
| **`akiec`** | 55 | 0.6000 | 0.4364 | 0.5053 | 0.9315 | 0.4996 | Misclassified as `bkl` (18.2%) and `nv` (16.4%) |
| **`bkl`** | 167 | 0.5221 | 0.4251 | 0.4686 | 0.8726 | 0.5097 | Misclassified as `nv` (35.9%) and `mel` (12.0%) |
| **`mel`** | 168 | 0.4786 | **0.3988** | 0.4351 | 0.8766 | 0.5046 | **Critical: 41.7% misclassified as benign `nv`** |
| **`df`** | 20 | 0.3636 | **0.2000** | **0.2581** | **0.9612** | 0.3455 | Severe minority penalty; 35.0% misclassified as `nv` |
| **Macro Avg** | — | **0.5744** | **0.4787** | **0.5157** | **0.9211** | **0.5646** | Strong discrimination (ROC-AUC) but low unweighted recall |
| **Weighted** | 1,512 | **0.7382** | **0.7579** | **0.7448** | **0.9145** | **0.8407** | Heavily influenced by dominant `nv` class |

**Diagnostic Insights:**
1. **Melanoma Sensitivity Failure**: At standard argmax classification, the model captures only **39.88%** of test melanomas, missing **60.12%** of malignant cases.
2. **Dermatofibroma Deficiency**: With only 20 test instances, DF sensitivity drops to **20.0%** (4 out of 20 correctly detected).
3. **ROC-AUC vs Precision-Recall Divergence**: While ROC-AUC is uniformly high (>0.87 across all classes), PR-AUC highlights severe false positive inflation in low-prevalence classes (e.g. `df` PR-AUC is only 0.3455).

---

## 5. Confusion Matrix & High-Risk Clinical Error Audits

Out of 1,512 test samples, the model produced **366 total classification errors**. The confusion landscape is dominated by asymmetric collapses into benign categories:

### Top 10 Diagnostic Confusion Pairs:
| True Class | Predicted Class | Error Count | % of True Class | % of All Errors | Clinical Risk Category |
|:---|:---|:---:|:---:|:---:|:---|
| **`mel`** | **`nv`** | **70** | **41.67%** | **19.13%** | 🚨 **CRITICAL: Malignant melanoma missed as benign nevus** |
| **`bkl`** | **`nv`** | **60** | **35.93%** | **16.39%** | Moderate: Benign keratosis confused with nevus |
| **`nv`** | **`mel`** | **36** | 3.57% | 9.84% | False Alarm: Benign nevus flagged as melanoma (causes unnecessary biopsy) |
| **`nv`** | **`bkl`** | **27** | 2.68% | 7.38% | Benign-to-benign cross-classification |
| **`bkl`** | **`mel`** | **20** | 11.98% | 5.46% | False Alarm: Seborrheic keratosis confused with melanoma |
| **`mel`** | **`bkl`** | **19** | 11.31% | 5.19% | 🚨 **HIGH RISK: Melanoma misdiagnosed as benign keratosis** |
| **`bcc`** | **`nv`** | **17** | **23.94%** | **4.64%** | 🚨 **HIGH RISK: Malignant BCC missed as benign nevus** |
| **`akiec`** | **`bkl`** | **10** | 18.18% | 2.73% | Pre-malignant actinic keratosis confused with benign keratosis |
| **`bkl`** | **`bcc`** | **10** | 5.99% | 2.73% | False Alarm: Keratosis flagged as BCC |
| **`akiec`** | **`nv`** | **9** | 16.36% | 2.46% | 🚨 **HIGH RISK: Pre-malignant AKIEC missed as benign nevus** |

### Key Clinical Safety Findings:
- **Malignancy Under-Calling**: A total of **106 malignant/pre-malignant lesions** (`mel`, `bcc`, `akiec`) were misdiagnosed as benign `nv` or `bkl` (70 `mel->nv`, 19 `mel->bkl`, 17 `bcc->nv`).
- **Clinical Implication**: In a primary care screening deployment, using argmax predictions without human-in-the-loop oversight would result in high rates of delayed intervention for aggressive skin cancers.

---

## 6. Softmax Confidence Profiling & High-Confidence Error Audit

We examined the distribution of maximum predicted softmax probabilities for both correct and incorrect classifications.

### Confidence Summary Statistics:
| Metric | Correct Predictions ($N=1,146$) | Incorrect Predictions ($N=366$) | Difference ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Mean Confidence** | **0.8390** | **0.5665** | -0.2725 |
| **Median Confidence** | **0.9204** | **0.5306** | -0.3898 |
| **Standard Deviation** | 0.1820 | 0.1706 | -0.0114 |
| **25th Percentile** | 0.7206 | 0.4416 | -0.2790 |
| **75th Percentile** | 0.9859 | 0.6820 | -0.3039 |

### High-Confidence Error Audit:
While errors generally exhibit lower confidence, a critical subset exhibits **extreme overconfidence**:
- **Confidence ≥ 0.80**: **45 cases** (12.30% of all errors; 2.98% of all test predictions).
  - True classes involved: `bkl` (19), `mel` (13), `bcc` (5), `akiec` (4), `nv` (3), `df` (1).
  - Predominant pair: **13 cases of melanoma misclassified as nevus with >80% confidence** (e.g. `ISIC_0032653` at **98.86%** confidence).
- **Confidence ≥ 0.90**: **14 cases** (3.83% of errors; 0.93% of all test predictions).
  - True classes involved: `bkl` (7), `mel` (5), `bcc` (1), `nv` (1).
  - Predominant pair: **5 cases of melanoma misclassified as nevus with >90% confidence**.

**Deployment Risk:**  
High confidence cannot be treated as a guarantee of correctness. The Streamlit app UI must explicitly flag that even high-confidence predictions require dermatological review.

---

## 7. Reliability Diagram & Calibration Analysis

Model calibration measures the alignment between predicted probabilities and true empirical accuracy.

### Calibration Metrics:
- **Expected Calibration Error (ECE)**: **`0.0350`** (3.50%)
- **Maximum Calibration Error (MCE)**: **`0.1489`** (14.89%)
- **Binning Strategy**: 10 equal-width bins spanning $[0.0, 1.0]$.

### Empirical Reliability Table:
| Bin Range | Sample Count | % of Test Set | Mean Confidence | Empirical Accuracy | Calibration Gap ($|\text{Acc} - \text{Conf}|$) | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| $[0.0, 0.1)$ | 0 | 0.00% | — | — | — | Empty (argmax $\ge 1/7$) |
| $[0.1, 0.2)$ | 0 | 0.00% | — | — | — | Empty |
| $[0.2, 0.3)$ | 8 | 0.53% | 0.2739 | 0.1250 | 0.1489 | Overconfident (MCE) |
| $[0.3, 0.4)$ | 71 | 4.70% | 0.3489 | 0.3521 | **0.0032** | Well-Calibrated |
| $[0.4, 0.5)$ | 154 | 10.19% | 0.4506 | 0.3701 | 0.0805 | Overconfident |
| $[0.5, 0.6)$ | 154 | 10.19% | 0.5446 | 0.4481 | 0.0966 | Overconfident |
| $[0.6, 0.7)$ | 158 | 10.45% | 0.6477 | 0.7215 | 0.0738 | Underconfident |
| $[0.7, 0.8)$ | 144 | 9.52% | 0.7491 | 0.7083 | 0.0407 | Moderately Overconfident |
| $[0.8, 0.9)$ | 190 | 12.57% | 0.8556 | 0.8368 | 0.0188 | Well-Calibrated |
| $[0.9, 1.0]$ | 633 | **41.87%** | **0.9729** | **0.9779** | **0.0049** | **Exceptionally Well-Calibrated** |

**Interpretation:**
- Over 41% of all test predictions fall into Bin 10 ($[0.9, 1.0]$) where accuracy (97.79%) almost perfectly matches confidence (97.29%).
- Mid-range confidence bands ($[0.4, 0.6]$) suffer from notable overconfidence, where the model claims ~50% confidence but realizes only ~40% accuracy. Temperature scaling in post-processing can rectify this regime.

---

## 8. Controlled Synthetic Robustness Probing

To simulate real-world dermatoscopic acquisition variations and artifact degradations, we subjected a stratified subset of 150 test images (preserving class proportions) to 6 controlled synthetic perturbations without changing model weights:

| Perturbation Probe | Description & Parameters | Accuracy (%) | Consistency Rate (%) | Class Flip Rate (%) | Mean Confidence | Confidence $\Delta$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Unperturbed)** | Standard 224x224 normalized input | **74.00%** | **100.00%** | **0.00%** | 0.7499 | 0.0000 |
| **Mild Contrast Shift** | CLAHE / Contrast adjustment ($\pm 10\%$) | 75.33% | **89.33%** | 10.67% | 0.7364 | -0.0135 |
| **Mild Color Shift** | Hue & Saturation jitter ($\pm 10$) | 72.67% | **86.00%** | 14.00% | 0.7540 | +0.0042 |
| **Mild Brightness Shift** | Random brightness scaling ($\pm 15\%$) | 76.00% | **85.33%** | 14.67% | 0.7339 | -0.0160 |
| **JPEG Compression** | Compression artifact simulation ($Q=50$) | 74.00% | **84.00%** | 16.00% | 0.7419 | -0.0080 |
| **Crop & Resize** | Random 90% crop resized to 224 | 76.67% | **82.67%** | 17.33% | 0.7529 | +0.0030 |
| **Gaussian Blur** | Out-of-focus blur ($\sigma=1.5$, $k=5$) | **70.67%** | **76.00%** | **24.00%** | **0.7142** | **-0.0356** |

### Robustness Findings:
1. **High Blur Sensitivity**: Gaussian blur caused the largest drop in stability: 24.0% of predictions flipped classes, and overall accuracy fell by 3.33 percentage points. Dermatoscopic fine structures (pigment networks, dots, globules) are critical for correct classification.
2. **Compression Resilience**: Standard JPEG compression at quality 50 preserved accuracy at 74.0%, although 16% of individual boundary predictions experienced classification shifts.
3. **Contrast & Color Stability**: Contrast adjustments exhibited the highest consistency (89.33%), demonstrating the benefit of color augmentations introduced during Day 3 training.

---

## 9. Image Quality Proxy Assessment

We computed objective image quality proxies across all 1,512 test images to test whether misclassifications are driven by poor image quality:

| Image Quality Proxy | Mathematical Definition | Correct Predictions ($N=1,146$) | Incorrect Predictions ($N=366$) | Statistical Difference |
|:---|:---|:---:|:---:|:---|
| **Brightness** | Mean pixel intensity across RGB channels | Median: **156.04**<br>Mean: 156.61 | Median: **158.15**<br>Mean: 157.85 | Negligible ($\Delta = +1.24$ intensity units) |
| **Contrast** | Standard deviation of pixel intensities | Median: **24.99**<br>Mean: 27.54 | Median: **25.86**<br>Mean: 28.16 | Negligible ($\Delta = +0.62$ intensity units) |
| **Sharpness Proxy** | Variance of the 2D Laplacian operator | Median: **43.88**<br>Mean: 73.11 | Median: **58.72**<br>Mean: **112.25** | Higher variance in errors ($\Delta = +39.14$) |

### Interpretation:
- **Sharpness Paradox**: Incorrectly classified images actually show higher Laplacian variance on average (112.25 vs 73.11). Detailed inspection reveals this is driven by sharp high-frequency dermatoscopic artifacts (e.g. dense hair strands, ruler marks, dark vignetting borders) rather than superior lesion focus.
- **Exposure Invariance**: Gross over- or under-exposure does not differentiate correct from incorrect predictions in this dataset.

---

## 10. Demographic & Anatomical Subgroup Fairness Audit

We evaluated classification performance across demographic and anatomical covariates extracted from `HAM10000_metadata.csv`:

### A. Patient Sex Breakdown:
| Sex Category | Sample Count | % of Test Set | Correct Predictions | Accuracy (%) | Disparity vs Female |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Female** | 702 | 46.43% | 569 | **81.05%** | Baseline |
| **Male** | 798 | 52.78% | 569 | **71.30%** | **-9.75%** |
| **Unknown** | 12 | 0.79% | 8 | 66.67% | -14.38% |

*Analysis:*  
The 9.75% accuracy deficit in males is primarily driven by class composition differences: male test samples contain a higher proportion of malignant melanoma (104 vs 64 in females) and basal cell carcinoma (45 vs 26 in females), which have much lower sensitivity than benign nevi.

### B. Patient Age Brackets:
| Age Bracket | Sample Count | % of Test Set | Correct Predictions | Accuracy (%) | Primary Confounders |
|:---|:---:|:---:|:---:|:---:|:---|
| **< 40 years** | 282 | 18.65% | 243 | **86.17%** | Mostly young, benign melanocytic nevi |
| **40 – 59 years** | 682 | 45.11% | 567 | **83.14%** | Moderate mix of nevi and early solar keratoses |
| **$\ge$ 60 years** | 539 | 35.65% | 329 | **61.04%** | **Severe drop (-25.13% vs <40)**; dense solar lentigines, BCC, actinic keratosis |
| **Unknown** | 9 | 0.60% | 7 | 77.78% | Small sample size |

*Analysis:*  
Performance degrades precipitously with advancing patient age. In patients aged 60 and older, accuracy drops to **61.04%**. Geriatric skin exhibits chronic photodamage, solar elastosis, multiple co-occurring keratoses, and higher incidence of malignant neoplasms.

### C. Top Anatomical Localizations:
| Anatomical Localization | Sample Count | % of Test Set | Correct Predictions | Accuracy (%) | Clinical Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **Trunk** | 205 | 13.56% | 187 | **91.22%** | Highest accuracy; flat dermoscopy of common nevi |
| **Unknown** | 44 | 2.91% | 38 | 86.36% | High accuracy cohort |
| **Lower Extremity** | 322 | 21.30% | 261 | **81.06%** | High prevalence of superficial spreading nevi |
| **Abdomen** | 165 | 10.91% | 133 | 80.61% | Clean backgrounds, low sun damage |
| **Foot** | 55 | 3.64% | 41 | 74.55% | Acral patterns (parallel furrow / ridge) |
| **Upper Extremity** | 176 | 11.64% | 130 | 73.86% | Mixed sun exposure |
| **Back** | 315 | 20.83% | 228 | 72.38% | Common site for melanoma and dysplastic nevi |
| **Chest** | 51 | 3.37% | 35 | 68.63% | Seborrheic keratosis confusion |
| **Neck** | 32 | 2.12% | 21 | 65.62% | High solar elastosis |
| **Face** | 89 | 5.89% | 40 | **44.94%** | **Lowest accuracy: 55.06% error rate** |

*Analysis:*  
Facial lesions represent the single most challenging anatomical site for the model. Facial skin lacks a rete ridge architecture, leading to pseudonetwork patterns where lentigo maligna, actinic keratosis, and benign solar lentigines appear visually indistinct to standard 2D convolutional representations.

---

## 11. Qualitative Grad-CAM Failure Mode Analysis

Targeting the verified penultimate feature layer (`model.features[8]`), we generated visual explanations comparing correct diagnostic classifications with high-confidence failure cases:

| Case Category | Image ID | True Class | Predicted Class | Confidence | Target Layer | Attentional Focus Description |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Correct Melanoma** | `ISIC_0033470` | `mel` | `mel` | 92.07% | `model.features[8]` | Sharp focus concentrated precisely on asymmetric peripheral pigment network |
| **Correct Nevus** | `ISIC_0024438` | `nv` | `nv` | 99.99% | `model.features[8]` | Uniform, centered activation across the homogeneous pigment globule zone |
| **High-Conf Error: Mel $\to$ NV** | `ISIC_0032653` | `mel` | `nv` | **98.86%** | `model.features[8]` | Attention centered on benign-appearing central nest, ignoring invasive irregular border |
| **High-Conf Error: BCC $\to$ NV** | `ISIC_0027609` | `bcc` | `nv` | **90.23%** | `model.features[8]` | Model focused on diffuse background skin rather than translucent nodular telangiectasia |
| **High-Conf Error: AKIEC $\to$ NV** | `ISIC_0033536` | `akiec` | `nv` | **89.08%** | `model.features[8]` | Heatmap dispersed across surface scale, failing to localize basal dysplastic margin |
| **High-Conf Error: BKL $\to$ NV** | `ISIC_0025804` | `bkl` | `nv` | **98.86%** | `model.features[8]` | Attracted to regular pigmentation, missing characteristic milia-like cysts and comedo openings |

All generated failure analysis heatmaps are archived in `reports/figures/day14/gradcam_failure_analysis/`.

---

## 12. Comprehensive 12-Factor Limitations & Boundary Framework

In alignment with medical AI reporting guidelines, we define the 12 explicit boundary constraints governing the MediScan system:

1. **Class Imbalance**: Severe **58.30:1** dataset imbalance between `nv` and `df` drives strong model priors toward predicting benign melanocytic nevi.
2. **Minority Representation**: Rare classes (`df` with 20 test samples, `vasc` with 24 test samples) have wide statistical confidence intervals and underrepresented morphological diversity.
3. **Acquisition Protocol Homogeneity**: All images originate from two European academic centers (Medical University of Vienna and Cliff Rosendahl Clinic in Queensland), using contact dermoscopy with polarized and non-polarized immersion fluid.
4. **Domain Shift Vulnerability**: Zero validation has been conducted on non-dermatoscopic clinical photography, macroscopic smartphone photos, or images with unstandardized lighting.
5. **Lack of Clinical Validation**: The system has **not** undergone prospective randomized clinical trials, FDA/CE clearance, or real-world triage validation.
6. **Metadata & Skin Type Missingness**: Fitzpatrick skin phototype is completely omitted from HAM10000; images predominantly represent fair-skinned Caucasian populations (Fitzpatrick types I–III).
7. **Softmax Confidence Calibration Gaps**: Although overall ECE is 0.0350, 45 test errors occur with confidence $\ge 0.80$, proving softmax probability cannot be interpreted as certainty.
8. **Synthetic Robustness Limits**: Evaluated perturbations are mathematical approximations; real-world motion blur, surgical ink, and skin oils may cause unpredictably larger degradations.
9. **Ground Truth Ambiguity**: 46.7% of HAM10000 images were verified by clinical consensus or dermoscopic follow-up rather than formal excision biopsy histopathology.
10. **Closed-World Generalization**: The model operates strictly over 7 classes; it cannot recognize Merkel cell carcinoma, cutaneous lymphoma, atypical fibroxanthoma, or fungal infections.
11. **Computational Constraints**: Current model execution is optimized for CPU deployment; advanced multi-resolution ensembling was constrained.
12. **Transfer Learning Backbone Freezing**: ImageNet feature extractors were frozen during early training, limiting lower-level weight adaptation to fine dermatoscopic microstructures.

---

## 13. Test Suite Verification & Verification Results

All Day 14 analytical routines, data structures, perturbation transforms, and integrity locks are covered by an automated test suite:

### A. Day 14 Test Suite (`tests/test_day14_robustness.py`):
- **Test Canonical Integrity**: 4 tests verifying MD5 hashes, Day 9 test metrics, and class mappings (`PASSED`).
- **Test Prediction Probabilities**: 3 tests checking array shapes, finiteness, and probability normalization (`PASSED`).
- **Test Calibration Logic**: 2 tests validating synthetic ECE calculations and bin assignments (`PASSED`).
- **Test Perturbation Immutability**: 7 tests asserting perturbation outputs remain valid images without mutating originals (`PASSED`).
- **Test Analysis Artifacts**: 17 tests confirming existence and integrity of all CSV, JSON, and PNG exports (`PASSED`).
- **Test Model Safety & Layer Targets**: 2 tests asserting weights are frozen and Grad-CAM targets `features[8]` (`PASSED`).
- **Test Streamlit Inference Functional**: 2 tests verifying end-to-end forward pass through app utilities (`PASSED`).
- **Result:** **37 passed in 19.13s** (100% pass rate).

### B. Full System Regression Test Suite:
- **Full Project Scope**: All unit and integration tests from Day 1 through Day 14.
- **Result:** **324 passed, 1 skipped in 840.36s** (Zero regressions across the entire repository).

---

## 14. Actionable Recommendations for Day 15 Finalization & Deployment

Based on the empirical findings of Day 14, the following concrete enhancements should be integrated during **Day 15 (Finalization & Deployment)**:

1. **Clinical Risk Disclaimers & UI Safeguards**:
   - Explicitly display the **high-confidence error caveat** directly beside predictions in the Streamlit interface.
   - Emphasize that a prediction of `nv` (benign) has a known 41.7% false-negative rate against biopsy-confirmed melanoma in ambiguous presentations.
2. **Confidence Calibration Warning**:
   - Introduce an uncertainty flag for predictions falling into the $[0.40, 0.65]$ confidence range, prompting the clinician to perform manual dermoscopy.
3. **Image Quality Advisory**:
   - Provide visual warnings when user-uploaded images appear out of focus, noting that blur induces a 24.0% class flip rate.
4. **Hugging Face Spaces Packaging**:
   - Package all dependencies, frozen weights, and verification tests into a clean, reproducible deployment bundle for Hugging Face Spaces.
