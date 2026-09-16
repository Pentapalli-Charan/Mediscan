# MediScan — Day 9 Completion Report: Comprehensive Baseline Model Evaluation & Error Analysis

**Author:** Pentapalli Charan  
**Date:** September 14, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 9 — Comprehensive Baseline Model Evaluation & Clinical Diagnostic Error Analysis  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | scikit-learn 1.9.0 | OS: Windows 11  

---

## 1. Executive Summary

Day 9 executed the first and only formal evaluation of the validation-selected EfficientNet-B0 baseline model on the completely held-out, unblinded HAM10000 test set ($N = 1,512$ images). In strict accordance with the anti-leakage protocol, the test dataset remained completely unsealed and untouched throughout Days 1–8; the model weights, hyperparameters, and architecture were frozen prior to test inference.

### Key Milestones & Diagnostic Findings:
1. **Top-1 & Top-2 Accuracy**:
   - **Top-1 Test Accuracy**: **`75.79%`** ($1,146 / 1,512$ correct).
   - **Top-2 Test Accuracy**: **`89.55%`** ($1,354 / 1,512$ correct). In 89.55% of clinical cases, the true diagnosis was within the model's two highest-probability candidates.
2. **Macro vs. Weighted Metrics**:
   - **Macro F1-score**: **`0.5157`** | **Weighted F1-score**: **`0.7448`**.
   - The large divergence between macro F1 (`0.5157`) and weighted F1 (`0.7448`) mirrors the severe class imbalance of HAM10000, where `nv` (melanocytic nevi) accounts for 66.6% of test cases.
3. **Discriminative Capability (ROC-AUC & PR-AUC)**:
   - **Macro ROC-AUC (One-vs-Rest)**: **`0.9211`** | **Weighted ROC-AUC**: **`0.9145`**.
   - **Macro PR-AUC (Average Precision)**: **`0.5646`**.
   - Every individual diagnostic category achieved an OvR ROC-AUC exceeding $0.87$, demonstrating strong continuous ranking capability despite discrete classification challenges.
4. **Malignant / Premalignant Error Profile**:
   - Evaluated critical categories: Melanoma (`mel`), Basal Cell Carcinoma (`bcc`), and Actinic Keratoses (`akiec`).
   - Combined malignant/premalignant sensitivity was **`44.22%`** ($130 / 294$ detected).
   - **`32.65%`** ($96 / 294$) of malignant/premalignant cases were misclassified as benign melanocytic nevi (`nv`), with Melanoma exhibiting the highest false-negative rate to `nv` ($70 / 168 = 41.67\%$).
5. **Inference Performance**:
   - Total test inference duration: **`49.39s`** on CPU ($30.62$ images/second in batches of 32).
   - Single-image inference latency: **`31.64ms`** mean ($0.0316\text{s}$), well below the project target of $<3.0\text{s}$.
6. **Integrity & Test Protection**:
   - Zero test data leakage: Test split checksum unchanged (`6a5ae1b65c25d504f78bc1da2361d82c`). Zero lesion or image overlap with train/val. Zero backpropagation, retraining, or checkpoint reselection.

---

## 2. Model & Checkpoint Specification

The evaluated model is the definitive baseline checkpoint established during Day 6 and confirmed in Day 8:
- **Architecture**: EfficientNet-B0 (Torchvision `IMAGENET1K_V1` pretrained weights)
- **Classifier Head**: `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 7))`
- **Feature Extractor**: Frozen backbone (4,007,548 frozen parameters)
- **Trainable Parameters**: 8,967 head parameters (`requires_grad=False` during evaluation)
- **Checkpoint Source**: `models/checkpoints/efficientnet_b0_best.pth`
- **Checkpoint Epoch**: 14
- **Checkpoint Validation Loss**: `0.6422`
- **Checkpoint Validation Accuracy**: `77.38%`
- **Evaluation Mode**: `model.eval()` with `torch.no_grad()`

---

## 3. Test Dataset Integrity & Isolation Verification

Prior to inference, the test partition was independently audited for structural integrity:
- **File**: `reports/data/split_test.csv`
- **MD5 Hash**: `6a5ae1b65c25d504f78bc1da2361d82c` (Verified identical to Day 2 and Day 8 records)
- **Total Test Samples**: Exactly **`1,512`**
- **Unique Test Images**: Exactly **`1,512`** (0 duplicate images)
- **Unique Test Lesions**: **`1,121`**
- **Missing Images on Disk**: **`0`** (All 1,512 image files confirmed present in `data/raw/`)
- **Valid Class Labels**: All labels strictly match the 7 classes: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.

### Lesion-Level Patient Isolation Audit:
$$\text{Train Lesions} \cap \text{Test Lesions} = \emptyset \quad (0 \text{ overlap})$$
$$\text{Val Lesions} \cap \text{Test Lesions} = \emptyset \quad (0 \text{ overlap})$$
$$\text{Train Images} \cap \text{Test Images} = \emptyset \quad (0 \text{ overlap})$$
$$\text{Val Images} \cap \text{Test Images} = \emptyset \quad (0 \text{ overlap})$$

---

## 4. Overall Test Metrics

Recorded directly from execution in `reports/data/day9_metrics.json`:

| Metric | Score | Clinical / Technical Interpretation |
|:---|:---:|:---|
| **Top-1 Accuracy** | **`75.79%`** (`0.757937`) | Correct primary class assigned in 1,146 of 1,512 test cases. |
| **Top-2 Accuracy** | **`89.55%`** (`0.895503`) | Correct class present within the top 2 predicted probabilities. |
| **Macro Precision** | **`0.5744`** (`0.574358`) | Unweighted mean of positive predictive values across all 7 classes. |
| **Macro Recall** | **`0.4787`** (`0.478688`) | Unweighted mean of sensitivity across all 7 classes. |
| **Macro F1-Score** | **`0.5157`** (`0.515696`) | Balanced diagnostic measure, penalized heavily by minority classes. |
| **Weighted Precision** | **`0.7382`** (`0.738198`) | Precision weighted by class prevalence in the test set. |
| **Weighted Recall** | **`0.7579`** (`0.757937`) | Identical to Top-1 accuracy under multi-class evaluation. |
| **Weighted F1-Score** | **`0.7448`** (`0.744771`) | Prevalence-weighted harmonic mean. |
| **Test CrossEntropyLoss**| **`0.6638`** (`0.663784`) | Evaluated loss on held-out test data (compared to val loss `0.6422`). |

---

## 5. Top-1 vs. Top-2 Accuracy Analysis

- **Top-1 Accuracy**: **`75.79%`** ($1,146$ samples)
- **Top-2 Accuracy**: **`89.55%`** ($1,354$ samples)
- **Absolute Gain**: **`+13.76%`** ($208$ additional correct cases)

### Clinical Interpretation:
In dermatological triage and decision-support systems, a differential diagnosis frequently encompasses the top two candidate conditions. For $208$ test images where the model's top prediction was incorrect, the true pathology was identified as the immediate second-highest probability. This suggests that while linear head capacity under a frozen backbone limits fine boundary separation, the underlying convolutional representations capture strong semantic proximity between clinically similar lesion types.

---

## 6. Per-Class Diagnostic Classification Report

Recorded in `reports/data/day9_classification_report.csv`:

| Diagnostic Class | Full Name | Precision | Recall (Sensitivity) | F1-Score | Test Support ($N$) | Test Prevalence (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **`akiec`** | Actinic keratoses | 0.6000 | 0.4364 | 0.5053 | 55 | 3.64% |
| **`bcc`** | Basal cell carcinoma | 0.5417 | 0.5493 | 0.5455 | 71 | 4.70% |
| **`bkl`** | Benign keratosis | 0.5221 | 0.4251 | 0.4686 | 167 | 11.05% |
| **`df`** | Dermatofibroma | 0.3636 | 0.2000 | 0.2581 | 20 | 1.32% |
| **`mel`** | Melanoma | 0.4786 | 0.3988 | 0.4351 | 168 | 11.11% |
| **`nv`** | Melanocytic nevi | 0.8479 | 0.9245 | 0.8846 | 1,007 | 66.60% |
| **`vasc`** | Vascular lesions | 0.6667 | 0.4167 | 0.5128 | 24 | 1.59% |
| **Macro Average** | — | **0.5744** | **0.4787** | **0.5157** | **1,512** | 100.0% |
| **Weighted Average**| — | **0.7382** | **0.7579** | **0.7448** | **1,512** | 100.0% |

### Diagnostic Observations:
1. **Majority Class Dominance (`nv`)**:
   - Precision: $84.79\%$, Recall: $92.45\%$, F1: $0.8846$.
   - Because `nv` constitutes two-thirds of the dataset, the unweighted baseline naturally develops an inductive bias toward predicting `nv`.
2. **Melanoma Sensitivity Gap (`mel`)**:
   - Sensitivity is only **`39.88%`** ($67 / 168$). Over $60\%$ of melanoma cases failed to be identified as the top prediction.
3. **Lowest Performing Minority Class (`df`)**:
   - Dermatofibroma had the lowest recall (**`20.00%`**, $4 / 20$) and lowest F1 (**`0.2581`**). With only 20 test samples, the model lacked sufficient gradient exposure during frozen training to isolate distinct features for `df`.

---

## 7. Multi-Class ROC-AUC (One-vs-Rest)

ROC curves evaluate the model's threshold-independent discriminatory power across predicted continuous probabilities:

| Diagnostic Class | One-vs-Rest ROC-AUC | Clinical Ranking Assessment |
|:---|:---:|:---|
| **`akiec`** (Actinic keratoses) | **`0.9315`** | Excellent class separability |
| **`bcc`** (Basal cell carcinoma) | **`0.9396`** | Excellent class separability |
| **`bkl`** (Benign keratosis) | **`0.8726`** | Good separability; visual overlap with `nv` and `mel` |
| **`df`** (Dermatofibroma) | **`0.9612`** | High ranking AUC despite low discrete recall |
| **`mel`** (Melanoma) | **`0.8766`** | Good separability; lower bound among classes |
| **`nv`** (Melanocytic nevi) | **`0.9235`** | Excellent separability against non-nevi |
| **`vasc`** (Vascular lesions) | **`0.9430`** | Excellent distinct vascular spectral features |
| **Macro ROC-AUC** | **`0.9211`** | High overall multi-class ranking capability |
| **Weighted ROC-AUC**| **`0.9145`** | High prevalence-weighted separability |

- **Figure Artifact**: `reports/figures/day9_roc_curves.png`

> [!NOTE]
> The disparity between high ROC-AUC ($0.9211$) and moderate top-1 accuracy ($75.79\%$) reveals that the model's *continuous probability ranking* is robust, but the default argmax decision threshold ($\text{argmax} = 0.5$ equivalent) is heavily skewed by class prevalence. Operating threshold adjustment or class weighting in later iterations would dramatically improve minority recall.

---

## 8. Precision-Recall Analysis (Supplementary Metrics)

Precision-Recall AUC (Average Precision) provides a rigorous assessment in highly skewed distributions:

| Diagnostic Class | One-vs-Rest Average Precision (PR-AUC) | Baseline Prevalence (Chance Level) |
|:---|:---:|:---:|
| **`akiec`** | **`0.4996`** | 0.0364 (13.7× chance) |
| **`bcc`** | **`0.5704`** | 0.0470 (12.1× chance) |
| **`bkl`** | **`0.5097`** | 0.1105 (4.6× chance) |
| **`df`** | **`0.3455`** | 0.0132 (26.2× chance) |
| **`mel`** | **`0.5046`** | 0.1111 (4.5× chance) |
| **`nv`** | **`0.9605`** | 0.6660 (1.4× chance) |
| **`vasc`** | **`0.5619`** | 0.0159 (35.3× chance) |
| **Macro PR-AUC** | **`0.5646`** | — |

- **Figure Artifact**: `reports/figures/day9_precision_recall_curves.png`
- For all minority classes, the model's Average Precision is $4.5\times$ to $35.3\times$ higher than random chance, verifying substantial discriminative signal.

---

## 9. Confusion Matrix Analysis

The full $7 \times 7$ contingency matrix is recorded below:

### Raw-Count Confusion Matrix ($N = 1,512$):

| True \ Pred | `akiec` | `bcc` | `bkl` | `df` | `mel` | `nv` | `vasc` | Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`akiec`** | **24** | 5 | 10 | 1 | 6 | 9 | 0 | 55 |
| **`bcc`** | 3 | **39** | 6 | 1 | 4 | 17 | 1 | 71 |
| **`bkl`** | 4 | 10 | **71** | 1 | 20 | 60 | 1 | 167 |
| **`df`** | 2 | 0 | 3 | **4** | 3 | 7 | 1 | 20 |
| **`mel`** | 6 | 5 | 19 | 1 | **67** | 70 | 0 | 168 |
| **`nv`** | 1 | 8 | 27 | 2 | 36 | **931** | 2 | 1,007 |
| **`vasc`** | 0 | 5 | 0 | 1 | 4 | 4 | **10** | 24 |
| **Total Pred**| 40 | 72 | 136 | 11 | 140 | 1,098 | 15 | 1,512 |

- **Figure Artifact (Raw)**: `reports/figures/day9_confusion_matrix.png`
- **Figure Artifact (Row-Normalized)**: `reports/figures/day9_confusion_matrix_normalized.png`

---

## 10. Malignant / Premalignant Diagnostic Error Analysis

In dermatopathology, false negatives on malignant lesions constitute the highest clinical risk:

### Breakdown for Critical Classes:
1. **Melanoma (`mel`)**:
   - Total Support: $168$
   - Correct Predictions: $67$ ($39.88\%$ recall)
   - **False Negatives**: **`101`** ($60.12\%$)
   - **Misclassified as Benign Nevi (`nv`)**: **`70`** cases ($41.67\%$ of all Melanomas)
   - Misclassified as Benign Keratosis (`bkl`): $19$ cases ($11.31\%$)
   - Misclassified as Actinic Keratoses (`akiec`): $6$ cases ($3.57\%$)
2. **Basal Cell Carcinoma (`bcc`)**:
   - Total Support: $71$
   - Correct Predictions: $39$ ($54.93\%$ recall)
   - **False Negatives**: **`32`** ($45.07\%$)
   - **Misclassified as Benign Nevi (`nv`)**: **`17`** cases ($23.94\%$ of all BCC)
   - Misclassified as `bkl`: $6$ cases ($8.45\%$)
   - Misclassified as `mel`: $4$ cases ($5.63\%$)
3. **Actinic Keratoses / Intraepithelial Carcinoma (`akiec`)**:
   - Total Support: $55$
   - Correct Predictions: $24$ ($43.64\%$ recall)
   - **False Negatives**: **`31`** ($56.36\%$)
   - **Misclassified as Benign Nevi (`nv`)**: **`9`** cases ($16.36\%$)
   - Misclassified as `bkl`: $10$ cases ($18.18\%$)

### Aggregated Malignant Risk Summary:
- **Total Malignant / Premalignant Cases**: **`294`** ($19.44\%$ of test set)
- **Correctly Classified**: **`130`**
- **Total False Negatives**: **`164`**
- **Malignant Sensitivity**: **`44.22%`**
- **Total Malignant Misclassified as Benign `nv`**: **`96`** ($32.65\%$ of all malignant samples)

### Top Diagnostic Confusion Pairs Across All Errors:
1. `mel` $\rightarrow$ `nv`: **70** occurrences ($19.13\%$ of all 366 errors)
2. `bkl` $\rightarrow$ `nv`: **60** occurrences ($16.39\%$)
3. `nv` $\rightarrow$ `mel`: **36** occurrences ($9.84\%$)
4. `nv` $\rightarrow$ `bkl`: **27** occurrences ($7.38\%$)
5. `bkl` $\rightarrow$ `mel`: **20** occurrences ($5.46\%$)
6. `mel` $\rightarrow$ `bkl`: **19** occurrences ($5.19\%$)
7. `bcc` $\rightarrow$ `nv`: **17** occurrences ($4.64\%$)
8. `akiec` $\rightarrow$ `bkl`: **10** occurrences ($2.73\%$)
9. `bkl` $\rightarrow$ `bcc`: **10** occurrences ($2.73\%$)
10. `akiec` $\rightarrow$ `nv`: **9** occurrences ($2.46\%$)

---

## 11. Model Confidence Dynamics

Evaluated across the 1,512 test predictions:

| Prediction Group | Sample Count | Mean Confidence | Median Confidence | Std Dev | Min Conf | Max Conf |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Correct Predictions** | 1,146 | **`83.90%`** | **`92.04%`** | 0.1819 | 28.23% | 100.0% |
| **Incorrect Predictions** | 366 | **`56.65%`** | **`53.06%`** | 0.1703 | 23.92% | 98.86% |

### High-Confidence Misclassifications ($\ge 0.80$):
- **Total High-Confidence Errors**: **`45`** cases ($12.30\%$ of all errors, $2.98\%$ of the test set).
- Breakdown by true diagnostic class:
  - `bkl`: 19 cases
  - `mel`: 13 cases
  - `bcc`: 5 cases
  - `akiec`: 4 cases
  - `nv`: 3 cases
  - `df`: 1 case
- **Finding**: While the model's average confidence on errors is significantly lower ($56.65\%$ vs $83.90\%$), 45 cases exhibited overconfidence ($\ge 80\%$), including 13 Melanomas misclassified as benign nevi.

---

## 12. Representative Misclassifications Grid

A curated 16-sample diagnostic panel was compiled and plotted:
- **Artifact Path**: `reports/figures/day9/errors/day9_misclassifications_grid.png`
- **Selection Criteria**:
  - Critical malignant-to-benign errors (`mel -> nv`, `bcc -> nv`, `akiec -> nv`)
  - High-confidence incorrect predictions ($\text{Conf} \ge 80\%$)
  - Rare minority class errors (`df`, `vasc`)
  - Frequent keratosis/nevus confusion (`bkl -> nv`, `nv -> bkl`)
- **Key Observation**: Visually ambiguous lesions presenting atypical pigmentation, irregular borders, or low contrast against skin tones are frequently assigned to the majority prior (`nv`).

---

## 13. Validation vs. Test Performance Comparison

| Metric | Validation Set (Day 6/8 Best Epoch 14) | Held-Out Test Set (Day 9) | Generalization Delta |
|:---|:---:|:---:|:---:|
| **Loss** | **`0.6422`** | **`0.6638`** | $+0.0216$ (Minimal degradation) |
| **Top-1 Accuracy** | **`77.38%`** (Peak $77.65\%$) | **`75.79%`** | $-1.59\%$ (Well within standard margin) |
| **Top-2 Accuracy** | Not calculated | **`89.55%`** | — |
| **Macro F1-Score** | Not calculated | **`0.5157`** | — |
| **Weighted F1-Score**| Not calculated | **`0.7448`** | — |
| **Sample Size ($N$)**| 1,521 images | 1,512 images | Independent splits |

### Generalization Assessment:
The model demonstrates **strong generalization fidelity**. Test accuracy ($75.79\%$) is within $1.59\%$ of the validation loss-selected checkpoint accuracy ($77.38\%$), and test loss ($0.6638$) increased by only $0.0216$. There is no evidence of overfitting or catastrophic performance collapse on unseen patient lesions.

---

## 14. Inference Latency & Benchmark

Measured on host machine CPU (Python 3.14.5, PyTorch 2.10.0+cpu, Windows 11):
- **Batch Size**: 32
- **Total Test Samples**: 1,512 images (48 batches)
- **Total Batch Inference Duration**: **`49.39s`**
- **Batch Throughput**: **`30.62 images/second`**
- **Single-Image Latency (Timed across 30 runs with 5 warmups)**:
  - **Mean Latency**: **`31.64 ms`** ($0.0316\text{s}$)
  - **Median Latency**: **`30.60 ms`** ($0.0306\text{s}$)
- **Requirement Verification**: Both single-image latency ($31.64\text{ms}$) and amortized batch time ($32.66\text{ms}$) are orders of magnitude below the project's $< 3.0\text{s}$ requirement.

---

## 15. Dataset Limitations

1. **Severe Imbalance**: `nv` accounts for $66.6\%$ of the test set, while `df` ($1.32\%$) and `vasc` ($1.59\%$) have negligible support ($20$ and $24$ images).
2. **Melanocytic Bias**: Lesions sharing pigmentary networks (nevi, dysplastic nevi, early superficial melanomas) present subtle microscopic distinctions that a frozen $224 \times 224$ ImageNet backbone cannot reliably resolve.
3. **Demographic / Skin Type Shift**: HAM10000 images originate predominantly from Austrian and Australian dermatology clinics with fair-skinned populations (Fitzpatrick types I–III). Performance on darker skin phototypes (Fitzpatrick IV–VI) is uncharacterized.

---

## 16. Medical & Ethical Disclaimer

> [!CAUTION]
> **RESEARCH & EDUCATIONAL DISCLAIMER**  
> MediScan is strictly an academic, educational, and research project. It is **NOT a medical device**, **NOT a diagnostic tool**, and **NOT certified by any regulatory agency (FDA, CE, EMA)** for clinical decision-making.  
> - Predictions must never be used to diagnose, triage, treat, or manage dermatological conditions in patients.
> - The model exhibits a **`60.12%` false-negative rate on melanoma** ($101 / 168$ missed as the primary diagnosis, with $70$ misclassified as benign nevi). Deploying such a model in a clinical setting would create catastrophic diagnostic risk.
> - Any medical assessment requires consultation with a qualified board-certified dermatologist and histopathological biopsy.

---

## 17. Test Set Usage Audit

A comprehensive code and execution audit confirms strict test set protection:
- The test set was **never exposed** to gradient computation (`torch.no_grad()` enforced).
- The test set was **never used** for optimizer updates, learning rate scheduling, or loss backpropagation.
- No model architecture, dropout rate, or optimizer setting was modified after seeing test metrics.
- The evaluated checkpoint (`efficientnet_b0_best.pth`, Epoch 14) remained strictly unchanged and unmodified.
- The test split file `split_test.csv` was preserved bit-for-bit (MD5: `6a5ae1b65c25d504f78bc1da2361d82c`).

---

## 18. Generated Artifacts & File Register

All artifacts have been generated, verified, and saved to disk:

| File Type | Path | Size / Details |
|:---|:---|:---|
| **Predictions CSV** | `reports/data/day9_test_predictions.csv` | 1,512 rows, 12 columns |
| **Classification Report** | `reports/data/day9_classification_report.csv` | 7 classes + support/F1 |
| **Complete Metrics JSON**| `reports/data/day9_metrics.json` | Complete quantitative audit |
| **Raw Confusion Matrix** | `reports/figures/day9_confusion_matrix.png` | 300 DPI publication heatmap |
| **Norm Confusion Matrix**| `reports/figures/day9_confusion_matrix_normalized.png` | 300 DPI row-normalized heatmap |
| **ROC Curves Plot** | `reports/figures/day9_roc_curves.png` | 7 classes + chance + macro AUC |
| **PR Curves Plot** | `reports/figures/day9_precision_recall_curves.png` | 7 classes + macro AP |
| **Error Grid Plot** | `reports/figures/day9/errors/day9_misclassifications_grid.png`| 16 curated error examples |
| **Evaluation Module** | `src/evaluation/` | Modular inference, metrics, viz, error |
| **Runner Script** | `scripts/evaluate_day9.py` | Orchestration pipeline |
| **Test Suite** | `tests/test_day9_evaluation.py` | 29 unit tests |

---

## 19. Automated Test Suite Verification

Execution of the Day 9 test suite:
```bash
pytest tests/test_day9_evaluation.py -v
```
**Result: 29 / 29 PASSED (100%)**
- `TestDatasetIntegrity`: 6 passed
- `TestPredictions`: 7 passed
- `TestMetrics`: 7 passed
- `TestOutputArtifacts`: 8 passed
- `TestProtectionAudit`: 1 passed

---

## 20. Final Conclusion & Status Decision

# **PASS — DAY 9 COMPLETE**

Day 9 has fulfilled all requirements for comprehensive baseline evaluation and error analysis with verified, unblinded test results. The baseline model is thoroughly benchmarked and characterized. The project is completely prepared to proceed to **Day 10 (Grad-CAM Explainability)**.

### Verification Summary Table:

| Area | Status | Evidence |
|:---|:---:|:---|
| **Test Integrity** | **PASS** | Exactly 1,512 images, 0 missing, 0 duplicate, 0 train/val overlap, MD5 unaltered (`6a5ae1b65c25d504f78bc1da2361d82c`). |
| **Checkpoint** | **PASS** | Loaded fixed `efficientnet_b0_best.pth` (Epoch 14, Val Loss 0.6422). No weight modification. |
| **Test Inference** | **PASS** | Complete inference on all 1,512 test samples with `torch.no_grad()`. Probabilities sum to 1.0. |
| **Top-1 Accuracy** | **PASS** | Measured at **75.79%** (1,146 / 1,512). |
| **Top-2 Accuracy** | **PASS** | Measured at **89.55%** (1,354 / 1,512). |
| **Per-Class Metrics**| **PASS** | Complete precision, recall, F1, and support for all 7 classes in `day9_classification_report.csv`. |
| **ROC-AUC** | **PASS** | Macro ROC-AUC **0.9211**, Weighted **0.9145**, per-class values in $[0.8726, 0.9612]$. |
| **PR Analysis** | **PASS** | Macro PR-AUC **0.5646**, per-class APs generated and plotted. |
| **Confusion Matrix** | **PASS** | $7 \times 7$ raw and normalized heatmaps generated at 300 DPI. |
| **Error Analysis** | **PASS** | Detailed malignant breakdown: $96 / 294$ malignant as `nv`, $70 / 168$ `mel` as `nv`. Top 10 pairs ranked. |
| **Confidence Analysis**| **PASS** | Correct mean $83.90\%$, Incorrect mean $56.65\%$. 45 high-confidence errors documented. |
| **Test Protection** | **PASS** | Zero backprop, zero retraining, zero tuning, test used only for post-hoc evaluation. |
| **Inference Benchmark**| **PASS** | Measured: 30.62 imgs/s batch throughput, 31.64ms single-image latency ($< 3.0$s requirement satisfied). |
| **Tests** | **PASS** | 29 / 29 tests passed in `test_day9_evaluation.py`. Full regression suite passing. |
| **Overall** | **PASS** | **DAY 9 COMPLETE**. Ready for Day 10. |
