# MediScan — Day 10 Completion Report: Grad-CAM Explainability & Visual Heatmaps

**Author:** Pentapalli Charan  
**Date:** September 14, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 10 — Grad-CAM Explainability & Visual Diagnostic Heatmaps  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | OpenCV 4.13.0 | OS: Windows 11  

---

## 1. Executive Summary & Objective

Day 10 established a reliable, hook-based **Grad-CAM (Gradient-weighted Class Activation Mapping)** explainability pipeline for the fixed, validation-selected EfficientNet-B0 baseline model. Operating strictly as a post-hoc diagnostic investigation, Day 10 generated spatial activation heatmaps across representative test cases without altering model weights, retraining, or modifying test partitions.

### Key Milestones & Findings:
1. **Target Layer Verification**:
   - The final convolutional block of the baseline architecture was verified as `model.features[8]` (`Conv2dNormActivation`), outputting feature activations of shape `[1, 1280, 7, 7]`.
2. **Hook-Based Grad-CAM Engine**:
   - Implemented clean PyTorch forward and backward hook lifecycle management in `src/explainability/gradcam.py` with automatic detaching and zero gradient accumulation.
3. **Comprehensive Sample Coverage (12 Distinct Case Visualizations)**:
   - Correctly identified malignant lesions: Melanoma (`mel`), Basal Cell Carcinoma (`bcc`), Actinic Keratoses (`akiec`).
   - Diagnostic errors on malignant lesions: Melanoma misclassified as Nevus (`mel -> nv`), BCC as Nevus (`bcc -> nv`), AKIEC as Nevus (`akiec -> nv`).
   - Correct and overcalled benign cases: Nevus correctly predicted (`nv -> nv`), Nevus overcalled as Melanoma (`nv -> mel`).
   - Rare classes: Dermatofibroma (`df`) and Vascular lesion (`vasc`).
   - High-confidence error: Benign Keratosis misclassified as Nevus with $93.3\%$ confidence.
   - Dual-target comparison: Contrasting Predicted-Class CAM vs. True-Class CAM on identical lesion imagery.
4. **Gradient Safety & Parameter Immutability**:
   - Model parameters were snapshotted before Grad-CAM and verified bitwise identical after all 12 forward/backward iterations.
5. **Quality Artifacts Generated**:
   - 12 individual 3-panel triplet figures saved to `reports/figures/gradcam/`.
   - Main consolidated grid saved to `reports/figures/day10_gradcam_examples.png`.
   - Focused malignant error grid saved to `reports/figures/day10_gradcam_malignant_errors.png`.
   - Complete numerical verification recorded in `reports/data/day10_validation.json` and `reports/data/day10_gradcam_samples.csv`.
6. **Automated Verification**:
   - 16 / 16 unit tests passed in `tests/test_day10_gradcam.py` (100%).

---

## 2. Target Layer Architecture & Selection

Before implementing Grad-CAM, the exact instantiated module hierarchy of `models/checkpoints/efficientnet_b0_best.pth` was inspected:
- **Base Architecture**: `torchvision.models.efficientnet_b0` (weights: `IMAGENET1K_V1`)
- **Module Decomposition**:
  - `model.features`: `torch.nn.Sequential` with 9 blocks (indices 0 through 8).
  - Blocks 0 to 7: Initial convolution and inverted residual blocks (MBConv).
  - Block 8: Final convolutional feature expansion block:
    ```
    model.features[8]: Conv2dNormActivation(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, bias=True)
      (2): SiLU(inplace=True)
    )
    ```
  - `model.avgpool`: AdaptiveAvgPool2d(output_size=1)
  - `model.classifier`: Sequential(Dropout(p=0.2), Linear(1280, 7))
- **Selected Target Layer**: `model.features[8]`
- **Rationale**: Layer 8 captures the richest semantic representations (1280 feature channels at $7 \times 7$ spatial resolution) immediately preceding spatial pooling and the classification head.

---

## 3. Grad-CAM Mathematical Formulation & Hook Mechanism

Grad-CAM computes the importance of each feature map $k$ in target layer $A$ for a chosen target class $c$:

### 1. Neuron Importance Weights:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i,j}^k}$$
where $y^c$ is the unnormalized logit for class $c$, $A_{i,j}^k$ is the activation value at spatial coordinate $(i, j)$ in channel $k$, and $Z = H \times W = 7 \times 7 = 49$.

### 2. Rectified Linear Combination:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k=1}^{1280} \alpha_k^c A^k\right)$$
The $\text{ReLU}$ non-linearity ensures that only features positively correlating with class $c$ are retained.

### 3. Normalization & Spatial Interpolation:
The raw $7 \times 7$ activation map is normalized into $[0.0, 1.0]$:
$$\tilde{L}^c = \frac{L^c - \min(L^c)}{\max(L^c) - \min(L^c) + \epsilon}$$
and upsampled to the input image resolution ($224 \times 224$) via bilinear interpolation.

---

## 4. Checkpoint & Preprocessing Specifications

- **Model Checkpoint**: `models/checkpoints/efficientnet_b0_best.pth` (Epoch 14, Val Loss: 0.6422).
- **Frozen Backbone**: The feature extractor parameters remain fixed.
- **Preprocessing Pipeline**:
  - Image RGB conversion.
  - Deterministic bilinear resize to $224 \times 224$.
  - ImageNet normalization: Mean = $[0.485, 0.456, 0.406]$, Std = $[0.229, 0.224, 0.225]$.
  - Input tensor shape: $[1, 3, 224, 224]$, float32.

---

## 5. Sample Selection Methodology

Samples were selected deterministically from the Day 9 held-out test predictions (`reports/data/day9_test_predictions.csv`), satisfying all required clinical and technical categories:

| ID | Sample Case | Image ID | True Class | Pred Class | Conf (%) | Target Class | Output Artifact |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Correct Melanoma | `ISIC_0033470` | `mel` | `mel` | 84.8% | `mel` | `mel_correct_ISIC_0033470.png` |
| 2A | Melanoma Missed as Nevus (Pred Target) | `ISIC_0032653` | `mel` | `nv` | 91.1% | `nv` | `mel_to_nv_predtarget_ISIC_0032653.png` |
| 2B | Melanoma Missed as Nevus (True Target) | `ISIC_0032653` | `mel` | `nv` | 91.1% | `mel` | `mel_to_nv_truetarget_ISIC_0032653.png` |
| 3 | Correct BCC | `ISIC_0026213` | `bcc` | `bcc` | 89.2% | `bcc` | `bcc_correct_ISIC_0026213.png` |
| 4 | BCC Missed as Nevus | `ISIC_0027609` | `bcc` | `nv` | 88.3% | `nv` | `bcc_to_nv_ISIC_0027609.png` |
| 5 | Correct AKIEC | `ISIC_0029781` | `akiec` | `akiec` | 81.3% | `akiec` | `akiec_correct_ISIC_0029781.png` |
| 6 | AKIEC Missed as Nevus | `ISIC_0033536` | `akiec` | `nv` | 85.9% | `nv` | `akiec_to_nv_ISIC_0033536.png` |
| 7 | Correct Melanocytic Nevus | `ISIC_0024438` | `nv` | `nv` | 99.8% | `nv` | `nv_correct_ISIC_0024438.png` |
| 8 | Nevus Overcalled as Melanoma | `ISIC_0027379` | `nv` | `mel` | 74.2% | `mel` | `nv_to_mel_ISIC_0027379.png` |
| 9 | Rare Class: Correct DF | `ISIC_0030579` | `df` | `df` | 68.3% | `df` | `df_correct_ISIC_0030579.png` |
| 10 | Rare Class: Correct VASC | `ISIC_0031201` | `vasc` | `vasc` | 94.7% | `vasc` | `vasc_correct_ISIC_0031201.png` |
| 11 | High-Confidence Error (BKL to NV) | `ISIC_0025804` | `bkl` | `nv` | 93.3% | `nv` | `high_conf_bkl_to_nv_ISIC_0025804.png` |

---

## 6. Correct Prediction Heatmap Analysis

For true positive predictions, Grad-CAM heatmaps demonstrate focused spatial localization over the core lesion structures:
1. **Melanoma (`ISIC_0033470`, Conf: $84.8\%$)**:
   - Activation is tightly centered on the irregular, hyperpigmented central macule and its asymmetric borders. Peripheral unaffected skin shows near-zero activation ($< 0.10$).
2. **Basal Cell Carcinoma (`ISIC_0026213`, Conf: $89.2\%$)**:
   - Strong focal activation concentrated over the translucent nodular central zone and telangiectatic border.
3. **Actinic Keratoses (`ISIC_0029781`, Conf: $81.3\%$)**:
   - Activation maps follow the scaly, erythematous central plaque.
4. **Melanocytic Nevus (`ISIC_0024438`, Conf: $99.8\%$)**:
   - Symmetric, well-bounded activation localized directly over the uniform pigment network.
5. **Vascular Lesion (`ISIC_0031201`, Conf: $94.7\%$)**:
   - Intense circumscribed activation over the dark-red vascular lacunae.

---

## 7. Diagnostic Error Heatmap Analysis (Malignant Missed as Nevus)

Day 9 identified that $32.65\%$ of malignant/premalignant cases were misclassified as benign nevi (`nv`). Grad-CAM reveals the spatial mechanics behind these failures:
1. **Melanoma Missed as Nevus (`ISIC_0032653`, True: `mel`, Pred: `nv`, Conf: $91.1\%$)**:
   - **Predicted-Target CAM (`nv`)**: Activation spans broadly across homogenous background pigment, treating the lesion as a benign uniform network.
   - **True-Target CAM (`mel`)**: When queried for Melanoma, the activation shifts distinctly toward the darker, irregular peripheral notch, confirming that the network did register atypical morphology in the deeper layers, but the linear classifier gave stronger weight to the nevus prior.
2. **BCC Missed as Nevus (`ISIC_0027609`, True: `bcc`, Pred: `nv`, Conf: $88.3\%$)**:
   - The network focused primarily on the central pigment rather than the translucent periphery, leading to a benign nevus classification.
3. **AKIEC Missed as Nevus (`ISIC_0033536`, True: `akiec`, Pred: `nv`, Conf: $85.9\%$)**:
   - Heatmap highlights the pigmented component of the lesion rather than the subtle keratotic texture.

---

## 8. High-Confidence Error Analysis

- **Case**: `ISIC_0025804` (True: `bkl` / Benign Keratosis, Pred: `nv`, Conf: **`93.3%`**).
- **Heatmap Behavior**: The activation is concentrated squarely on the central globule, completely ignoring the verrucous, stuck-on border characteristics typical of seborrheic keratosis.
- **Finding**: High confidence in errors is driven by strong activation on uniform pigment patterns that mimic canonical nevi, reinforcing that model confidence reflects activation strength along learned linear projections, not clinical certainty.

---

## 9. True-Class vs. Predicted-Class Grad-CAM Contrasts

Comparing target policies on misclassified cases (`ISIC_0032653`):
- **Predicted Target (`nv`)**: Global, diffuse activation over the general pigmented area.
- **True Target (`mel`)**: Focal, concentrated activation over the localized asymmetric region.
- **Insight**: Grad-CAM target policy is a critical diagnostic tool. Generating both targets confirms that the feature extractor captures multi-class evidence, and errors frequently stem from threshold/head prioritization rather than complete blindness to atypical features.

---

## 10. Technical & Clinical Limitations of Grad-CAM

> [!CAUTION]
> **INTERPRETABILITY & ETHICAL DISCLAIMER**  
> 1. **Not a Clinical Diagnostic Tool**: Grad-CAM visualizes mathematical gradient correlations. It does **not** provide clinical reasoning, proof of malignancy, or histopathological confirmation.
> 2. **Coarse Spatial Resolution**: Because the final feature map is $7 \times 7$ pixels, upsampling to $224 \times 224$ introduces spatial blurriness; small sub-millimeter dermoscopic criteria (e.g. subtle pigment dots or atypical pseudopods) cannot be resolved precisely.
> 3. **Faithfulness Caveats**: High activation indicates feature importance for a specific linear logit, but does not guarantee that the highlighted pixels are causal drivers of the underlying biological pathology.

---

## 11. Numerical Validation & Gradient Safety Audit

Recorded in `reports/data/day10_validation.json`:
- **Dimensions**: All generated CAMs have exact shape $[224, 224]$.
- **Finiteness**: Zero NaN values, zero Inf values across all 12 executions.
- **Range Bounds**: All CAM values strictly bounded in $[0.0, 1.0]$ with $\min = 0.0$ and $\max = 1.0$.
- **Latency**: Mean Grad-CAM generation latency per sample was **`0.071s`** ($71\text{ms}$).
- **Parameter Immutability**:
  - Model parameters snapshotted prior to execution.
  - Bitwise comparison $\theta_{\text{after}} \equiv \theta_{\text{before}}$ evaluated via `torch.equal()`: **PASSED (100% identical)**.
  - Zero backpropagation into parameters, zero optimizer calls.

---

## 12. Automated Test Suite Results

Execution of the Day 10 test suite:
```bash
pytest tests/test_day10_gradcam.py -v
```
**Result: 16 / 16 PASSED (100%)**
- `TestTargetLayerArchitecture`: 2 passed
- `TestGradCAMNumericalValidation`: 4 passed
- `TestHookLifecycle`: 2 passed
- `TestOverlayGeneration`: 1 passed
- `TestWeightImmutability`: 1 passed
- `TestOutputArtifacts`: 6 passed

---

## 13. Generated Artifacts Register

| Artifact Type | Path | Details |
|:---|:---|:---|
| **Main Consolidated Grid** | `reports/figures/day10_gradcam_examples.png` | 10-sample comparison grid |
| **Malignant Error Grid** | `reports/figures/day10_gradcam_malignant_errors.png` | Focused diagnostic error panel |
| **Individual Triplet Figures** | `reports/figures/gradcam/*.png` | 12 high-res triplet figures (300 DPI) |
| **Sample Register CSV** | `reports/data/day10_gradcam_samples.csv` | Full metadata and filename mapping |
| **Numerical Validation JSON** | `reports/data/day10_validation.json` | Complete shape and finiteness audit |
| **Explainability Module** | `src/explainability/` | Modular Grad-CAM engine and plotting |
| **Runner Script** | `scripts/run_day10_gradcam.py` | Orchestration runner |
| **Test Suite** | `tests/test_day10_gradcam.py` | 16 automated tests |

---

## 14. Final Conclusion & Status Decision

# **PASS — DAY 10 COMPLETE**

Grad-CAM explainability has been successfully implemented, numerically validated, and verified on the baseline EfficientNet-B0 model. Model weights remain bitwise identical, Day 9 test evaluations remain untouched, and the project is fully prepared to proceed to **Day 11 (ResNet50 Model Comparison)**.

### Verification Summary Table:

| Area | Status | Evidence |
|:---|:---:|:---|
| **Target Layer** | **PASS** | Verified at `model.features[8]` (`Conv2dNormActivation`, 1280 channels, $7 \times 7$). |
| **Grad-CAM Computation** | **PASS** | Hook-based pipeline captures activations and gradients, applies GAP, ReLU, and normalization. |
| **Numerical Validation** | **PASS** | All CAMs verified $[224, 224]$, finite, bounded in $[0.0, 1.0]$. Recorded in `day10_validation.json`. |
| **Correct Examples** | **PASS** | Generated for `mel`, `bcc`, `akiec`, `nv`, `df`, `vasc`. |
| **Error Examples** | **PASS** | Generated for `mel -> nv`, `bcc -> nv`, `akiec -> nv`, and `nv -> mel`. |
| **High-Confidence Errors** | **PASS** | Evaluated on `bkl -> nv` ($93.3\%$) and `mel -> nv` ($91.1\%$). |
| **Malignant/Premalignant**| **PASS** | Dedicated analysis and focused grid saved to `day10_gradcam_malignant_errors.png`. |
| **Visualization** | **PASS** | 12 individual 3-panel triplets in `reports/figures/gradcam/` and 2 consolidated grids. |
| **Weight Integrity** | **PASS** | Parameters verified bitwise identical before and after execution (`verify_model_weights_unchanged`). |
| **Tests** | **PASS** | 16 / 16 passed in `test_day10_gradcam.py`. Regression test suite passing. |
| **Overall** | **PASS** | **DAY 10 COMPLETE**. Ready for Day 11. |
