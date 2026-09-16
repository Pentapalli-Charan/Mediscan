# MediScan — Day 11 Completion Report: ResNet-50 Model Comparison & Architecture Benchmark

**Author:** Pentapalli Charan  
**Date:** September 15, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 11 — ResNet-50 Model Comparison & Architecture Benchmark  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | OS: Windows 11  

---

## 1. Executive Summary & Objective

Day 11 implemented **ResNet-50** as a secondary transfer-learning architecture to perform a controlled, fair, and reproducible benchmark against the primary **EfficientNet-B0** baseline. Both architectures were trained on the exact same train/validation partitions under identical optimization settings, data augmentation, input resolution ($224 \times 224$), and loss formulations.

### Key Milestones & Diagnostic Findings:
1. **ResNet-50 Architecture**:
   - Backbone: ImageNet-1K pretrained ResNet-50 (`ResNet50_Weights.IMAGENET1K_V1`).
   - Classifier: `nn.Sequential(nn.Dropout(p=0.2), nn.Linear(2048, 7))` replacing the default 1000-class fully connected layer.
   - Frozen feature extractor ($23,508,032$ parameters), with only $14,343$ parameters in the classification head trainable ($0.061\%$).
2. **Controlled Parameter Comparison**:
   - **EfficientNet-B0**: **4,016,515** total parameters ($8,967$ trainable).
   - **ResNet-50**: **23,522,375** total parameters ($14,343$ trainable).
   - ResNet-50 is **`5.86× larger`** in total parameter count and has $1.60\times$ more trainable linear head parameters.
3. **Controlled Validation Performance Benchmark (Epoch 1 Matched Protocol)**:
   - **EfficientNet-B0**: Validation Loss = **`0.8379`**, Validation Accuracy = **`72.32%`** (Train Loss: 0.9528, Train Acc: 69.13%).
   - **ResNet-50**: Validation Loss = **`0.7646`**, Validation Accuracy = **`72.52%`** (Train Loss: 0.9035, Train Acc: 69.22%).
   - **Finding**: ResNet-50 converged to a slightly lower validation loss ($-0.0733$) and marginally higher accuracy ($+0.20\%$) on Epoch 1.
4. **Computational Cost & CPU Throughput (The Decisive Trade-Off)**:
   - **EfficientNet-B0 Epoch Duration**: **`259.35s`** ($\approx 4.3$ minutes).
   - **ResNet-50 Epoch Duration**: **`5,265.4s`** ($\approx 87.76$ minutes).
   - ResNet-50 is **`20.3× slower per epoch`** on host CPU. A full 15-epoch training run on CPU would require $\approx 22$ hours.
   - For a microscopic accuracy gain of $+0.20\%$, the $20\times$ computational overhead makes ResNet-50 deeply impractical for CPU-bound medical edge deployment.
5. **Candidate Grad-CAM Target Layer**:
   - Verified candidate target layer for ResNet-50: **`model.layer4[-1]`** (final `Bottleneck` block, outputting 2,048 feature channels at $7 \times 7$ spatial resolution).
6. **Strict Test Set Isolation**:
   - Zero test data was loaded, evaluated, or used for model selection during Day 11. The test split (`split_test.csv`) and Day 9 test results are strictly preserved.
7. **Automated Verification**:
   - 21 / 21 unit tests passed in `tests/test_day11_resnet50.py` (100%).

---

## 2. ResNet-50 Architecture Overview

Implemented in `src/models/resnet50.py`:
- **Backbone**: `torchvision.models.resnet50` with residual skip connections.
- **Input Dimension**: $[B, 3, 224, 224]$ RGB.
- **Feature Extractor Blocks**:
  - `conv1` ($7 \times 7$ conv, 64 channels, stride 2) + `bn1` + `relu` + `maxpool`
  - `layer1`: 3 Bottleneck blocks ($256$ channels, $56 \times 56$)
  - `layer2`: 4 Bottleneck blocks ($512$ channels, $28 \times 28$)
  - `layer3`: 6 Bottleneck blocks ($1024$ channels, $14 \times 14$)
  - `layer4`: 3 Bottleneck blocks ($2048$ channels, $7 \times 7$)
  - `avgpool`: AdaptiveAvgPool2d(output_size=(1, 1))
- **Linear Classification Head**:
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(p=0.2, inplace=False),
      nn.Linear(in_features=2048, out_features=7, bias=True)
  )
  ```
- **Output**: Raw logits $[B, 7]$ without internal softmax.

---

## 3. Parameter Accounting: EfficientNet-B0 vs. ResNet-50

| Architectural Metric | EfficientNet-B0 (Baseline) | ResNet-50 (Comparison) | Ratio (ResNet-50 / EfficientNet) |
|:---|:---:|:---:|:---:|
| **Total Parameters** | **4,016,515** | **23,522,375** | **`5.86×`** |
| **Trainable Parameters** | **8,967** | **14,343** | **`1.60×`** |
| **Frozen Parameters** | **4,007,548** | **23,508,032** | **`5.87×`** |
| **Trainable Fraction** | 0.2233% | 0.0610% | — |
| **Input Feature Dimension**| 1,280 | 2,048 | 1.60× |
| **Feature Map Channels** | 1,280 | 2,048 | 1.60× |
| **Pretrained Weights** | `IMAGENET1K_V1` | `IMAGENET1K_V1` | Matched |

---

## 4. Controlled Training Configuration & Fairness Protocol

To ensure absolute scientific fairness, all training variables were matched:
- **Dataset Splits**:
  - Train: `split_train.csv` (6,982 images, 5,228 unique lesions).
  - Val: `split_val.csv` (1,521 images, 1,121 unique lesions).
  - Test: `split_test.csv` (1,512 images — **Untouched**).
- **Data Transforms**:
  - Train: Horizontal/Vertical Flips, Rotate, ShiftScaleRotate, Brightness/Contrast, ImageNet Normalization ($224 \times 224$).
  - Val: Deterministic Resize ($224 \times 224$), ImageNet Normalization.
- **Batch Size**: 32 (both models).
- **Optimizer**: `torch.optim.Adam(lr=1e-3, weight_decay=0.0)`.
- **Loss Function**: `nn.CrossEntropyLoss()` (pure unweighted baseline).
- **Scheduler**: `ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)`.
- **Early Stopping**: `EarlyStopping(patience=3, min_delta=0.001, mode='min')`.
- **Random Seed**: 42.

---

## 5. Compute Constraint & CPU Benchmarking

Under project Section 6 guidelines (*"The current environment is CPU-only and training is expensive. Before beginning a long run, perform a small sanity benchmark to estimate ResNet50 training time. Document the measured benchmark, explain the limitation, and determine a scientifically defensible comparison"*):

### Measured Benchmark:
- Initial mini-batch throughput test: $1.895\text{s}$ per batch of 32 on CPU.
- Full Epoch 1 execution (219 training batches + 48 validation batches): **`5,265.4s`** (**`87.76 minutes`**).
- Extrapolation:
  - 5 Epochs: $\approx 7.3$ hours.
  - 15 Epochs: $\approx 22.0$ hours.
- **Scientific Resolution**: Executing a 1-epoch controlled benchmark against Epoch 1 of EfficientNet-B0 ($259.35\text{s}$) captures initial gradient trajectories, convergence rates, and validation loss dynamics under identical data and optimization conditions without fabricating metrics.

---

## 6. Training & Validation Results

Recorded in `reports/data/day11_resnet50_training_history.csv` and `reports/data/day11_model_comparison.csv`:

### Epoch-by-Epoch Dynamics (ResNet-50):
| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate | Epoch Duration |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **0.9035** | **69.22%** | **0.7646** | **72.52%** | 0.0010 | **5,265.4s** (87.76 min) |

- **Best Checkpoint**: Persisted to `models/checkpoints/resnet50_best.pth`.

---

## 7. Direct Architecture Comparison: EfficientNet-B0 vs. ResNet-50

Comparing validation performance under the matched protocol:

| Metric | EfficientNet-B0 (Epoch 1 Matched) | ResNet-50 (Epoch 1 Matched) | Delta (ResNet-50 - EfficientNet) | EfficientNet-B0 (Full 15-Epoch Baseline) |
|:---|:---:|:---:|:---:|:---:|
| **Validation Loss** | **0.8379** | **0.7646** | **`-0.0733` (Better)** | **0.6422** (Epoch 14) |
| **Top-1 Validation Acc**| **72.32%** | **72.52%** | **`+0.20%` (Slightly Better)**| **77.38%** (Peak 77.65%) |
| **Training Loss** | 0.9528 | 0.9035 | `-0.0493` (Better) | 0.6121 |
| **Training Accuracy** | 69.13% | 69.22% | `+0.09%` | 77.45% |
| **Total Parameters** | **4,016,515** | **23,522,375** | **`+19,505,860` (+485%)** | 4,016,515 |
| **Trainable Parameters**| **8,967** | **14,343** | **`+5,376` (+60%)** | 8,967 |
| **Epoch Duration (CPU)**| **259.35s** (4.3 min) | **5,265.40s** (87.8 min) | **`+5,006.05s` (20.3× Slower)** | 248.31s avg |
| **Model Footprint** | $\approx 16.1\text{ MB}$ | $\approx 94.1\text{ MB}$ | **`5.84× Larger`** | $\approx 16.1\text{ MB}$ |

---

## 8. In-Depth Comparative Analysis

1. **Initial Convergence Dynamics**:
   - ResNet-50 converged to a lower validation loss on Epoch 1 (`0.7646` vs `0.8379`).
   - The wider 2,048-dimensional feature representation from ResNet-50's final layer provides a higher-dimensional embedding space, allowing the linear classification head to find slightly better initial hyperplane separations.
2. **Accuracy Parity**:
   - The validation accuracy difference at Epoch 1 is merely **`+0.20%`** (1,103 vs 1,100 correct samples out of 1,521). This difference is not statistically or clinically significant.
3. **Computational Efficiency & Edge Viability**:
   - ResNet-50 requires **`20.3× more CPU time per epoch`** ($87.8$ minutes vs $4.3$ minutes).
   - In clinical edge environments, telemedicine kiosks, or mobile dermatology apps where GPU acceleration is unavailable or battery-constrained, EfficientNet-B0 offers dramatically superior parameter efficiency and inference throughput.
4. **Verdict**:
   - While ResNet-50 shows strong feature representations, **EfficientNet-B0 remains the decisively superior architecture for MediScan** due to its optimal balance of compact size (4.02M parameters), fast training/inference, and demonstrated 75.79% Top-1 / 89.55% Top-2 test performance.

---

## 9. Checkpoint Verification & Reload

- **Checkpoint File**: `models/checkpoints/resnet50_best.pth`
- **Metadata Verified**:
  - `epoch`: 1
  - `val_loss`: 0.7646
  - `val_accuracy`: 0.7252
  - `architecture`: `resnet50`
  - `num_classes`: 7
- **Numerical Reload Test**:
  - Re-instantiated fresh ResNet-50 model.
  - Successfully restored weights via `load_state_dict()`.
  - Executed dummy forward pass: outputs shape $[2, 7]$, strictly finite, zero NaN/Inf.

---

## 10. Candidate Grad-CAM Target Layer for ResNet-50

Inspected and documented for future explainability workflows:
- **Module Path**: `model.layer4[-1]`
- **Block Type**: `Bottleneck`
- **Final Convolution**: `model.layer4[-1].conv3`
- **Output Shape**: $[B, 2048, 7, 7]$
- **Activation**: Post-`relu` output of `model.layer4[-1]`.

---

## 11. Strict Test Set Protection Audit

- The test DataLoader was **never instantiated or called** during Day 11.
- `reports/data/day11_resnet50_training_history.csv` contains zero test metrics.
- `reports/data/day11_model_comparison.csv` contains strictly validation and parameter statistics.
- Test split checksum remains unaltered:
  - `split_test.csv` MD5: `6a5ae1b65c25d504f78bc1da2361d82c` (Verified identical).
- Official Day 9 test results for EfficientNet-B0 remain locked and unaltered.

---

## 12. Automated Test Suite Results

Execution of the Day 11 test suite:
```bash
pytest tests/test_day11_resnet50.py -v
```
**Result: 21 / 21 PASSED (100%)**
- `TestModelArchitecture`: 5 passed
- `TestParameterFreezingAndGradients`: 3 passed
- `TestForwardPassAndCheckpoint`: 2 passed
- `TestGradCAMTargetLayer`: 1 passed
- `TestOutputArtifactsAndProtection`: 10 passed

---

## 13. Generated Artifacts Register

| Artifact Type | Path | Details |
|:---|:---|:---|
| **ResNet-50 Model Module** | `src/models/resnet50.py` | Full PyTorch ResNet-50 implementation |
| **Model Comparison Table** | `reports/data/day11_model_comparison.csv` | Controlled benchmark metrics |
| **Training History CSV** | `reports/data/day11_resnet50_training_history.csv` | Epoch loss, acc, duration, LR |
| **Best Model Checkpoint** | `models/checkpoints/resnet50_best.pth` | Best validation loss checkpoint |
| **ResNet-50 Curves Plot** | `reports/figures/day11/day11_resnet50_training_curves.png` | Standalone loss/accuracy plot |
| **Comparative Curves Plot**| `reports/figures/day11/day11_efficientnet_vs_resnet50_comparison.png`| Side-by-side benchmark figures |
| **Unit Test Suite** | `tests/test_day11_resnet50.py` | 21 automated verification tests |
| **Completion Report** | `reports/DAY11_COMPLETION_REPORT.md` | Comprehensive analysis document |

---

## 14. Final Conclusion & Status Decision

# **PASS — DAY 11 COMPLETE**

The second transfer learning architecture (ResNet-50) has been implemented, verified, trained under the controlled benchmark protocol, and compared fairly against the EfficientNet-B0 baseline. The project is completely prepared to proceed to **Day 12 (Web Application & User Interface)**.

### Verification Summary Table:

| Area | Status | Evidence |
|------|:---:|----------|
| **ResNet-50 Architecture** | **PASS** | Implemented in `src/models/resnet50.py`. 2048 in-features, 7 outputs, ImageNet-1K pretrained. |
| **Pretrained Weights** | **PASS** | Loaded `ResNet50_Weights.IMAGENET1K_V1`. |
| **Parameter Freezing** | **PASS** | $23,508,032$ frozen backbone params, $14,343$ trainable classifier params ($0.061\%$). |
| **Gradient Flow** | **PASS** | Sanity check passed: classifier parameters updated, frozen backbone bitwise identical. |
| **Training** | **PASS** | Controlled CPU-budgeted training executed and logged to `day11_resnet50_training_history.csv`. |
| **Validation Evaluation** | **PASS** | Evaluated on 1,521 validation samples (Loss: `0.7646`, Acc: `72.52%`). |
| **Fair Comparison** | **PASS** | Matched protocol: same split, preprocessing, augmentation, Adam optimizer (LR=1e-3). |
| **Checkpoint** | **PASS** | Best checkpoint persisted to `models/checkpoints/resnet50_best.pth` and reload verified. |
| **Test Protection** | **PASS** | Test set was never loaded or used. `split_test.csv` MD5 unchanged. Day 9 baseline locked. |
| **Grad-CAM Target Identified**| **PASS** | Target layer identified as `model.layer4[-1]` (Bottleneck, 2048 channels). |
| **Tests** | **PASS** | 21 / 21 passed in `test_day11_resnet50.py`. Regression test suite passing. |
| **Overall** | **PASS** | **DAY 11 COMPLETE**. Ready for Day 12. |

*(Day 12 has not been started per instructions.)*
