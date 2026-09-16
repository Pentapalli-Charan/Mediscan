# MediScan — Day 12 Pre-Flight Audit Report: `app/utils.py` Verification

**Date:** September 16, 2026  
**Auditor:** Pentapalli Charan / AntiGravity  
**Phase:** Day 12 Pre-Flight Audit (Streamlit Web Application Utilities)  
**Status:** **PASS — ALL 10 CRITERIA VERIFIED**  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | Streamlit 1.54.0 | OS: Windows 11  

---

## 1. Executive Summary

Prior to implementing the interactive Streamlit user interface (`app/app.py`), a comprehensive, line-by-line pre-flight audit of `app/utils.py` was conducted against the verified pipeline artifacts from Days 1–11. 

All 10 verification criteria passed without compromising or altering any ML model weights, configuration parameters, preprocessing definitions, test partitions, or explainability targets. The full test regression suite (**272 passed, 1 skipped**) and targeted Day 12 utility tests (**21 passed**) confirmed complete backward compatibility, strict test-set isolation, and robust error resilience.

---

## 2. Files Inspected

The audit cross-referenced `app/utils.py` against the following canonical implementations:

1. **Model Architecture & Factory**:
   - `src/models/efficientnet.py` (Architecture definition, classification head, freezing strategy, `build_model`, `CLASS_NAMES`, `NUM_CLASSES`)
   - `src/models/resnet50.py` (Comparison model to ensure zero cross-contamination)
   - `src/models/__init__.py` (Model registry exports)
2. **Evaluation & Inference Pipeline**:
   - `src/evaluation/inference.py` (Forward inference, softmax formulation, confidence computation)
   - `src/evaluation/metrics.py` (Top-1/Top-2 accuracy, multi-class diagnostics)
   - `reports/data/day9_metrics.json` (Locked Day 9 test metrics baseline)
3. **Explainability & Heatmaps**:
   - `src/explainability/gradcam.py` (Hook lifecycle, target layer lookup, weight snapshotting, gradient pooling)
   - `src/explainability/visualization.py` (`create_gradcam_overlay`, colormap blending)
4. **Data Preprocessing & Augmentation**:
   - `src/data/preprocessing_day3.py` (`load_config`, `get_augmentation_transform` mode="test", `get_class_label_mapping`)
   - `src/data/dataloader.py` (Deterministic DataLoader validation)
5. **Configuration & Checkpoints**:
   - `config/config.yaml` (Centralized hyperparameters, paths, class names)
   - `models/checkpoints/efficientnet_b0_best.pth` (Primary trained model checkpoint)
   - `reports/data/split_test.csv` (Isolated test split)
6. **Prior Test Suites**:
   - `tests/test_day9_evaluation.py`
   - `tests/test_day10_gradcam.py`
   - `tests/test_day11_resnet50.py`

---

## 3. `app/utils.py` Functions Inspected

Every utility function was audited for determinism, memory hygiene, type safety, and error handling:

| Function Name | Return Type | Role in Day 12 App |
|:---|:---|:---|
| `resolve_project_path` | `Path` | Resolves relative paths cleanly whether launched from workspace root or `app/`. |
| `get_inference_device` | `torch.device` | Detects CUDA capability; safely falls back to CPU. |
| `load_mediscan_model` | `Tuple[nn.Module, Dict]` | Instantiates EfficientNet-B0 and restores trained weights with requires_grad=False. |
| `load_cached_mediscan_model` | `Tuple[nn.Module, Dict]` | Streamlit resource caching wrapper (`st.cache_resource`) with headless fallback. |
| `validate_and_load_image` | `PIL.Image.Image` | Validates file extensions, size limits (<=25MB), corrupt buffers, and formats into RGB. |
| `preprocess_image_for_inference` | `Tuple[torch.Tensor, np.ndarray]` | Applies exact Day 2–4 deterministic test preprocessing (Resize 224x224, ImageNet norm). |
| `run_model_inference` | `Dict[str, Any]` | Executes forward pass under `torch.no_grad()`, returning 7 class probabilities and confidence. |
| `compute_gradcam_explanation` | `Dict[str, Any]` | Generates Grad-CAM heatmap on `model.features[8]` with bitwise weight immutability verification. |
| `get_ham10000_sample_catalog` | `List[Dict[str, Any]]` | Discovers sample images across all 7 classes without loading image arrays into memory. |

---

## 4. Verification Criteria & Audit Findings

### Criterion 1: Model Construction — **PASS**
- **Architecture**: Uses torchvision `efficientnet_b0`.
- **Classification Head**: Exactly `nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 7, bias=True))`.
- **Number of Classes**: Exactly 7 (`NUM_CLASSES = 7`).
- **Class Mapping**: Exactly matched to canonical HAM10000 diagnostics (`["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]`).
- **Human-Readable Descriptions**: Complete clinical names provided via `CLASS_DESCRIPTIONS`.

### Criterion 2: Checkpoint Loading & Integrity — **PASS**
- **Target File**: Loads `models/checkpoints/efficientnet_b0_best.pth`.
- **Architecture Safeguard**: Explicitly checks `raw_checkpoint.get("architecture") == "efficientnet_b0"`. Rejects ResNet-50 checkpoints (`ValueError`).
- **Offline Loading**: Model skeleton instantiated with `pretrained=False` before loading `checkpoint["model_state_dict"]`, eliminating unnecessary external network requests.
- **Weight Immutability**: All model parameters set to `requires_grad = False` and placed in `.eval()` mode.

### Criterion 3: Preprocessing Pipeline — **PASS**
- **Consistency**: Calls `get_augmentation_transform(cfg, mode="test")`.
- **Deterministic Pipeline**:
  1. RGB conversion (`pil_image.convert("RGB")`).
  2. Bilinear resize to $224 \times 224$ (`interpolation=1`).
  3. ImageNet normalization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
  4. PyTorch conversion (`ToTensorV2`).
- **Tensor Format**: Output shape strictly verified as `[1, 3, 224, 224]`, finite, with zero random augmentations.
- **Visual Reference**: Also returns matching `resized_rgb` ($224 \times 224 \times 3$, `uint8`) for artifact-free Grad-CAM overlays.

### Criterion 4: Forward Prediction — **PASS**
- **Activation**: Correctly applies `torch.softmax(logits, dim=1)`.
- **Distribution**: Generates exactly 7 non-negative probabilities summing to $1.0 \pm 10^{-4}$.
- **Finite Check**: Explicitly verifies `np.isfinite(probs).all()`.
- **Ranking**: Generates sorted probability pairs and selects top-1 class index via `np.argmax(probs)`.
- **Test Set Isolation**: Zero evaluation is performed on `split_test.csv`.

### Criterion 5: Grad-CAM Explainability — **PASS**
- **Reusability**: Reuses verified `GradCAM` from `src.explainability.gradcam`.
- **Target Layer**: Strictly targets `model.features[8]` (final conv block of EfficientNet-B0). Does **not** use ResNet-50's `model.layer4[-1]`.
- **Hook Lifecycle**: Hooks registered and removed inside a `try...finally:` block.
- **Weight Snapshotting**: Takes `snapshot_model_weights` before execution and runs `verify_model_weights_unchanged` after execution.
- **Decoupled Failure**: `run_model_inference` and `compute_gradcam_explanation` are separate functions. If Grad-CAM fails, prediction remains fully accessible.

### Criterion 6: Device Handling — **PASS**
- **Device Detection**: `get_inference_device()` queries `torch.cuda.is_available()`.
- **No False Claims**: Correctly uses CPU on this host environment (`device("cpu")`).
- **Dynamic Tensor Placement**: Automatically checks `next(model.parameters()).device` if no device is passed.

### Criterion 7: Streamlit Caching & Memory Hygiene — **PASS**
- **Model Caching**: Implemented `load_cached_mediscan_model` using `@st.cache_resource` when `st.runtime.exists()` is true.
- **No Stale User Predictions**: Inference is a pure function taking an input tensor and model; no session state cross-talk or global prediction caches.
- **Ephemeral Image Memory**: Uploaded files and PIL images are processed on-demand and garbage-collected; zero global image accumulators.

### Criterion 8: Robust Error Handling — **PASS**
- **Missing Checkpoint**: Raises descriptive `FileNotFoundError`.
- **Corrupt Checkpoint**: Raises descriptive `ValueError` on missing `model_state_dict`.
- **Architecture Mismatch**: Rejects non-EfficientNet checkpoints (`ValueError`).
- **Unsupported Image Format**: Rejects extensions not in `{".jpg", ".jpeg", ".png"}` (`ValueError`).
- **Empty Image Data**: Rejects 0-byte files or buffers (`ValueError`).
- **Oversized Images**: Enforces 25 MB ceiling (`MAX_IMAGE_FILE_SIZE_BYTES`).
- **Corrupted Image Buffer**: Catches decoding errors and raises `ValueError`.
- **Invalid Input Tensor**: Enforces shape `[B, 3, H, W]` and finite floats (`ValueError`).
- **Grad-CAM Failure**: Cleans up forward/backward hooks even if backward pass errors.

### Criterion 9: Code Quality & Dependency Hygiene — **PASS**
- **No Redundant ML Code**: Directly reuses `src.models`, `src.data`, `src.training`, and `src.explainability`.
- **No Hardcoded Absolute Paths**: All paths utilize `resolve_project_path` against `PROJECT_ROOT`.
- **No Internet Dependency**: Fully offline; no external weight downloads required.
- **Package Structure**: Added `app/__init__.py` marking `app` as an importable package.

### Criterion 10: Testability & Determinism — **PASS**
- **Headless Execution**: `app/utils.py` imports cleanly in bare Python without launching Streamlit.
- **Determinism**: Given the same image input, `preprocess_image_for_inference` and `run_model_inference` produce bitwise identical outputs.
- **Zero Hidden State**: Model evaluation mode and weight frozen state strictly maintained.

---

## 5. Modifications Made to `app/utils.py`

To satisfy all 10 criteria, the following specific refinements were made to `app/utils.py`:

1. **Path Resolution**:
   Added `PROJECT_ROOT` and `resolve_project_path()` to guarantee that checkpoints, configs, and sample images resolve correctly regardless of the caller's working directory.
2. **Architecture Safeguard & Offline Loading**:
   Added checkpoint dictionary validation in `load_mediscan_model()` to verify `architecture == 'efficientnet_b0'`, and set `pretrained=False` in `build_model()` so the model instantiates instantly without contacting external CDNs.
3. **Safe Streamlit Caching**:
   Added `load_cached_mediscan_model()` with conditional `st.runtime.exists()` detection so that caching is active in Streamlit while avoiding warning messages or mock requirements during CLI/pytest runs.
4. **Input Extension & Size Validation**:
   Added explicit checks against `SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}` and empty buffers (`len == 0`) for both file paths and Streamlit `UploadedFile` objects.
5. **Tensor Shape & Finite Checks**:
   Added strict shape `[B, 3, H, W]` and `torch.isfinite()` assertions in `run_model_inference()`.
6. **Package Initialization**:
   Created `app/__init__.py` to enable standard Python module importing.

---

## 6. Verification Evidence: Exact Tests Executed & Pass Counts

### A. Targeted `app/utils.py` Test Suite
```bash
pytest tests/test_app_utils.py -v
```
**Results: 21 PASSED, 0 FAILED (100%) in 14.22s**

| Test Class | Test Method | Status |
|:---|:---|:---:|
| `TestModelConstructionAndCheckpoint` | `test_model_construction_and_eval_mode` | **PASSED** |
| | `test_all_parameters_frozen` | **PASSED** |
| | `test_missing_checkpoint_raises_filenotfound` | **PASSED** |
| | `test_missing_config_raises_filenotfound` | **PASSED** |
| | `test_reject_resnet50_checkpoint` | **PASSED** |
| | `test_reject_corrupt_checkpoint` | **PASSED** |
| `TestImageValidationAndLoading` | `test_valid_pil_image` | **PASSED** |
| | `test_valid_image_bytes` | **PASSED** |
| | `test_empty_bytes_raises_value_error` | **PASSED** |
| | `test_unsupported_file_extension_raises` | **PASSED** |
| | `test_corrupted_image_bytes_raises` | **PASSED** |
| | `test_nonexistent_file_path_raises` | **PASSED** |
| | `test_uploaded_file_mock` | **PASSED** |
| `TestPreprocessing` | `test_preprocessing_output_shape_and_range` | **PASSED** |
| | `test_preprocessing_determinism` | **PASSED** |
| `TestModelInference` | `test_prediction_probabilities` | **PASSED** |
| | `test_prediction_rejects_invalid_tensor` | **PASSED** |
| `TestGradCAMExplanation` | `test_gradcam_output_and_weight_immutability` | **PASSED** |
| `TestStreamlitCachingAndCatalog` | `test_device_detection` | **PASSED** |
| | `test_load_cached_mediscan_model_outside_streamlit` | **PASSED** |
| | `test_ham10000_sample_catalog` | **PASSED** |

---

### B. Full Project Regression Suite
```bash
pytest tests/ -q
```
**Results: 272 PASSED, 1 SKIPPED, 0 FAILED (100%) in 173.84s (02:53)**

All prior milestones from Day 2 through Day 11 remain completely intact:
- `test_day2_preprocessing.py`: All passed
- `test_day3_augmentation.py`: All passed
- `test_day4_dataloader.py`: All passed
- `test_day5_model.py`: All passed
- `test_day6_training.py`: All passed
- `test_day7_analysis.py`: All passed
- `test_day8_tuning.py`: All passed
- `test_day9_evaluation.py`: All passed
- `test_day10_gradcam.py`: All passed
- `test_day11_resnet50.py`: All passed
- `test_app_utils.py`: All passed

---

## 7. Pipeline Integrity Confirmations

1. **EfficientNet-B0 Checkpoint Integrity**:
   - **Path**: `models/checkpoints/efficientnet_b0_best.pth`
   - **MD5**: `7c1c6fcbe02e93f0ff8b4a20f62e29e3` (**Unchanged, verified**)
   - **Best Epoch**: 14
   - **Best Validation Loss**: `0.642151`
   - **Best Validation Accuracy**: `77.38%`
2. **Day 9 Quantitative Metrics & Test Partition Integrity**:
   - **Test Split Path**: `reports/data/split_test.csv`
   - **MD5**: `6a5ae1b65c25d504f78bc1da2361d82c` (**Unaltered, verified**)
   - **Top-1 Test Accuracy**: `75.79%` (**Locked**)
   - **Top-2 Test Accuracy**: `89.55%` (**Locked**)
   - **Macro-F1**: `0.5157` / **Weighted-F1**: `0.7448` (**Locked**)
3. **Grad-CAM Target Layer**:
   - Strictly `model.features[8]` (Final conv block of EfficientNet-B0) (**Verified**).
   - ResNet-50's `model.layer4[-1]` is not referenced or accessed in `app/utils.py`.

---

## 8. Remaining Issues & Status

- **Remaining Issues**: **NONE**.
- **Scope Compliance**:
  - `app/app.py` has **NOT** been created or implemented yet.
  - Day 13 has **NOT** been started.
- **Audit Verdict**: **PASS**. `app/utils.py` is fully verified, tested, and ready for Day 12 frontend implementation.
