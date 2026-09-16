# MediScan — Day 13 Completion Report: Experiment Tracking & MLflow Integration

**Author:** Pentapalli Charan  
**Date:** September 16, 2026  
**Project:** MediScan — Skin Lesion Classification (HAM10000)  
**Phase:** Day 13 — Experiment Tracking & MLflow Integration  
**Environment:** Python 3.14.5 | PyTorch 2.10.0+cpu | torchvision 0.28.0+cpu | MLflow 3.16.0 | Streamlit 1.54.0 | OS: Windows 11  

---

## 1. Executive Summary & Objective

Day 13 integrated **MLflow** into the MediScan codebase to provide reproducible, offline experiment tracking, hyperparameter logging, epoch metric histories, artifact versioning, and architectural benchmarking.

All runs tracked in MLflow correspond to **actual executed experiments** from Days 6, 8, and 11. No skipped experiments were fabricated, no models were unnecessarily retrained, and the locked Day 9 held-out test evaluation was preserved with explicit provenance tags.

### Key Milestones Delivered:
1. **Local MLflow Store**: Initialized local file-based tracking backend at `mlruns/` under experiment name **`MediScan`** (Experiment ID: `944157940989004540`).
2. **Three Genuine Historical Runs Synchronized**:
   - `EfficientNet-B0-Baseline` (Day 6 / Day 8 Exp 0): Run ID `24bf332a99a647e2847c04591165ab59`
   - `EfficientNet-B0-LR-5e-4` (Day 8 Exp 1): Run ID `202df14fe4a84ac085431ee958123c98`
   - `ResNet50-Benchmark` (Day 11 Comparison): Run ID `ae1efd6de0b847d88fc93f5b7d59d939`
3. **Preserved Test Set Isolation**: Day 9 test metrics were recorded with explicit metadata tags indicating they are from the historical held-out test evaluation and were **not** used for model selection or tuning.
4. **Offline Resilience**: Runs completely offline without external network requests or remote server dependencies.
5. **UI & Streamlit Integration**: Added an optional, resilient "Model Development & Benchmarks" section to the Streamlit app sidebar.
6. **Automated Verification**: 10 / 10 unit tests passed in `tests/test_day13_mlflow.py`, and the full project regression test suite passed (**287 passed, 1 skipped**).

---

## 2. MLflow Setup & Tracking Architecture

- **MLflow Version:** `3.16.0` (installed via pip and added to `requirements.txt`).
- **Tracking Store Backend:** Local file-based store at `mlruns/` (normalized URI: `file:///C:/Users/pchar/OneDrive/Desktop/MediScan%20Project/mlruns`).
- **File Store Configuration:** Enabled via `os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"` in tracking utilities to ensure seamless operation under MLflow 3.16+.
- **Experiment Registry:**
  - Experiment Name: **`MediScan`**
  - Experiment ID: **`944157940989004540`**
  - Lifecycle Stage: `active`
  - Tags: `project=MediScan`, `dataset=HAM10000`, `task=Dermoscopic Skin Lesion Classification`, `framework=PyTorch`.

---

## 3. Synchronized Runs & Benchmark Comparison

The 3 actual executed runs in the canonical `MediScan` experiment:

| Run Name | Run ID | Architecture | LR | Dropout | Epochs | Best Val Loss | Val Acc (%) | Test Top-1 Acc (%) | Milestone Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **`EfficientNet-B0-Baseline`** | `24bf332a` | `efficientnet_b0` | 0.0010 | 0.2 | 15 | **`0.6422`** | **`77.38%`** | **`75.79%`** | Production Deployment Model |
| **`EfficientNet-B0-LR-5e-4`** | `202df14f` | `efficientnet_b0` | 0.0005 | 0.2 | 5 | `0.7283` | `75.21%` | N/A | Hyperparameter Candidate |
| **`ResNet50-Benchmark`** | `ae1efd6d` | `resnet50` | 0.0010 | 0.2 | 1 | `0.7646` | `72.52%` | N/A | Architecture Comparison |

---

## 4. Parameter & Metric Tracking Breakdown

### A. Tracked Parameters (30 per run):
- **Model Architecture**: `architecture`, `pretrained_weights`, `num_classes` (7), `classifier_head`, `dropout`, `backbone_frozen` (`True`), `total_parameters`, `trainable_parameters`.
- **Dataset & Split**: `dataset_name` (`HAM10000`), `train_samples` (6,982), `val_samples` (1,521), `test_samples` (1,512), `split_strategy` (`stratified_group_lesion_id`), `input_resolution` (`224x224`), `normalize_mean`, `normalize_std`, `augmentation`.
- **Training Optimization**: `optimizer` (`Adam`), `learning_rate`, `batch_size` (32), `max_epochs`, `scheduler` (`ReduceLROnPlateau`), `early_stopping_patience` (3), `loss_function` (`CrossEntropyLoss`), `random_seed` (42).
- **Environment & Provenance**: `device` (`cpu`), `python_version` (`3.14.5`), `pytorch_version` (`2.10.0+cpu`), `torchvision_version` (`0.28.0+cpu`), `git_commit` (`07c463cfb146197e06d5f483e7a4ce85048193f8`).

### B. Tracked Metrics:
- **Epoch-Level (step-logged from CSVs)**:
  - `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `learning_rate`, `epoch_time_seconds` (or `epoch_duration_seconds`).
- **Summary Validation Metrics**:
  - `best_epoch`, `best_val_loss`, `val_accuracy_at_best_loss`, `peak_val_accuracy`, `total_training_duration_seconds`.
- **Locked Day 9 Held-Out Test Metrics (`EfficientNet-B0-Baseline` only)**:
  - `test_top1_accuracy`: `0.757937`
  - `test_top2_accuracy`: `0.895503`
  - `test_macro_precision`: `0.574358`
  - `test_macro_recall`: `0.478688`
  - `test_macro_f1`: `0.515696`
  - `test_weighted_precision`: `0.738198`
  - `test_weighted_recall`: `0.757937`
  - `test_weighted_f1`: `0.744771`
  - `test_macro_roc_auc`: `0.9211`
  - `test_weighted_roc_auc`: `0.9145`
  - `test_macro_pr_auc`: `0.5646`
  - `test_ce_loss`: `0.663784`

### C. Tracked Artifacts:
- `reports/data/day6_training_history.csv` (15 epochs)
- `reports/data/day8_history_exp1.csv` (5 epochs)
- `reports/data/day8_experiments.csv` (tuning comparison record)
- `reports/data/day11_resnet50_training_history.csv`
- `reports/data/day11_model_comparison.csv`
- `reports/data/day9_classification_report.csv`
- `reports/data/day9_metrics.json`
- `config/config.yaml`
- `reports/figures/day11/day11_resnet50_training_curves.png`
- `reports/figures/day11/day11_efficientnet_vs_resnet50_comparison.png`

---

## 5. Scientific Integrity & Historical Experiment Representation

1. **Strict Test Set Protection**:
   - Test-set metrics were **not** used for hyperparameter selection or model selection.
   - Day 9 metrics were imported into MLflow tagged as:
     - `test_evaluation_status = "locked_held_out_evaluation"`
     - `test_evaluation_note = "Day 9 final locked test evaluation. NOT used for model selection."`
   - `reports/data/split_test.csv` MD5 checksum remains unaltered (`6a5ae1b65c25d504f78bc1da2361d82c`).
2. **Documented Skipped Experiments (Day 8 Exp 2, 3, 4)**:
   - Planned tuning experiments (`higher_lr_2e3`, `dropout_04`, `stronger_augmentation`) were skipped during Day 8 due to CPU compute constraints.
   - These experiments are **not** fabricated in MLflow. Only the genuine completed runs (`baseline_control` and `lower_lr_5e4`) exist in the tracking store.
3. **ResNet-50 Benchmark Context**:
   - Tagged with `compute_constraint_note = "Controlled 1-epoch benchmark executed under CPU budget constraint. Full 15-epoch test evaluation was NOT performed for ResNet-50."`
   - Demonstrates that EfficientNet-B0 is 20.3× faster per epoch on CPU while delivering comparable feature representations.

---

## 6. Checkpoint & Artifact Integrity Verification

| Checkpoint / Artifact | Expected MD5 | Actual MD5 | Status |
|:---|:---:|:---:|:---:|
| `models/checkpoints/efficientnet_b0_best.pth` | `7c1c6fcbe02e93f0ff8b4a20f62e29e3` | `7c1c6fcbe02e93f0ff8b4a20f62e29e3` | **LOCKED & BITWISE IDENTICAL** |
| `reports/data/split_test.csv` | `6a5ae1b65c25d504f78bc1da2361d82c` | `6a5ae1b65c25d504f78bc1da2361d82c` | **LOCKED & BITWISE IDENTICAL** |
| `reports/data/day9_metrics.json` | Top-1: `0.757937` | Top-1: `0.757937` | **LOCKED & IDENTICAL** |

The canonical model checkpoint file was **not** overwritten or replaced with an MLflow-generated copy.

---

## 7. Streamlit App Integration

In `app/app.py`, an optional expander was added to the sidebar:
`📊 Model Development & Benchmarks`
- Retrieves the experiment comparison table via `get_experiment_summary_df()` without exposing internal file paths.
- Provides fallback static metrics if the MLflow store is absent.
- Does not affect normal inference or image processing performance.

---

## 8. Automated Verification & Testing

### A. Targeted MLflow Test Suite (`tests/test_day13_mlflow.py`):
```bash
pytest tests/test_day13_mlflow.py -v
```
**Result: 10 / 10 PASSED (100%) in 12.49s**
- `TestMLflowCoreFunctionality`:
  - `test_mlflow_import_and_version`: PASSED
  - `test_isolated_experiment_and_run_lifecycle`: PASSED
- `TestCanonicalMediScanTrackingStore`:
  - `test_mediscan_experiment_exists`: PASSED
  - `test_canonical_runs_count_and_names`: PASSED
  - `test_baseline_run_metrics_and_provenance`: PASSED
  - `test_summary_dataframe_generation`: PASSED
- `TestPipelineIntegrityAndNonInterference`:
  - `test_efficientnet_checkpoint_hash_unaltered`: PASSED
  - `test_test_split_hash_unaltered`: PASSED
  - `test_day9_metrics_json_unaltered`: PASSED
  - `test_app_utils_inference_still_operational`: PASSED

### B. Frontend App Test Suite (`tests/test_day12_app.py`):
```bash
pytest tests/test_day12_app.py -v
```
**Result: 5 / 5 PASSED (100%)**

### C. Full Regression Test Suite (Days 2–13):
```bash
pytest tests/ -q
```
**Result: 287 PASSED, 1 SKIPPED (100%) in 371.48s (06:11)**

Zero regressions across all test suites from Days 2 to 13.

---

## 9. How to Use MLflow in MediScan

### 1. Synchronize or Ingest Experiments
```bash
python scripts/sync_day13_mlflow.py
```

### 2. Launch Local MLflow UI
```bash
mlflow ui --backend-store-uri mlruns --port 5000
```
Open `http://localhost:5000` in your web browser.

### 3. Query Runs via CLI
```bash
# Search experiments
mlflow experiments search --tracking-uri mlruns

# List runs for MediScan experiment
mlflow runs list --experiment-id 944157940989004540 --tracking-uri mlruns

# Describe specific run
mlflow runs describe --run-id 24bf332a99a647e2847c04591165ab59 --tracking-uri mlruns
```

---

## 10. Conclusion & Final Status Decision

# **PASS — DAY 13 COMPLETE**

MLflow has been integrated cleanly into MediScan. All 18 final acceptance criteria have been satisfied without compromising model checkpoints, test splits, or Day 9 locked metrics.

### Final Acceptance Criteria Checklist:

| Criterion | Status | Verification Evidence |
|:---|:---:|:---|
| MLflow dependency installed/documented | **PASS** | MLflow 3.16.0 installed, added to `requirements.txt`. |
| Local tracking backend works | **PASS** | FileStore at `mlruns/` verified offline. |
| MediScan experiment exists | **PASS** | Experiment ID `944157940989004540` active. |
| Actual experiment data is tracked | **PASS** | 3 genuine runs tracked (`Baseline`, `LR-5e-4`, `ResNet50`). |
| Parameters are logged | **PASS** | 30 parameters per run logged. |
| Metrics are logged | **PASS** | Epoch-level curves and summary metrics logged. |
| Artifacts are logged | **PASS** | CSV histories, config YAMLs, metrics JSON, figures logged. |
| Historical experiments represented honestly | **PASS** | Exact values from Days 6, 8, 11 history CSVs. |
| Skipped experiments NOT fabricated | **PASS** | Day 8 Exp 2, 3, 4 noted as skipped; no fake runs created. |
| Day 9 test metrics remain locked | **PASS** | Top-1 75.79%, Top-2 89.55%, Macro-F1 0.5157 preserved with provenance. |
| Test split remains unchanged | **PASS** | `split_test.csv` MD5 `6a5ae1b65c25d504f78bc1da2361d82c`. |
| EfficientNet checkpoint remains unchanged | **PASS** | `efficientnet_b0_best.pth` MD5 `7c1c6fcbe02e93f0ff8b4a20f62e29e3`. |
| Existing Streamlit inference functional | **PASS** | Streamlit app runs and renders with 0 errors. |
| MLflow UI/CLI access verified | **PASS** | CLI `search`, `list`, `describe` commands verified. |
| Day 13 tests pass | **PASS** | 10/10 passed in `tests/test_day13_mlflow.py`. |
| Full regression has 0 failures | **PASS** | 287 passed, 1 skipped (100%). |
| README is updated | **PASS** | `## MLflow Experiment Tracking (Day 13)` added. |
| DAY13_COMPLETION_REPORT.md exists | **PASS** | Created and verified. |

*(Day 14 has not been started per instructions.)*
