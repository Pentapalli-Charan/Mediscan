# Day 4 — PyTorch Dataset & DataLoader Pipeline — Completion Report

## A. Files Inspected

| File | Purpose |
|------|---------|
| [`config/config.yaml`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/config/config.yaml) | Project configuration |
| [`src/data/preprocessing.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/src/data/preprocessing.py) | Day 1–2 preprocessing utilities |
| [`src/data/preprocessing_day3.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/src/data/preprocessing_day3.py) | Day 3 augmentation, HAM10000Dataset, transforms |
| [`src/data/__init__.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/src/data/__init__.py) | Module init |
| [`src/utils/seed.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/src/utils/seed.py) | Reproducibility utilities |
| [`requirements.txt`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/requirements.txt) | Dependencies |
| [`reports/data/split_train.csv`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/data/split_train.csv) | Train split metadata |
| [`reports/data/split_val.csv`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/data/split_val.csv) | Validation split metadata |
| [`reports/data/split_test.csv`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/reports/data/split_test.csv) | Test split metadata |
| [`tests/test_day2_preprocessing.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/tests/test_day2_preprocessing.py) | Day 2 tests |
| [`tests/test_day3_augmentation.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/tests/test_day3_augmentation.py) | Day 3 tests |

---

## B. Files Created

| File | Purpose |
|------|---------|
| [`src/data/dataloader.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/src/data/dataloader.py) | Day 4 DataLoader factory + verification utilities |
| [`tests/test_day4_dataloader.py`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/tests/test_day4_dataloader.py) | Day 4 test suite (44 tests) |

---

## C. Files Modified

| File | Change |
|------|--------|
| [`config/config.yaml`](file:///c:/Users/pchar/OneDrive/Desktop/MediScan%20Project/config/config.yaml) | Added `dataloader` section with batch_size, num_workers, pin_memory, drop_last, persistent_workers |

No Day 2 or Day 3 source files were modified. No split CSVs were touched.

---

## D. Dataset Sizes

| Split | Expected | Actual | ✓ |
|-------|----------|--------|---|
| Train | 6,982 | 6,982 | ✅ |
| Validation | 1,521 | 1,521 | ✅ |
| Test | 1,512 | 1,512 | ✅ |
| **Total** | **10,015** | **10,015** | ✅ |

---

## E. DataLoader Configuration

All settings sourced from `config.yaml → dataloader`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `batch_size` | 32 | Standard for T4 GPU with EfficientNet-B0 |
| `num_workers` | 2 | Safe default for Windows + Colab T4 |
| `pin_memory` | true | Faster host-to-GPU transfer (no-op without GPU) |
| `drop_last` | false | Preserves all samples for evaluation accuracy |
| `persistent_workers` | false | Avoids Windows multiprocessing issues |
| train `shuffle` | True | Required for stochastic training |
| val `shuffle` | False | Deterministic evaluation |
| test `shuffle` | False | Deterministic evaluation |

---

## F. Actual Batch Shapes

| Split | Image Shape | Label Shape | Label Range |
|-------|-------------|-------------|-------------|
| Train | `[32, 3, 224, 224]` | `[32]` | `[2, 6]` |
| Val | `[32, 3, 224, 224]` | `[32]` | `[2, 2]` |
| Test | `[32, 3, 224, 224]` | `[32]` | `[2, 2]` |

All batches are CHW format float32 tensors. Labels are int64.

---

## G. Class Mapping Verification ✅

```
akiec = 0
bcc   = 1
bkl   = 2
df    = 3
mel   = 4
nv    = 5
vasc  = 6
```

- Same mapping used by all three datasets: ✅
- No unexpected class labels in any split CSV: ✅

---

## H. Image Loading Verification ✅

- All 10,015 images on disk: ✅
- Every `image_id` in `split_train.csv` (6,982) found: ✅
- Every `image_id` in `split_val.csv` (1,521) found: ✅
- Every `image_id` in `split_test.csv` (1,512) found: ✅
- 0 missing images: ✅
- No NaN values in any batch: ✅
- No Inf values in any batch: ✅

---

## I. DataLoader Performance Benchmark

Measured on Windows (CPU-only, no GPU), 10 batches per split:

| Split | Samples | Batches | Time (s) | Samples/sec |
|-------|---------|---------|----------|-------------|
| Train | 320 | 10 | 12.840 | 24.9 |
| Val | 320 | 10 | 8.652 | 37.0 |
| Test | 320 | 10 | 13.059 | 24.5 |

> [!NOTE]
> Train is slower due to augmentation transforms. Val is faster due to resize+normalize only.
> These are CPU-only numbers; Colab T4 with GPU pin_memory will be faster.

---

## J. Memory/Loading Behaviour

- Images loaded per-sample via `__getitem__` — NOT loaded into RAM at once: ✅
- Standard PyTorch DataLoader lazy batching: ✅
- No custom caching or prefetching: ✅
- No over-engineered memory management: ✅

---

## K. Tests Executed

| Test Suite | Tests | Result | Time |
|------------|-------|--------|------|
| Day 2 (`test_day2_preprocessing.py`) | 17 | **17 passed** | ~15s |
| Day 3 (`test_day3_augmentation.py`) | 29 | **29 passed** | ~60s |
| Day 4 (`test_day4_dataloader.py`) | 44 | **44 passed** | 226.72s |
| **Total** | **90** | **90 passed** | ~5 min |

---

## L. Exact Test Results

### Day 4 — 44/44 PASSED

```
tests/test_day4_dataloader.py::TestDatasetAccounting::test_train_dataset_length PASSED
tests/test_day4_dataloader.py::TestDatasetAccounting::test_val_dataset_length PASSED
tests/test_day4_dataloader.py::TestDatasetAccounting::test_test_dataset_length PASSED
tests/test_day4_dataloader.py::TestDatasetAccounting::test_total_samples PASSED
tests/test_day4_dataloader.py::TestDatasetAccounting::test_verify_dataset_accounting_utility PASSED
tests/test_day4_dataloader.py::TestDataLoaderCreation::test_loaders_contain_three_splits PASSED
tests/test_day4_dataloader.py::TestDataLoaderCreation::test_loaders_are_dataloader_instances PASSED
tests/test_day4_dataloader.py::TestDataLoaderCreation::test_train_batch_size_from_config PASSED
tests/test_day4_dataloader.py::TestDataLoaderCreation::test_val_batch_size_from_config PASSED
tests/test_day4_dataloader.py::TestDataLoaderCreation::test_test_batch_size_from_config PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_train_batch_shape PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_val_batch_shape PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_test_batch_shape PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_train_labels_type PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_train_labels_range PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_val_labels_range PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_test_labels_range PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_train_no_nan PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_train_no_inf PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_val_no_nan_inf PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_test_no_nan_inf PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_verify_batch_utility_train PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_verify_batch_utility_val PASSED
tests/test_day4_dataloader.py::TestBatchVerification::test_verify_batch_utility_test PASSED
tests/test_day4_dataloader.py::TestClassLabelMapping::test_canonical_mapping PASSED
tests/test_day4_dataloader.py::TestClassLabelMapping::test_all_datasets_use_same_mapping PASSED
tests/test_day4_dataloader.py::TestClassLabelMapping::test_verify_class_labels_utility PASSED
tests/test_day4_dataloader.py::TestClassLabelMapping::test_no_unexpected_labels_in_csvs PASSED
tests/test_day4_dataloader.py::TestImagePathResolution::test_all_split_images_exist PASSED
tests/test_day4_dataloader.py::TestImagePathResolution::test_image_count_on_disk PASSED
tests/test_day4_dataloader.py::TestImagePathResolution::test_image_loading_samples PASSED
tests/test_day4_dataloader.py::TestShuffleBehaviour::test_train_shuffle_enabled PASSED
tests/test_day4_dataloader.py::TestShuffleBehaviour::test_val_shuffle_disabled PASSED
tests/test_day4_dataloader.py::TestShuffleBehaviour::test_test_shuffle_disabled PASSED
tests/test_day4_dataloader.py::TestSplitIntegrity::test_split_csvs_exist PASSED
tests/test_day4_dataloader.py::TestSplitIntegrity::test_no_image_overlap_between_splits PASSED
tests/test_day4_dataloader.py::TestSplitIntegrity::test_no_lesion_overlap_between_splits PASSED
tests/test_day4_dataloader.py::TestSplitIntegrity::test_all_csv_rows_have_split_column PASSED
tests/test_day4_dataloader.py::TestSplitIntegrity::test_dataset_does_not_resplit PASSED
tests/test_day4_dataloader.py::TestConfiguration::test_config_has_dataloader_section PASSED
tests/test_day4_dataloader.py::TestConfiguration::test_config_batch_size PASSED
tests/test_day4_dataloader.py::TestConfiguration::test_config_num_workers PASSED
tests/test_day4_dataloader.py::TestConfiguration::test_config_pin_memory PASSED
tests/test_day4_dataloader.py::TestConfiguration::test_config_drop_last PASSED
```

### Day 2 + Day 3 — 46/46 PASSED

```
================= 46 passed, 10 warnings in 74.75s (0:01:14) ==================
```

---

## M. Split Assignment Verification ✅

- Split CSVs were NOT modified: ✅
- No new train/val/test split was created: ✅
- Dataset reads split CSVs as-is: ✅
- No image overlap between splits: ✅
- No lesion overlap between splits: ✅
- len(train_dataset) == len(split_train.csv) == 6,982: ✅
- len(val_dataset) == len(split_val.csv) == 1,521: ✅
- len(test_dataset) == len(split_test.csv) == 1,512: ✅

---

## N. Warnings/Deprecations

| Warning | Impact | Action |
|---------|--------|--------|
| `ShiftScaleRotate is a special case of Affine` | Albumentations API change | Future: migrate to `A.Affine` (Day 3 scope, not changed today) |
| `pin_memory set but no accelerator found` | No GPU on this dev machine | Harmless; will auto-use on Colab T4 |
| `Error fetching version info` (SSL timeout) | Albumentations version check | Network issue, no impact on functionality |

---

## O. Unresolved Issues

**None.** All Day 4 requirements are fully implemented and verified.

---

## Day 4 Completion Checklist

- [x] Dataset works for all three splits
- [x] All three DataLoaders successfully produce batches
- [x] Shapes `[B, 3, 224, 224]` and labels `[0–6]` verified
- [x] All 10,015 split images are accessible on disk
- [x] Loading performance measured (25–37 samples/sec CPU-only)
- [x] 90 tests pass across Day 2, 3, and 4
- [x] No split/leakage changes introduced
- [x] No models trained
- [x] No Day 5 work started
- [x] Raw dataset preserved
