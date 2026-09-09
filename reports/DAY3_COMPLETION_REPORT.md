## MediScan - DAY 3 COMPLETION REPORT

**Status:** COMPLETE  
**Date:** 2026-09-09

## EXECUTIVE SUMMARY

Day 3 data augmentation is implemented and verified. Training receives conservative random augmentation after split selection. Validation and test receive deterministic resize and ImageNet normalization only. No model training or later milestone work was performed.

- 26 Day 3 tests passing
- 46 Day 2 + Day 3 data pipeline tests passing
- Train/validation/test safety checks passed
- 3 representative visualization figures generated
- Existing split assignments verified unchanged
- No lesion or image leakage introduced

## A. FILES INSPECTED

- `config/config.yaml`
- `src/data/preprocessing.py`
- `src/data/preprocessing_day3.py`
- `reports/data/split_train.csv`
- `reports/data/split_val.csv`
- `reports/data/split_test.csv`
- `requirements.txt`
- `tests/test_day3_augmentation.py`
- Existing Dataset references under `src/`

## B. FILES CREATED

- `src/data/preprocessing_day3.py`
- `tests/test_day3_augmentation.py`
- `scripts/day3_augmentation.py`
- `reports/day3_augmentation_report.json`
- `reports/DAY3_COMPLETION_REPORT.md`
- `reports/figures/day3_augmentation_examples_ISIC_0029417.png`
- `reports/figures/day3_augmentation_examples_ISIC_0034093.png`
- `reports/figures/day3_augmentation_examples_ISIC_0027419.png`

## C. FILES MODIFIED

- `config/config.yaml`: documented the seven class labels and added configurable probabilities for compound transforms.
- `src/data/preprocessing.py`: repaired an accidental Day 3 insertion that had made the existing Day 2 module syntactically invalid. Day 2 split logic and assignments were not changed.

## D. EXACT AUGMENTATION PIPELINE

Training pipeline:

1. `HorizontalFlip`
2. `VerticalFlip`
3. `Rotate`
4. `ShiftScaleRotate`
5. `RandomBrightnessContrast`
6. `Resize(224, 224)`
7. ImageNet `Normalize`
8. `ToTensorV2`

The transforms are applied after the Dataset reads an already assigned split CSV row. No augmented copies are created on disk.

## E. EXACT PROBABILITIES AND MAGNITUDES

| Transform | Probability | Magnitude |
|---|---:|---|
| HorizontalFlip | 0.5 | horizontal mirror |
| VerticalFlip | 0.5 | vertical mirror |
| Rotate | 0.5 | limit +/-30 degrees |
| ShiftScaleRotate | 0.5 | shift 0.1, scale 0.15, rotate +/-30 degrees |
| RandomBrightnessContrast | 0.5 | brightness 0.2, contrast 0.2 |

All values are loaded from `config/config.yaml`. Saturation and hue values remain documented configuration options but are not applied because color changes can alter clinically meaningful dermatoscopic features and were not necessary for this conservative pipeline.

## F. TRAIN PREPROCESSING PIPELINE

`random augmentation -> resize to 224x224 -> ImageNet normalization -> ToTensorV2`

Output verified as a `torch.float32` tensor with shape `(3, 224, 224)`.

## G. VALIDATION PREPROCESSING PIPELINE

`RGB conversion -> resize to 224x224 -> ImageNet normalization -> ToTensorV2`

No random augmentation is applied. Repeated validation reads are deterministic.

## H. TEST PREPROCESSING PIPELINE

`RGB conversion -> resize to 224x224 -> ImageNet normalization -> ToTensorV2`

No random augmentation is applied. The test split was used only for pipeline safety verification, not tuning or training.

## I. CLASS LABEL MAPPING

The same deterministic mapping is used for every split:

| Label | Integer |
|---|---:|
| akiec | 0 |
| bcc | 1 |
| bkl | 2 |
| df | 3 |
| mel | 4 |
| nv | 5 |
| vasc | 6 |

The mapping is implemented in `src/data/preprocessing_day3.py` and documented in `config/config.yaml`.

## J. VISUALIZATION FILES GENERATED

- `reports/figures/day3_augmentation_examples_ISIC_0029417.png`
- `reports/figures/day3_augmentation_examples_ISIC_0034093.png`
- `reports/figures/day3_augmentation_examples_ISIC_0027419.png`

Each figure includes the original image, horizontal flip, vertical flip, rotation, brightness/contrast, and combined mild augmentation. Examples are selected from different diagnostic classes where available.

## K. TESTS EXECUTED AND RESULTS

Focused Day 3 suite:

- `python -m pytest tests/test_day3_augmentation.py -q`
- **26 passed, 10 warnings**

Combined data pipeline regression suite:

- `python -m pytest tests/test_day2_preprocessing.py tests/test_day3_augmentation.py -q`
- **46 passed, 10 warnings**

Additional checks:

- `python -m py_compile src/data/preprocessing.py src/data/preprocessing_day3.py`
- Passed
- `python scripts/day3_augmentation.py`
- Completed successfully

The checks sampled 10 images from each train, validation, and test pipeline. All sampled outputs were `(3, 224, 224)`, `torch.float32`, and had zero NaN and zero infinite values.

## L. SPLIT ASSIGNMENTS UNCHANGED

Verified from `reports/data/split_all.csv` and the three split CSVs:

| Split | Images | Percentage |
|---|---:|---:|
| Train | 6,982 | 69.72% |
| Validation | 1,521 | 15.19% |
| Test | 1,512 | 15.10% |
| Total | 10,015 | 100.00% |

All rows retained valid split assignments. The existing Day 2 assignments were not recreated or modified.

## M. NO LEAKAGE INTRODUCED

Verified:

- No lesion ID occurs in more than one split.
- No image ID occurs in more than one split.
- All 10,015 images remain accounted for.
- Augmentation occurs inside the Dataset after reading the assigned split CSV.
- No augmented files are written into the dataset or shared between splits.

## CLASS DISTRIBUTION

The existing split distribution was reported without rebalancing:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| akiec | 224 | 48 | 55 |
| bcc | 361 | 82 | 71 |
| bkl | 772 | 160 | 167 |
| df | 71 | 24 | 20 |
| mel | 773 | 172 | 168 |
| nv | 4,682 | 1,016 | 1,007 |
| vasc | 99 | 19 | 24 |

No sampler, class weighting, or rebalance strategy was implemented.

## N. WARNINGS AND UNRESOLVED ISSUES

- Albumentations emits a deprecation warning that `ShiftScaleRotate` is a special case of `Affine`. The configured transform remains conservative and all tests pass. Migration to `Affine` can be handled in a later maintenance change.
- `saturation_limit` and `hue_limit` remain in configuration for future controlled experiments but are intentionally unused in this medical-safe baseline.
- No model training, EfficientNet, ResNet, Grad-CAM, Streamlit, or MLflow work was performed.

## CONCLUSION

DAY 3 IS COMPLETE. The project now has separate train, validation, and test preprocessing behavior, a shared class mapping, a tested PyTorch Dataset, representative visual evidence, and verified split/leakage safety.
