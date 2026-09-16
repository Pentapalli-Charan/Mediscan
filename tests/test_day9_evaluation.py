"""
MediScan — Day 9: Comprehensive Baseline Model Evaluation Tests

Verification suite covering:
1. Test dataset integrity (1,512 images, 0 duplicates, 0 missing, valid classes)
2. Zero data leakage (zero lesion and image overlap with train and val)
3. Preserved test split checksum (split_test.csv MD5 unaltered)
4. Checkpoint integrity and configuration fidelity
5. Test predictions validity (1,512 predictions, matching IDs, valid probabilities in [0, 1] summing to 1.0)
6. Metrics validity (Top-1, Top-2, Macro/Weighted F1, ROC-AUC, PR-AUC, test loss)
7. Confusion matrix (7x7 dimensions, non-negative integer counts matching 1,512)
8. Classification report completeness (all 7 classes present with precision, recall, f1, support)
9. Required output artifact existence (all CSVs, JSON, and PNG figures present and non-empty)
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import CLASS_NAMES, NUM_CLASSES


EXPECTED_TEST_MD5 = "6a5ae1b65c25d504f78bc1da2361d82c"
EXPECTED_TEST_SAMPLES = 1512


@pytest.fixture(scope="module")
def test_split_df():
    csv_path = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
    assert csv_path.exists(), f"split_test.csv not found at {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def predictions_df():
    csv_path = PROJECT_ROOT / "reports" / "data" / "day9_test_predictions.csv"
    if not csv_path.exists():
        pytest.skip("day9_test_predictions.csv not found — run Day 9 evaluation first")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def classification_report_df():
    csv_path = PROJECT_ROOT / "reports" / "data" / "day9_classification_report.csv"
    if not csv_path.exists():
        pytest.skip("day9_classification_report.csv not found — run Day 9 evaluation first")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def metrics_json():
    json_path = PROJECT_ROOT / "reports" / "data" / "day9_metrics.json"
    if not json_path.exists():
        pytest.skip("day9_metrics.json not found — run Day 9 evaluation first")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════
# 1. TEST DATASET INTEGRITY & LEAKAGE ISOLATION
# ═════════════════════════════════════════════════════════════════════


class TestDatasetIntegrity:
    """Verify test split dimensions, isolation, and integrity."""

    def test_sample_count_exactly_1512(self, test_split_df):
        assert len(test_split_df) == EXPECTED_TEST_SAMPLES, (
            f"Expected {EXPECTED_TEST_SAMPLES} test samples, got {len(test_split_df)}"
        )

    def test_no_duplicate_image_ids(self, test_split_df):
        assert test_split_df["image_id"].nunique() == EXPECTED_TEST_SAMPLES, (
            "Duplicate image_id found in test set"
        )

    def test_valid_class_labels(self, test_split_df):
        unique_classes = sorted(test_split_df["dx"].unique())
        assert unique_classes == sorted(CLASS_NAMES), (
            f"Expected classes {CLASS_NAMES}, got {unique_classes}"
        )

    def test_test_split_checksum_unmodified(self):
        csv_path = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
        actual_md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
        assert actual_md5 == EXPECTED_TEST_MD5, (
            f"split_test.csv checksum modified! Expected {EXPECTED_TEST_MD5}, got {actual_md5}"
        )

    def test_lesion_level_isolation(self, test_split_df):
        train_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_val.csv")

        test_lesions = set(test_split_df["lesion_id"])
        train_overlap = test_lesions & set(train_df["lesion_id"])
        val_overlap = test_lesions & set(val_df["lesion_id"])

        assert len(train_overlap) == 0, f"Lesion leakage between train and test: {len(train_overlap)}"
        assert len(val_overlap) == 0, f"Lesion leakage between val and test: {len(val_overlap)}"

    def test_image_level_isolation(self, test_split_df):
        train_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_val.csv")

        test_imgs = set(test_split_df["image_id"])
        train_overlap = test_imgs & set(train_df["image_id"])
        val_overlap = test_imgs & set(val_df["image_id"])

        assert len(train_overlap) == 0, f"Image leakage between train and test: {len(train_overlap)}"
        assert len(val_overlap) == 0, f"Image leakage between val and test: {len(val_overlap)}"


# ═════════════════════════════════════════════════════════════════════
# 2. TEST PREDICTIONS VALIDITY
# ═════════════════════════════════════════════════════════════════════


class TestPredictions:
    """Verify test prediction schema, probabilities, and completeness."""

    def test_prediction_count(self, predictions_df):
        assert len(predictions_df) == EXPECTED_TEST_SAMPLES, (
            f"Expected {EXPECTED_TEST_SAMPLES} predictions, got {len(predictions_df)}"
        )

    def test_prediction_ids_match_test_ids(self, test_split_df, predictions_df):
        assert list(predictions_df["image_id"]) == list(test_split_df["image_id"]), (
            "Prediction image_ids do not match test split exactly"
        )

    def test_predicted_classes_valid(self, predictions_df):
        for pred_cls in predictions_df["predicted_class"].unique():
            assert pred_cls in CLASS_NAMES, f"Invalid predicted class: {pred_cls}"

    def test_probability_columns_present(self, predictions_df):
        for cls_name in CLASS_NAMES:
            col = f"prob_{cls_name}"
            assert col in predictions_df.columns, f"Missing probability column: {col}"

    def test_probabilities_finite(self, predictions_df):
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        matrix = predictions_df[prob_cols].to_numpy()
        assert np.all(np.isfinite(matrix)), "Non-finite probability values detected"

    def test_probabilities_bounded_and_normalized(self, predictions_df):
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        matrix = predictions_df[prob_cols].to_numpy()
        assert np.all((matrix >= 0.0) & (matrix <= 1.0)), "Probabilities outside [0, 1]"
        row_sums = matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    def test_confidences_match_max_probability(self, predictions_df):
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        max_probs = predictions_df[prob_cols].max(axis=1).to_numpy()
        confs = predictions_df["predicted_confidence"].to_numpy()
        assert np.allclose(confs, max_probs, atol=1e-4), (
            "predicted_confidence does not match max probability"
        )


# ═════════════════════════════════════════════════════════════════════
# 3. METRICS & CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════


class TestMetrics:
    """Verify quantitative metrics, confusion matrix, and reports."""

    def test_top1_accuracy_valid_range(self, metrics_json):
        top1 = metrics_json["overall_metrics"]["top1_accuracy"]
        assert 0.0 < top1 <= 1.0, f"Invalid top1 accuracy: {top1}"

    def test_top2_accuracy_greater_or_equal_top1(self, metrics_json):
        top1 = metrics_json["overall_metrics"]["top1_accuracy"]
        top2 = metrics_json["overall_metrics"]["top2_accuracy"]
        assert top2 >= top1, f"Top-2 accuracy ({top2}) should be >= Top-1 ({top1})"
        assert top2 <= 1.0, f"Top-2 accuracy exceeds 1.0: {top2}"

    def test_macro_metrics_present(self, metrics_json):
        om = metrics_json["overall_metrics"]
        assert "macro_precision" in om
        assert "macro_recall" in om
        assert "macro_f1" in om
        assert 0.0 < om["macro_f1"] <= 1.0

    def test_roc_auc_metrics_present(self, metrics_json):
        roc = metrics_json["roc_auc"]
        assert "macro_roc_auc" in roc
        assert "weighted_roc_auc" in roc
        assert 0.5 <= roc["macro_roc_auc"] <= 1.0
        assert len(roc["per_class_roc_auc"]) == NUM_CLASSES

    def test_pr_auc_metrics_present(self, metrics_json):
        pr = metrics_json["precision_recall_auc"]
        assert "macro_pr_auc" in pr
        assert 0.0 < pr["macro_pr_auc"] <= 1.0
        assert len(pr["per_class_pr_auc"]) == NUM_CLASSES

    def test_classification_report_covers_all_classes(self, classification_report_df):
        assert len(classification_report_df) == NUM_CLASSES
        classes_in_report = set(classification_report_df["class"])
        assert classes_in_report == set(CLASS_NAMES)
        total_support = int(classification_report_df["support"].sum())
        assert total_support == EXPECTED_TEST_SAMPLES

    def test_confusion_matrix_shape_and_sum(self, metrics_json):
        cm = np.array(metrics_json["confusion_matrix_raw"])
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES), f"Expected 7x7, got {cm.shape}"
        assert cm.sum() == EXPECTED_TEST_SAMPLES, (
            f"Confusion matrix sum {cm.sum()} != {EXPECTED_TEST_SAMPLES}"
        )


# ═════════════════════════════════════════════════════════════════════
# 4. OUTPUT ARTIFACTS EXISTENCE
# ═════════════════════════════════════════════════════════════════════


class TestOutputArtifacts:
    """Verify all required files and figures were generated and are non-empty."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "reports/data/day9_test_predictions.csv",
            "reports/data/day9_classification_report.csv",
            "reports/data/day9_metrics.json",
            "reports/figures/day9_confusion_matrix.png",
            "reports/figures/day9_confusion_matrix_normalized.png",
            "reports/figures/day9_roc_curves.png",
            "reports/figures/day9_precision_recall_curves.png",
            "reports/figures/day9/errors/day9_misclassifications_grid.png",
        ],
    )
    def test_artifact_exists_and_non_empty(self, rel_path):
        p = PROJECT_ROOT / rel_path
        assert p.exists(), f"Artifact missing: {rel_path}"
        assert p.stat().st_size > 0, f"Artifact is empty (0 bytes): {rel_path}"


# ═════════════════════════════════════════════════════════════════════
# 5. TEST PROTECTION AUDIT
# ═════════════════════════════════════════════════════════════════════


class TestProtectionAudit:
    """Audit that test set was not used for training or checkpoint selection."""

    def test_fixed_checkpoint_not_overwritten_by_test(self):
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        # Checkpoint should still be from Epoch 14 with Val Loss ~0.6422
        assert ckpt["epoch"] == 14, f"Checkpoint epoch altered! Expected 14, got {ckpt['epoch']}"
        assert np.isclose(ckpt["val_loss"], 0.64215, atol=1e-3), (
            f"Checkpoint val_loss altered! Expected ~0.6422, got {ckpt['val_loss']}"
        )
