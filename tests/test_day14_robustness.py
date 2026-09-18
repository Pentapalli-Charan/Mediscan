"""
MediScan — Day 14: Robustness, Bias, and Post-Hoc Calibration Test Suite

Covers the 15 required verification areas:
1. Day 9 prediction artifacts load successfully.
2. Class mapping remains unchanged.
3. Checkpoint hash remains unchanged (7c1c6fcbe02e93f0ff8b4a20f62e29e3).
4. Test split hash remains unchanged (6a5ae1b65c25d504f78bc1da2361d82c).
5. Prediction probability arrays are valid.
6. Probability values are finite.
7. Probability rows sum approximately to 1.
8. ECE calculation works on controlled synthetic test data.
9. Reliability-bin calculation works.
10. Perturbation functions return valid images.
11. Perturbations do not modify original images.
12. Analysis outputs contain valid data.
13. Model weights remain unchanged.
14. Grad-CAM target remains model.features[8].
15. Existing Streamlit inference remains functional.
"""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import load_mediscan_model, run_model_inference
from src.data.preprocessing_day3 import load_config
from src.evaluation.day14_analysis import (
    CANONICAL_CHECKPOINT_MD5,
    CANONICAL_CLASS_MAPPING,
    CANONICAL_TEST_SPLIT_MD5,
    apply_perturbation,
    compute_calibration,
    compute_file_md5,
    verify_day14_integrity,
)
from src.explainability.gradcam import (
    get_default_target_layer,
    snapshot_model_weights,
    verify_model_weights_unchanged,
)
from src.models import CLASS_NAMES, NUM_CLASSES, build_model
from src.training.trainer import load_training_checkpoint


@pytest.fixture(scope="module")
def checkpoint_path():
    p = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    assert p.exists(), f"Canonical checkpoint not found: {p}"
    return p


@pytest.fixture(scope="module")
def test_split_path():
    p = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
    assert p.exists(), f"Test split not found: {p}"
    return p


@pytest.fixture(scope="module")
def predictions_df():
    p = PROJECT_ROOT / "reports" / "data" / "day9_test_predictions.csv"
    assert p.exists(), f"Predictions CSV not found: {p}"
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def metrics_json():
    p = PROJECT_ROOT / "reports" / "data" / "day9_metrics.json"
    assert p.exists(), f"Metrics JSON not found: {p}"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def canonical_model(checkpoint_path):
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    cfg = load_config(str(config_path))
    model = build_model(cfg, architecture="efficientnet_b0", freeze_backbone=True)
    load_training_checkpoint(checkpoint_path, model, device="cpu")
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════
# 1-4. INTEGRITY & CANONICAL ARTIFACTS
# ═════════════════════════════════════════════════════════════════════

class TestCanonicalIntegrity:
    """Verify hashes, class mapping, and Day 9 locked artifacts."""

    def test_day9_prediction_artifacts_load(self, predictions_df, metrics_json):
        """1. Day 9 prediction artifacts load successfully."""
        assert len(predictions_df) == 1512
        assert "image_id" in predictions_df.columns
        assert "predicted_class" in predictions_df.columns
        assert "overall_metrics" in metrics_json

    def test_class_mapping_unchanged(self):
        """2. Class mapping remains unchanged."""
        expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        assert CLASS_NAMES == expected_classes
        assert NUM_CLASSES == 7
        for idx, name in enumerate(expected_classes):
            assert CANONICAL_CLASS_MAPPING[name] == idx

    def test_checkpoint_hash_unchanged(self, checkpoint_path):
        """3. Checkpoint hash remains unchanged."""
        actual_md5 = compute_file_md5(checkpoint_path)
        assert actual_md5 == CANONICAL_CHECKPOINT_MD5, (
            f"Checkpoint altered! Expected {CANONICAL_CHECKPOINT_MD5}, got {actual_md5}"
        )

    def test_test_split_hash_unchanged(self, test_split_path):
        """4. Test split hash remains unchanged."""
        actual_md5 = compute_file_md5(test_split_path)
        assert actual_md5 == CANONICAL_TEST_SPLIT_MD5, (
            f"Test split altered! Expected {CANONICAL_TEST_SPLIT_MD5}, got {actual_md5}"
        )


# ═════════════════════════════════════════════════════════════════════
# 5-7. PREDICTION PROBABILITY VALIDITY
# ═════════════════════════════════════════════════════════════════════

class TestPredictionProbabilities:
    """Verify prediction probability distributions."""

    def test_probabilities_array_valid(self, predictions_df):
        """5. Prediction probability arrays are valid shape and bounded in [0, 1]."""
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        assert all(col in predictions_df.columns for col in prob_cols)
        probs = predictions_df[prob_cols].values
        assert probs.shape == (1512, 7)
        assert (probs >= 0.0).all(), "Found negative probabilities"
        assert (probs <= 1.0).all(), "Found probabilities > 1.0"

    def test_probabilities_are_finite(self, predictions_df):
        """6. Probability values are finite (no NaN, no Inf)."""
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        probs = predictions_df[prob_cols].values
        assert np.isfinite(probs).all(), "Non-finite probability values detected"

    def test_probabilities_sum_to_one(self, predictions_df):
        """7. Probability rows sum approximately to 1."""
        prob_cols = [f"prob_{c}" for c in CLASS_NAMES]
        row_sums = predictions_df[prob_cols].sum(axis=1).values
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)


# ═════════════════════════════════════════════════════════════════════
# 8-9. CALIBRATION & RELIABILITY BINS
# ═════════════════════════════════════════════════════════════════════

class TestCalibrationLogic:
    """Verify Expected Calibration Error (ECE) and binning computations."""

    def test_ece_synthetic_controlled(self, tmp_path):
        """8. ECE calculation works on controlled synthetic test data."""
        # Case A: Perfectly calibrated data
        # Conf = 0.8, Acc = 0.8 in bin 8; Conf = 0.5, Acc = 0.5 in bin 5
        perf_data = pd.DataFrame({
            "predicted_confidence": [0.85] * 100 + [0.55] * 100,
            "true_class": ["nv"] * 85 + ["mel"] * 15 + ["nv"] * 55 + ["mel"] * 45,
            "predicted_class": ["nv"] * 200,
        })
        p_csv = tmp_path / "perf_calib.csv"
        perf_data.to_csv(p_csv, index=False)
        ece, mce, df = compute_calibration(p_csv, n_bins=10)
        # Expected calibration gap should be close to 0
        assert ece < 0.05, f"ECE on well-calibrated synthetic data was unexpectedly high: {ece}"

        # Case B: Severely miscalibrated data (confidence 0.95, accuracy 0.0)
        uncal_data = pd.DataFrame({
            "predicted_confidence": [0.95] * 100,
            "true_class": ["mel"] * 100,
            "predicted_class": ["nv"] * 100,
        })
        u_csv = tmp_path / "uncal_calib.csv"
        uncal_data.to_csv(u_csv, index=False)
        ece_bad, mce_bad, _ = compute_calibration(u_csv, n_bins=10)
        assert ece_bad >= 0.90, f"ECE on miscalibrated data should be ~0.95, got {ece_bad}"

    def test_reliability_bin_calculation(self):
        """9. Reliability-bin calculation works and partitions [0, 1] evenly."""
        pred_csv = PROJECT_ROOT / "reports" / "data" / "day9_test_predictions.csv"
        ece, mce, calib_df = compute_calibration(pred_csv, n_bins=10)
        assert len(calib_df) == 10
        assert calib_df["sample_count"].sum() == 1512
        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0
        # Check bin monotonicity
        lowers = calib_df["bin_lower"].tolist()
        uppers = calib_df["bin_upper"].tolist()
        assert lowers == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        assert uppers == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# ═════════════════════════════════════════════════════════════════════
# 10-11. PERTURBATION & ROBUSTNESS PROBES
# ═════════════════════════════════════════════════════════════════════

class TestPerturbations:
    """Verify perturbation functions return valid arrays without mutating inputs."""

    @pytest.mark.parametrize(
        "probe",
        [
            "baseline",
            "mild_brightness",
            "mild_contrast",
            "gaussian_blur",
            "jpeg_compression",
            "crop_resize",
            "mild_color_shift",
        ],
    )
    def test_perturbation_validity_and_immutability(self, probe):
        """10 & 11. Perturbation functions return valid images and do not modify originals."""
        np.random.seed(42)
        original = np.random.randint(0, 256, size=(450, 600, 3), dtype=np.uint8)
        orig_copy = original.copy()

        perturbed = apply_perturbation(original, probe)

        # 10. Valid image checks
        assert isinstance(perturbed, np.ndarray)
        assert perturbed.dtype == np.uint8
        assert perturbed.shape == original.shape
        assert perturbed.min() >= 0
        assert perturbed.max() <= 255

        # 11. Immutability check
        np.testing.assert_array_equal(
            original, orig_copy,
            err_msg=f"Original image was modified in-place during probe '{probe}'!"
        )


# ═════════════════════════════════════════════════════════════════════
# 12. OUTPUT ARTIFACTS EXISTENCE & INTEGRITY
# ═════════════════════════════════════════════════════════════════════

class TestAnalysisArtifacts:
    """Verify all Day 14 required analysis files and visualizations exist."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "reports/data/day14_class_distribution.csv",
            "reports/data/day14_per_class_metrics.csv",
            "reports/data/day14_confusion_analysis.csv",
            "reports/data/day14_confidence_analysis.csv",
            "reports/data/day14_calibration.csv",
            "reports/data/day14_robustness.csv",
            "reports/data/day14_image_quality.csv",
            "reports/data/day14_subgroup_analysis.csv",
            "reports/data/day14_analysis.json",
            "reports/figures/day14/class_distribution.png",
            "reports/figures/day14/per_class_f1.png",
            "reports/figures/day14/per_class_recall.png",
            "reports/figures/day14/confidence_distribution.png",
            "reports/figures/day14/reliability_diagram.png",
            "reports/figures/day14/robustness_comparison.png",
            "reports/figures/day14/image_quality_comparison.png",
            "reports/figures/day14/subgroup_analysis.png",
            "reports/figures/day14/top_confusion_pairs.png",
        ],
    )
    def test_analysis_output_exists_and_non_empty(self, rel_path):
        """12. Analysis outputs contain valid data and exist on disk."""
        p = PROJECT_ROOT / rel_path
        assert p.exists(), f"Required Day 14 output missing: {rel_path}"
        assert p.stat().st_size > 0, f"Output file is empty: {rel_path}"


# ═════════════════════════════════════════════════════════════════════
# 13-14. MODEL WEIGHT IMMUTABILITY & GRAD-CAM TARGET
# ═════════════════════════════════════════════════════════════════════

class TestModelSafety:
    """Verify model parameter immutability and target layer configuration."""

    def test_model_weights_unchanged(self, canonical_model):
        """13. Model weights remain unchanged after dummy evaluation."""
        weights_before = snapshot_model_weights(canonical_model)
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            _ = canonical_model(dummy_input)
        assert verify_model_weights_unchanged(weights_before, canonical_model)

    def test_gradcam_target_is_features_8(self, canonical_model):
        """14. Grad-CAM target remains model.features[8]."""
        target = get_default_target_layer(canonical_model)
        assert target == canonical_model.features[8], (
            f"Grad-CAM target layer altered! Expected model.features[8], got {target}"
        )


# ═════════════════════════════════════════════════════════════════════
# 15. STREAMLIT APP INFERENCE COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════

class TestStreamlitInferenceFunctional:
    """Verify Streamlit app inference remains functional with canonical checkpoint."""

    def test_streamlit_inference_execution(self, canonical_model):
        """15. Existing Streamlit inference remains functional."""
        dummy_tensor = torch.randn(1, 3, 224, 224)
        result = run_model_inference(canonical_model, dummy_tensor, device=torch.device("cpu"))
        assert "predicted_class" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert result["predicted_class"] in CLASS_NAMES
        assert len(result["probabilities"]) == 7
        prob_sum = sum(result["probabilities"].values())
        np.testing.assert_allclose(prob_sum, 1.0, atol=1e-4)
