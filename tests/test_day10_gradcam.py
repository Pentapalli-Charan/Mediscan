"""
MediScan — Day 10: Grad-CAM Explainability & Visual Heatmaps Tests

Verification suite covering:
1. Verification of target convolutional layer existence and architecture (model.features[8])
2. Grad-CAM execution and activation map generation
3. Numerical validation: CAM shape [224, 224], finiteness, [0.0, 1.0] normalization
4. Overlay dimension [224, 224, 3] and uint8 dtype verification
5. Multi-class target support: all 7 diagnostic classes handled without error
6. Hook lifecycle management: hooks properly detached after use
7. Gradient safety & weight immutability: parameter weights bitwise identical before and after
8. Required output artifacts existence (individual triplet figures, consolidated grids, JSON/CSV logs)
9. Preservation of Day 9 test predictions and baseline metrics
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import load_config
from src.explainability.gradcam import (
    GradCAM,
    get_default_target_layer,
    snapshot_model_weights,
    verify_model_weights_unchanged,
)
from src.explainability.visualization import create_gradcam_overlay
from src.models import CLASS_NAMES, NUM_CLASSES, build_model
from src.training.trainer import load_training_checkpoint


@pytest.fixture(scope="module")
def baseline_model():
    """Load the validation-selected EfficientNet-B0 model checkpoint."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(str(config_path))
    model = build_model(config, architecture="efficientnet_b0", freeze_backbone=True)
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    if not ckpt_path.exists():
        pytest.skip("Baseline checkpoint efficientnet_b0_best.pth not found")
    load_training_checkpoint(ckpt_path, model, device="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_tensor():
    """Deterministic synthetic input tensor."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


# ═════════════════════════════════════════════════════════════════════
# 1. TARGET LAYER ARCHITECTURE
# ═════════════════════════════════════════════════════════════════════


class TestTargetLayerArchitecture:
    """Verify target convolutional layer location and tensor output dimensions."""

    def test_target_layer_exists(self, baseline_model):
        target_layer = get_default_target_layer(baseline_model)
        assert target_layer is not None, "Target layer could not be found."
        assert target_layer == baseline_model.features[8]

    def test_target_layer_output_channels(self, baseline_model, sample_tensor):
        activations = []
        target_layer = get_default_target_layer(baseline_model)

        def hook(m, i, o):
            activations.append(o)

        h = target_layer.register_forward_hook(hook)
        with torch.no_grad():
            _ = baseline_model(sample_tensor)
        h.remove()

        assert len(activations) == 1
        act = activations[0]
        assert act.shape == torch.Size([1, 1280, 7, 7]), (
            f"Expected shape [1, 1280, 7, 7], got {act.shape}"
        )


# ═════════════════════════════════════════════════════════════════════
# 2. GRAD-CAM NUMERICAL VALIDATION
# ═════════════════════════════════════════════════════════════════════


class TestGradCAMNumericalValidation:
    """Verify CAM generation, shapes, finiteness, and bounds."""

    def test_cam_dimensions_match_input(self, baseline_model, sample_tensor):
        with GradCAM(baseline_model) as gradcam:
            cam, pred_idx, conf, probs = gradcam.generate_cam(sample_tensor, target_class=0)
            assert cam.shape == (224, 224), f"Expected CAM shape (224, 224), got {cam.shape}"
            assert isinstance(pred_idx, int)
            assert 0 <= pred_idx < NUM_CLASSES
            assert 0.0 <= conf <= 1.0
            assert len(probs) == NUM_CLASSES

    def test_cam_finite_no_nan_no_inf(self, baseline_model, sample_tensor):
        with GradCAM(baseline_model) as gradcam:
            cam, _, _, _ = gradcam.generate_cam(sample_tensor, target_class=4)
            assert not np.isnan(cam).any(), "CAM contains NaN values."
            assert not np.isinf(cam).any(), "CAM contains Inf values."

    def test_cam_normalization_bounds(self, baseline_model, sample_tensor):
        with GradCAM(baseline_model) as gradcam:
            cam, _, _, _ = gradcam.generate_cam(sample_tensor, target_class=5)
            assert float(cam.min()) >= 0.0, f"CAM min {cam.min()} < 0.0"
            assert float(cam.max()) <= 1.0, f"CAM max {cam.max()} > 1.0"
            assert np.isclose(float(cam.max()), 1.0, atol=1e-3)
            assert np.isclose(float(cam.min()), 0.0, atol=1e-3)

    def test_all_seven_class_targets_supported(self, baseline_model, sample_tensor):
        with GradCAM(baseline_model) as gradcam:
            for cls_idx in range(NUM_CLASSES):
                cam, _, _, _ = gradcam.generate_cam(sample_tensor, target_class=cls_idx)
                assert cam.shape == (224, 224)
                assert not np.isnan(cam).any()


# ═════════════════════════════════════════════════════════════════════
# 3. HOOK LIFECYCLE MANAGEMENT
# ═════════════════════════════════════════════════════════════════════


class TestHookLifecycle:
    """Verify hooks are removed and not leaked across executions."""

    def test_hooks_cleaned_up_on_exit(self, baseline_model):
        with GradCAM(baseline_model) as gradcam:
            assert len(gradcam.hooks) == 2
        # After context exit
        assert len(gradcam.hooks) == 0

    def test_explicit_remove_hooks(self, baseline_model):
        gradcam = GradCAM(baseline_model)
        assert len(gradcam.hooks) == 2
        gradcam.remove_hooks()
        assert len(gradcam.hooks) == 0


# ═════════════════════════════════════════════════════════════════════
# 4. OVERLAY GENERATION
# ═════════════════════════════════════════════════════════════════════


class TestOverlayGeneration:
    """Verify heatmap overlay blending and data types."""

    def test_overlay_shape_and_dtype(self):
        dummy_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_cam = np.ones((224, 224), dtype=np.float32) * 0.5
        overlay = create_gradcam_overlay(dummy_rgb, dummy_cam, alpha=0.45)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8
        assert overlay.max() > 0


# ═════════════════════════════════════════════════════════════════════
# 5. GRADIENT SAFETY & WEIGHT IMMUTABILITY
# ═════════════════════════════════════════════════════════════════════


class TestWeightImmutability:
    """Assert parameter tensors are bitwise identical before and after Grad-CAM."""

    def test_parameters_bitwise_unchanged_after_gradcam(self, baseline_model, sample_tensor):
        weights_before = snapshot_model_weights(baseline_model)

        with GradCAM(baseline_model) as gradcam:
            # Generate CAM on multiple classes
            _ = gradcam.generate_cam(sample_tensor, target_class=1)
            _ = gradcam.generate_cam(sample_tensor, target_class=4)

        # Assert zero change
        assert verify_model_weights_unchanged(weights_before, baseline_model)


# ═════════════════════════════════════════════════════════════════════
# 6. ARTIFACTS VERIFICATION
# ═════════════════════════════════════════════════════════════════════


class TestOutputArtifacts:
    """Verify existence and non-emptiness of generated Day 10 artifacts."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "reports/figures/day10_gradcam_examples.png",
            "reports/figures/day10_gradcam_malignant_errors.png",
            "reports/data/day10_gradcam_samples.csv",
            "reports/data/day10_validation.json",
        ],
    )
    def test_artifact_exists_and_non_empty(self, rel_path):
        p = PROJECT_ROOT / rel_path
        assert p.exists(), f"Artifact missing: {rel_path}"
        assert p.stat().st_size > 0, f"Artifact is empty: {rel_path}"

    def test_gradcam_figures_directory_has_files(self):
        fig_dir = PROJECT_ROOT / "reports" / "figures" / "gradcam"
        assert fig_dir.exists(), "reports/figures/gradcam/ directory missing"
        pngs = list(fig_dir.glob("*.png"))
        assert len(pngs) >= 10, f"Expected at least 10 individual figures, found {len(pngs)}"

    def test_day9_predictions_unaltered(self):
        day9_pred = PROJECT_ROOT / "reports" / "data" / "day9_test_predictions.csv"
        assert day9_pred.exists()
        df = pd.read_csv(day9_pred)
        assert len(df) == 1512, "Day 9 test predictions row count altered!"
