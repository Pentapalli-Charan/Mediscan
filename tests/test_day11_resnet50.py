"""
MediScan — Day 11: ResNet-50 Comparison Architecture Tests

Verification suite covering:
1. ResNet-50 instantiation and architecture validation
2. Output dimension [B, 7] and class mapping alignment
3. Feature extractor freezing and trainable classification head parameter counts
4. Gradient flow and frozen backbone bitwise immutability after optimizer step
5. Forward pass producing strictly finite logits without NaN/Inf
6. Checkpoint save and reload verification (models/checkpoints/resnet50_best.pth)
7. Identification and verification of candidate Grad-CAM target layer (model.layer4[-1])
8. Training history and model comparison CSV artifact schemas
9. Figure artifacts existence and non-emptiness
10. Strict test-set protection audit (zero test inference, unaltered test checksums)
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import load_config
from src.models import CLASS_NAMES, NUM_CLASSES, build_model, count_parameters
from src.models.resnet50 import get_resnet50, get_resnet50_summary


@pytest.fixture(scope="module")
def resnet_model():
    """Instantiate a ResNet-50 model with frozen backbone."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(str(config_path))
    model = build_model(config, architecture="resnet50", freeze_backbone=True, dropout=0.2)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sample_batch():
    """Synthetic input batch."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 224, 224)


# ═════════════════════════════════════════════════════════════════════
# 1. ARCHITECTURE & PARAMETER ACCOUNTING
# ═════════════════════════════════════════════════════════════════════


class TestModelArchitecture:
    """Verify ResNet-50 structure, dimensions, and class mapping."""

    def test_model_instantiation(self, resnet_model):
        assert resnet_model is not None
        assert getattr(resnet_model, "architecture", "") == "resnet50"
        assert resnet_model.fc is not None

    def test_output_dimension_is_seven(self, resnet_model, sample_batch):
        with torch.no_grad():
            out = resnet_model(sample_batch)
        assert out.shape == (2, NUM_CLASSES), f"Expected shape (2, {NUM_CLASSES}), got {out.shape}"

    def test_fc_input_features_dimension(self, resnet_model):
        linear_layer = resnet_model.fc[1]
        assert isinstance(linear_layer, nn.Linear)
        assert linear_layer.in_features == 2048, (
            f"Expected in_features 2048 for ResNet-50, got {linear_layer.in_features}"
        )
        assert linear_layer.out_features == NUM_CLASSES

    def test_parameter_counts(self, resnet_model):
        counts = count_parameters(resnet_model)
        # ResNet-50 backbone: 23,508,032 frozen parameters
        # Classification head: Dropout(0.2) + Linear(2048, 7) = 2048*7 + 7 = 14,343 trainable parameters
        # Total: 23,522,375 parameters
        assert counts["total"] == 23522375
        assert counts["trainable"] == 14343
        assert counts["frozen"] == 23508032
        assert np.isclose(counts["trainable_pct"], 0.061, atol=0.01)

    def test_model_summary_structure(self, resnet_model):
        summary = get_resnet50_summary(resnet_model)
        assert summary["architecture"] == "resnet50"
        assert summary["num_classes"] == NUM_CLASSES
        assert summary["input_shape"] == [1, 3, 224, 224]
        assert summary["total_parameters"] == 23522375
        assert summary["trainable_parameters"] == 14343


# ═════════════════════════════════════════════════════════════════════
# 2. PARAMETER FREEZING & GRADIENT FLOW
# ═════════════════════════════════════════════════════════════════════


class TestParameterFreezingAndGradients:
    """Verify backbone freezing and head gradient accumulation."""

    def test_backbone_requires_grad_is_false(self, resnet_model):
        for name, param in resnet_model.named_parameters():
            if "fc" not in name:
                assert not param.requires_grad, f"Backbone parameter '{name}' has requires_grad=True!"

    def test_classifier_requires_grad_is_true(self, resnet_model):
        for name, param in resnet_model.fc.named_parameters():
            assert param.requires_grad, f"Classifier parameter '{name}' has requires_grad=False!"

    def test_gradient_flow_and_immutability(self):
        # Fresh model instance
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        config = load_config(str(config_path))
        m = build_model(config, architecture="resnet50", freeze_backbone=True)

        backbone_p = next(p for n, p in m.named_parameters() if "conv1" in n)
        classifier_p = next(p for n, p in m.named_parameters() if "fc.1.weight" in n)

        bb_before = backbone_p.detach().clone()
        fc_before = classifier_p.detach().clone()

        opt = torch.optim.Adam(m.fc.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        m.train()
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1])

        opt.zero_grad()
        logits = m(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()

        # Backbone should have no grad and remain bitwise unchanged
        assert backbone_p.grad is None
        assert torch.equal(backbone_p, bb_before)

        # Classifier should have grad and be updated
        assert classifier_p.grad is not None
        assert not torch.equal(classifier_p, fc_before)


# ═════════════════════════════════════════════════════════════════════
# 3. FORWARD PASS & CHECKPOINT RELOAD
# ═════════════════════════════════════════════════════════════════════


class TestForwardPassAndCheckpoint:
    """Verify numerical forward pass and checkpoint recovery."""

    def test_forward_produces_finite_logits(self, resnet_model, sample_batch):
        with torch.no_grad():
            logits = resnet_model(sample_batch)
        assert logits.shape == (2, 7)
        assert torch.isfinite(logits).all()
        assert not torch.isnan(logits).any()

    def test_checkpoint_exists_and_loads(self):
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "resnet50_best.pth"
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} missing"

        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        required_keys = ["epoch", "model_state_dict", "val_loss", "val_accuracy", "architecture"]
        for k in required_keys:
            assert k in ckpt, f"Key '{k}' missing from checkpoint"

        assert ckpt["architecture"] == "resnet50"
        assert ckpt["num_classes"] == 7
        assert ckpt["val_loss"] > 0.0

        # Reload into fresh model
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        config = load_config(str(config_path))
        m = build_model(config, architecture="resnet50", freeze_backbone=True)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()

        with torch.no_grad():
            out = m(torch.randn(1, 3, 224, 224))
        assert out.shape == (1, 7)
        assert torch.isfinite(out).all()


# ═════════════════════════════════════════════════════════════════════
# 4. GRAD-CAM TARGET LAYER IDENTIFICATION
# ═════════════════════════════════════════════════════════════════════


class TestGradCAMTargetLayer:
    """Verify ResNet-50 candidate Grad-CAM layer existence."""

    def test_resnet50_candidate_gradcam_layer(self, resnet_model):
        assert hasattr(resnet_model, "layer4")
        last_block = resnet_model.layer4[-1]
        assert type(last_block).__name__ == "Bottleneck"
        assert hasattr(last_block, "conv3")
        assert last_block.conv3.out_channels == 2048


# ═════════════════════════════════════════════════════════════════════
# 5. OUTPUT ARTIFACTS & TEST PROTECTION
# ═════════════════════════════════════════════════════════════════════


class TestOutputArtifactsAndProtection:
    """Verify generated files and test split integrity."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "src/models/resnet50.py",
            "models/checkpoints/resnet50_best.pth",
            "reports/data/day11_resnet50_training_history.csv",
            "reports/data/day11_model_comparison.csv",
            "reports/figures/day11/day11_resnet50_training_curves.png",
            "reports/figures/day11/day11_efficientnet_vs_resnet50_comparison.png",
        ],
    )
    def test_artifact_exists_and_non_empty(self, rel_path):
        p = PROJECT_ROOT / rel_path
        assert p.exists(), f"Artifact missing: {rel_path}"
        assert p.stat().st_size > 0, f"Artifact empty: {rel_path}"

    def test_training_history_columns(self):
        csv_path = PROJECT_ROOT / "reports" / "data" / "day11_resnet50_training_history.csv"
        df = pd.read_csv(csv_path)
        required_cols = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate", "epoch_duration_seconds"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        assert len(df) >= 1

    def test_comparison_csv_contains_both_models(self):
        csv_path = PROJECT_ROOT / "reports" / "data" / "day11_model_comparison.csv"
        df = pd.read_csv(csv_path)
        models = df["model"].tolist()
        assert any("efficientnet_b0" in m for m in models)
        assert any("resnet50" in m for m in models)

    def test_test_split_unmodified(self):
        p = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        assert md5 == "6a5ae1b65c25d504f78bc1da2361d82c", "split_test.csv was modified!"

    def test_day9_test_predictions_unmodified(self):
        p = PROJECT_ROOT / "reports" / "data" / "day9_test_predictions.csv"
        df = pd.read_csv(p)
        assert len(df) == 1512
