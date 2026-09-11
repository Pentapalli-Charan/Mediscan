"""
MediScan — Day 5: EfficientNet-B0 Base Model Tests

Comprehensive test suite covering:
    - EfficientNet-B0 initialization from config and factory functions
    - Pretrained ImageNet weight loading
    - Classifier head structure and output dimension (7 classes)
    - Output tensor shape [B, 7] for various batch sizes
    - Forward pass produces finite raw logits (no NaN, no Inf)
    - Input tensor shape acceptance [B, 3, 224, 224]
    - Transfer learning backbone freezing (requires_grad=False on features)
    - Classifier head trainability (requires_grad=True on classifier)
    - Exact parameter counts (total, trainable, frozen, percentage)
    - CrossEntropyLoss calculation compatibility
    - CPU forward pass execution
    - CUDA forward pass execution (if GPU is available)
    - Forward pass on real DataLoader batch
    - Initialization checkpoint saving and loading integrity
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloaders
from src.data.preprocessing_day3 import load_config
from src.models import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_model,
    count_parameters,
    freeze_backbone,
    get_efficientnet_b0,
    get_model_summary,
    save_initialization_checkpoint,
    unfreeze_backbone,
)
from src.utils.device import get_device


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def config():
    """Load project configuration."""
    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))


@pytest.fixture(scope="module")
def base_model():
    """Build a standard base EfficientNet-B0 model once for the test module."""
    model = build_model()
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════
# 1. MODEL INITIALIZATION & CONFIG TESTS
# ═════════════════════════════════════════════════════════════════════


class TestModelInitialization:
    """Verify EfficientNet-B0 instantiates properly with correct settings."""

    def test_build_model_default(self, base_model):
        """Model built with defaults should be an nn.Module with correct architecture."""
        assert isinstance(base_model, nn.Module)
        assert base_model.architecture == "efficientnet_b0"
        assert base_model.num_classes == 7

    def test_build_model_from_config(self, config):
        """Model built using config dictionary matches config settings."""
        model = build_model(config)
        assert model.architecture == config["model"]["architecture"]
        assert model.num_classes == config["model"]["num_classes"]

    def test_get_efficientnet_b0_direct(self):
        """Direct factory instantiation works with explicit parameters."""
        model = get_efficientnet_b0(
            num_classes=7,
            pretrained=True,
            freeze_backbone_weights=True,
            dropout=0.2,
        )
        assert model.num_classes == 7
        assert model.backbone_frozen is True

    def test_weights_loaded_attribute(self, base_model):
        """Verify weights attribute indicates ImageNet pretrained weights."""
        assert "IMAGENET1K" in str(base_model.weights_used) or "DEFAULT" in str(base_model.weights_used)

    def test_unsupported_architecture_raises_error(self):
        """Unsupported architectures should raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            build_model(architecture="non_existent_net")


# ═════════════════════════════════════════════════════════════════════
# 2. CLASSIFIER HEAD & OUTPUT DIMENSION TESTS
# ═════════════════════════════════════════════════════════════════════


class TestClassifierHead:
    """Verify classifier head structure and output dimensions."""

    def test_classifier_is_sequential(self, base_model):
        """Classifier head must be an nn.Sequential module."""
        assert isinstance(base_model.classifier, nn.Sequential)

    def test_classifier_output_features(self, base_model):
        """Classifier final Linear layer out_features must equal 7."""
        linear_layer = base_model.classifier[1]
        assert isinstance(linear_layer, nn.Linear)
        assert linear_layer.out_features == 7
        assert linear_layer.in_features == 1280

    def test_classifier_dropout(self, base_model):
        """Classifier first layer must be nn.Dropout with p=0.2."""
        dropout_layer = base_model.classifier[0]
        assert isinstance(dropout_layer, nn.Dropout)
        assert dropout_layer.p == 0.2

    def test_num_classes_constant(self):
        """Verify NUM_CLASSES constant is 7 and matches class names length."""
        assert NUM_CLASSES == 7
        assert len(CLASS_NAMES) == 7
        assert CLASS_NAMES == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


# ═════════════════════════════════════════════════════════════════════
# 3. TRANSFER LEARNING & PARAMETER FREEZING TESTS
# ═════════════════════════════════════════════════════════════════════


class TestTransferLearningFreezing:
    """Verify backbone freezing and parameter counting."""

    def test_backbone_parameters_frozen(self, base_model):
        """All parameters in model.features must have requires_grad=False."""
        feature_params = list(base_model.features.parameters())
        assert len(feature_params) > 0
        for i, param in enumerate(feature_params):
            assert not param.requires_grad, f"Feature parameter index {i} is not frozen!"

    def test_classifier_parameters_trainable(self, base_model):
        """All parameters in model.classifier must have requires_grad=True."""
        classifier_params = list(base_model.classifier.parameters())
        assert len(classifier_params) > 0
        for i, param in enumerate(classifier_params):
            assert param.requires_grad, f"Classifier parameter index {i} is not trainable!"

    def test_exact_parameter_counts(self, base_model):
        """Verify exact parameter counts for base EfficientNet-B0 with 7 classes."""
        counts = count_parameters(base_model)
        # EfficientNet-B0 backbone: 4,007,548 params
        # Linear head (1280 * 7 weights + 7 biases): 8,967 params
        # Total: 4,016,515 params
        assert counts["total"] == 4016515, f"Expected total 4,016,515, got {counts['total']}"
        assert counts["trainable"] == 8967, f"Expected trainable 8,967, got {counts['trainable']}"
        assert counts["frozen"] == 4007548, f"Expected frozen 4,007,548, got {counts['frozen']}"
        assert abs(counts["trainable_pct"] - 0.2233) < 0.01

    def test_unfreeze_and_refreeze_utilities(self):
        """Verify unfreeze_backbone and freeze_backbone utilities."""
        model = get_efficientnet_b0(pretrained=False, freeze_backbone_weights=True)
        counts_init = count_parameters(model)
        assert counts_init["trainable"] == 8967

        # Unfreeze all
        unfreeze_backbone(model)
        counts_unfrozen = count_parameters(model)
        assert counts_unfrozen["trainable"] == counts_unfrozen["total"]
        assert counts_unfrozen["frozen"] == 0

        # Refreeze backbone
        freeze_backbone(model)
        counts_refrozen = count_parameters(model)
        assert counts_refrozen["trainable"] == 8967
        assert counts_refrozen["frozen"] == 4007548


# ═════════════════════════════════════════════════════════════════════
# 4. FORWARD PASS & SHAPE TESTS
# ═════════════════════════════════════════════════════════════════════


class TestForwardPass:
    """Verify forward pass behavior on synthetic and real inputs."""

    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_forward_pass_synthetic(self, base_model, batch_size):
        """Forward pass on [B, 3, 224, 224] returns [B, 7] finite logits."""
        x = torch.randn(batch_size, 3, 224, 224)
        with torch.no_grad():
            logits = base_model(x)

        assert logits.shape == (batch_size, 7)
        assert logits.dtype == torch.float32
        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"

    def test_forward_pass_real_dataloader_batch(self, base_model):
        """Forward pass on an actual batch from the training DataLoader."""
        loaders = create_dataloaders()
        images, labels = next(iter(loaders["train"]))

        with torch.no_grad():
            logits = base_model(images)

        assert logits.shape == (images.shape[0], 7)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_pass_raw_logits_no_softmax(self, base_model):
        """Logits must NOT sum to 1.0 (indicating raw logits, not softmax)."""
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            logits = base_model(x)

        # Softmax outputs sum to 1.0 per row. Raw logits almost certainly will not.
        row_sums = logits.sum(dim=-1)
        assert not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-2), (
            "Model output appears to have softmax applied; expected raw logits."
        )


# ═════════════════════════════════════════════════════════════════════
# 5. LOSS FUNCTION COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════


class TestLossFunctionCompatibility:
    """Verify logits are compatible with torch.nn.CrossEntropyLoss."""

    def test_cross_entropy_loss_synthetic(self, base_model):
        """CrossEntropyLoss computes cleanly on synthetic batch."""
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 2, 4, 6], dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = base_model(x)
            loss = criterion(logits, y)

        assert loss.dim() == 0, "Loss should be a scalar tensor"
        assert not torch.isnan(loss), "Loss is NaN"
        assert not torch.isinf(loss), "Loss is Inf"
        assert loss.item() > 0.0, "Loss must be positive"

    def test_cross_entropy_loss_real_batch(self, base_model):
        """CrossEntropyLoss computes cleanly on a real batch from train_loader."""
        loaders = create_dataloaders()
        images, labels = next(iter(loaders["train"]))

        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = base_model(images)
            loss = criterion(logits, labels)

        assert loss.item() > 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ═════════════════════════════════════════════════════════════════════
# 6. DEVICE COMPATIBILITY
# ═════════════════════════════════════════════════════════════════════


class TestDeviceCompatibility:
    """Verify model can be moved to detected device and forward pass succeeds."""

    def test_cpu_device_forward(self, base_model):
        """Model runs successfully on CPU."""
        device = torch.device("cpu")
        base_model.to(device)
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = base_model(x)
        assert out.device.type == "cpu"
        assert out.shape == (2, 7)

    def test_detected_device_forward(self, base_model):
        """Model runs on device detected by get_device()."""
        device_str = get_device()
        device = torch.device(device_str)
        base_model.to(device)
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = base_model(x)
        assert out.device.type == device_str
        assert out.shape == (2, 7)

    def test_cuda_if_available(self, base_model):
        """If CUDA is available, verify GPU forward pass; otherwise skip cleanly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA NOT AVAILABLE on this system")
        device = torch.device("cuda")
        base_model.to(device)
        x = torch.randn(2, 3, 224, 224, device=device)
        with torch.no_grad():
            out = base_model(x)
        assert out.device.type == "cuda"
        assert out.shape == (2, 7)
        # Clean up back to CPU
        base_model.to("cpu")


# ═════════════════════════════════════════════════════════════════════
# 7. MODEL SUMMARY & CHECKPOINT INTEGRITY
# ═════════════════════════════════════════════════════════════════════


class TestSummaryAndCheckpoint:
    """Verify model summary generation and initialization checkpoint saving."""

    def test_get_model_summary(self, base_model):
        """get_model_summary returns a comprehensive metadata dictionary."""
        summary = get_model_summary(base_model)
        assert summary["architecture"] == "efficientnet_b0"
        assert summary["num_classes"] == 7
        assert summary["total_parameters"] == 4016515
        assert summary["trainable_parameters"] == 8967
        assert summary["frozen_parameters"] == 4007548
        assert summary["input_shape"] == [1, 3, 224, 224]
        assert summary["output_shape"] == [1, 7]
        assert summary["backbone_frozen"] is True

    def test_save_and_load_initialization_checkpoint(self, base_model):
        """Initialization checkpoint saves cleanly and can be loaded back."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_init.pth"
            saved_path = save_initialization_checkpoint(base_model, ckpt_path)
            assert saved_path.exists()

            # Load checkpoint
            checkpoint = torch.load(str(saved_path), map_location="cpu")
            assert checkpoint["checkpoint_type"] == "initialization_pretrained"
            assert "model_state_dict" in checkpoint

            # Verify parameters load into fresh model
            new_model = get_efficientnet_b0(pretrained=False, freeze_backbone_weights=True)
            new_model.load_state_dict(checkpoint["model_state_dict"])

            # Verify same outputs on synthetic input
            x = torch.randn(2, 3, 224, 224)
            base_model.eval()
            new_model.eval()
            with torch.no_grad():
                out_base = base_model(x)
                out_new = new_model(x)
            assert torch.allclose(out_base, out_new, atol=1e-5)
