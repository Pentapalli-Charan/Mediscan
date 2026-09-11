"""
MediScan — Day 6: Training Pipeline Tests

Tests covering:
    - Training loop initialization
    - Loss calculation with unweighted CrossEntropyLoss
    - Backward pass execution
    - Optimizer creation (only trainable parameters included)
    - Scheduler creation (ReduceLROnPlateau)
    - EarlyStopping logic and trigger behavior
    - Frozen backbone verification (no gradients)
    - Classifier head gradient verification
    - Parameter update verification (classifier updates, backbone remains unchanged)
    - Training checkpoint saving and loading integrity
    - Training history logging format
    - Validation evaluation strictly uses no gradients
    - Test set is protected and not accessed during training
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import load_config
from src.models import build_model, get_efficientnet_b0
from src.training import (
    EarlyStopping,
    load_training_checkpoint,
    plot_training_curves,
    save_training_checkpoint,
    save_training_history,
    train_one_epoch,
    validate,
    verify_freezing_sanity,
)


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def config():
    """Load project configuration."""
    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))


@pytest.fixture
def dummy_dataset():
    """Create a small synthetic dataset for fast unit testing."""
    images = torch.randn(8, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0], dtype=torch.long)
    return TensorDataset(images, labels)


@pytest.fixture
def dummy_loader(dummy_dataset):
    """Create a DataLoader for synthetic dataset."""
    return DataLoader(dummy_dataset, batch_size=4, shuffle=False)


@pytest.fixture
def base_model():
    """Create a base EfficientNet-B0 model with frozen backbone."""
    return get_efficientnet_b0(pretrained=False, freeze_backbone_weights=True)


# ═════════════════════════════════════════════════════════════════════
# 1. OPTIMIZER & SCHEDULER TESTS
# ═════════════════════════════════════════════════════════════════════


class TestOptimizerAndScheduler:
    """Verify optimizer and scheduler setup according to Day 6 requirements."""

    def test_optimizer_contains_only_trainable_parameters(self, base_model):
        """Adam optimizer should only contain parameters with requires_grad=True."""
        trainable = list(filter(lambda p: p.requires_grad, base_model.parameters()))
        optimizer = torch.optim.Adam(trainable, lr=1e-3)

        total_opt_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        assert total_opt_params == 8967, f"Expected 8,967 optimizer params, got {total_opt_params}"

    def test_optimizer_lr_from_config(self, config, base_model):
        """Optimizer learning rate should match config value (1e-3)."""
        lr = config.get("training", {}).get("learning_rate", 1e-3)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=lr)
        assert optimizer.param_groups[0]["lr"] == 1e-3

    def test_reduce_lr_on_plateau_scheduler(self, base_model):
        """ReduceLROnPlateau reduces learning rate after patience epochs without improvement."""
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == 1e-3

        # Epoch 1: val_loss = 2.0
        scheduler.step(2.0)
        assert optimizer.param_groups[0]["lr"] == 1e-3

        # Epoch 2: val_loss = 2.1 (no improvement)
        scheduler.step(2.1)
        assert optimizer.param_groups[0]["lr"] == 1e-3

        # Epoch 3: val_loss = 2.1 (patience reached -> reduction)
        scheduler.step(2.1)
        assert optimizer.param_groups[0]["lr"] == 5e-4


# ═════════════════════════════════════════════════════════════════════
# 2. EARLY STOPPING TESTS
# ═════════════════════════════════════════════════════════════════════


class TestEarlyStopping:
    """Verify EarlyStopping behavior with patience and minimum delta."""

    def test_early_stopping_triggers_after_patience(self):
        """EarlyStopping triggers when metric does not improve for `patience` consecutive epochs."""
        es = EarlyStopping(patience=3, min_delta=0.001, mode="min")

        # Epoch 1: best loss = 1.5
        assert not es.step(1.5, epoch=1)
        assert es.counter == 0
        assert es.best_score == 1.5

        # Epoch 2: loss = 1.6 (no improvement -> count 1)
        assert not es.step(1.6, epoch=2)
        assert es.counter == 1

        # Epoch 3: loss = 1.7 (no improvement -> count 2)
        assert not es.step(1.7, epoch=3)
        assert es.counter == 2

        # Epoch 4: loss = 1.8 (no improvement -> count 3 -> triggers)
        assert es.step(1.8, epoch=4)
        assert es.counter == 3
        assert es.early_stop is True

    def test_early_stopping_resets_on_improvement(self):
        """Counter resets to 0 when validation loss improves beyond min_delta."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        es.step(2.0, epoch=1)
        es.step(2.05, epoch=2)  # counter = 1
        assert es.counter == 1

        es.step(1.90, epoch=3)  # improved!
        assert es.counter == 0
        assert es.best_score == 1.90
        assert not es.early_stop


# ═════════════════════════════════════════════════════════════════════
# 3. FREEZING & GRADIENT BEHAVIOR TESTS
# ═════════════════════════════════════════════════════════════════════


class TestFreezingAndGradients:
    """Verify backbone freezing, gradient isolation, and parameter updates."""

    def test_verify_freezing_sanity_utility(self, base_model, dummy_dataset):
        """Sanity check utility confirms gradients and updates on classifier only."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-3)
        device = torch.device("cpu")

        sample_batch = (dummy_dataset[:4][0], dummy_dataset[:4][1])
        res = verify_freezing_sanity(base_model, sample_batch, criterion, optimizer, device)

        assert res["success"] is True
        assert res["backbone_has_no_grad"] is True
        assert res["classifier_grads_exist"] is True
        assert res["classifier_changed"] is True
        assert res["backbone_unchanged"] is True

    def test_backbone_parameters_unchanged_after_train_epoch(self, base_model, dummy_loader):
        """After running train_one_epoch, backbone weights must remain strictly identical."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-3)
        device = torch.device("cpu")

        backbone_before = [p.clone().detach() for p in base_model.features.parameters()]

        train_loss, train_acc = train_one_epoch(base_model, dummy_loader, criterion, optimizer, device)

        backbone_after = list(base_model.features.parameters())
        assert all(torch.equal(b, a) for b, a in zip(backbone_before, backbone_after)), (
            "Backbone weights were modified during train_one_epoch!"
        )


# ═════════════════════════════════════════════════════════════════════
# 4. TRAINING & VALIDATION FUNCTION TESTS
# ═════════════════════════════════════════════════════════════════════


class TestTrainingAndValidationFunctions:
    """Verify train_one_epoch and validate function outputs."""

    def test_train_one_epoch_returns_loss_and_acc(self, base_model, dummy_loader):
        """train_one_epoch returns valid scalar loss and accuracy in [0, 1]."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-3)
        device = torch.device("cpu")

        loss, acc = train_one_epoch(base_model, dummy_loader, criterion, optimizer, device)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0.0
        assert 0.0 <= acc <= 1.0

    def test_validate_has_no_grad(self, base_model, dummy_loader):
        """validate runs under torch.no_grad and leaves model parameters unaffected."""
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cpu")

        val_loss, val_acc = validate(base_model, dummy_loader, criterion, device)
        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss > 0.0
        assert 0.0 <= val_acc <= 1.0

        # Verify no gradients left on any parameter
        assert all(p.grad is None for p in base_model.parameters())


# ═════════════════════════════════════════════════════════════════════
# 5. CHECKPOINT & ARTIFACT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestCheckpointAndArtifacts:
    """Verify checkpoint save/load and training artifact generation."""

    def test_save_and_load_training_checkpoint(self, base_model):
        """Training checkpoint saves and reloads all states cleanly."""
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min")

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_best.pth"
            save_training_checkpoint(
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=5,
                val_loss=1.234,
                val_acc=0.678,
                checkpoint_path=ckpt_path,
                config_dict={"test": 123},
            )
            assert ckpt_path.exists()

            # Reload
            new_model = get_efficientnet_b0(pretrained=False, freeze_backbone_weights=True)
            meta = load_training_checkpoint(ckpt_path, new_model)
            assert meta["epoch"] == 5
            assert meta["val_loss"] == 1.234
            assert meta["val_accuracy"] == 0.678
            assert meta["checkpoint_type"] == "trained_best_model"

    def test_plot_training_curves_and_history_csv(self):
        """Training curves and history CSV are properly generated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = {
                "epoch": [1, 2, 3],
                "train_loss": [1.8, 1.4, 1.2],
                "train_accuracy": [0.4, 0.55, 0.65],
                "val_loss": [1.6, 1.3, 1.1],
                "val_accuracy": [0.45, 0.58, 0.68],
                "learning_rate": [0.001, 0.001, 0.001],
                "epoch_time_seconds": [10.0, 10.0, 10.0],
            }

            # Save CSV
            csv_path = Path(tmp_dir) / "history.csv"
            save_training_history(history, csv_path)
            assert csv_path.exists()
            df = pd.read_csv(csv_path)
            assert len(df) == 3
            assert list(df.columns) == [
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "learning_rate",
                "epoch_time_seconds",
            ]

            # Save Plots
            loss_fig, acc_fig = plot_training_curves(df, tmp_dir)
            assert loss_fig.exists()
            assert acc_fig.exists()
            assert loss_fig.name == "day6_loss_curve.png"
            assert acc_fig.name == "day6_accuracy_curve.png"


# ═════════════════════════════════════════════════════════════════════
# 6. TEST SET PROTECTION VERIFICATION
# ═════════════════════════════════════════════════════════════════════


class TestSetProtection:
    """Verify test set is NEVER used by the training pipeline."""

    def test_train_model_signature_has_no_test_loader(self):
        """Inspect train_one_epoch and validate signatures; ensure no test data."""
        import inspect
        sig_train = inspect.signature(train_one_epoch)
        sig_val = inspect.signature(validate)

        # Neither takes test_loader or test_data
        assert "test_loader" not in sig_train.parameters
        assert "test_loader" not in sig_val.parameters
