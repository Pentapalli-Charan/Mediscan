"""
MediScan — Day 7: Training Analysis Tests

Tests covering:
    - Training history CSV existence and format
    - Epoch-by-epoch metric validity
    - Best checkpoint existence and integrity
    - Checkpoint reload produces correct output shape [B, 7]
    - No NaN/Inf in loaded model outputs
    - Class mapping unchanged (7 classes)
    - Split CSVs unchanged (MD5 integrity)
    - Test set protection (zero test artifacts)
    - Training curves generation
    - Scheduler/early stopping configuration
    - Reproducibility (seed configuration)
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import load_config
from src.models import CLASS_NAMES, NUM_CLASSES, build_model, get_efficientnet_b0
from src.training import load_training_checkpoint


# ═════════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def config():
    """Load project configuration."""
    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))


@pytest.fixture(scope="module")
def day7_history():
    """Load Day 7 training history CSV."""
    csv_path = PROJECT_ROOT / "reports" / "data" / "day7_training_history.csv"
    if not csv_path.exists():
        pytest.skip("Day 7 training history CSV not found — run training first")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def day6_history():
    """Load Day 6 training history CSV."""
    csv_path = PROJECT_ROOT / "reports" / "data" / "day6_training_history.csv"
    if not csv_path.exists():
        pytest.skip("Day 6 training history CSV not found — run training first")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def checkpoint_path():
    """Path to best checkpoint."""
    path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    if not path.exists():
        pytest.skip("Best checkpoint not found — run training first")
    return path


# ═════════════════════════════════════════════════════════════════════
# 1. TRAINING HISTORY TESTS
# ═════════════════════════════════════════════════════════════════════


class TestTrainingHistory:
    """Verify training history CSV exists and contains valid data."""

    def test_day7_history_csv_exists(self):
        """Day 7 training history CSV must exist."""
        csv_path = PROJECT_ROOT / "reports" / "data" / "day7_training_history.csv"
        assert csv_path.exists(), f"Day 7 training history CSV not found at {csv_path}"

    def test_day6_history_csv_exists(self):
        """Day 6 training history CSV must exist."""
        csv_path = PROJECT_ROOT / "reports" / "data" / "day6_training_history.csv"
        assert csv_path.exists(), f"Day 6 training history CSV not found at {csv_path}"

    def test_required_columns_present(self, day7_history):
        """History CSV must contain all required columns."""
        required = [
            "epoch", "train_loss", "train_accuracy",
            "val_loss", "val_accuracy", "learning_rate", "epoch_time_seconds",
        ]
        for col in required:
            assert col in day7_history.columns, f"Missing column: {col}"

    def test_history_not_empty(self, day7_history):
        """History must contain at least 1 epoch."""
        assert len(day7_history) >= 1, "Training history is empty"

    def test_epochs_monotonically_increasing(self, day7_history):
        """Epoch numbers must be monotonically increasing."""
        epochs = day7_history["epoch"].values
        assert all(epochs[i] < epochs[i + 1] for i in range(len(epochs) - 1)), \
            "Epochs are not monotonically increasing"

    def test_train_loss_positive(self, day7_history):
        """All training losses must be positive."""
        assert (day7_history["train_loss"] > 0).all(), "Some train_loss values are non-positive"

    def test_val_loss_positive(self, day7_history):
        """All validation losses must be positive."""
        assert (day7_history["val_loss"] > 0).all(), "Some val_loss values are non-positive"

    def test_train_accuracy_in_range(self, day7_history):
        """Training accuracy must be in [0, 1]."""
        assert (day7_history["train_accuracy"] >= 0).all(), "train_accuracy below 0"
        assert (day7_history["train_accuracy"] <= 1).all(), "train_accuracy above 1"

    def test_val_accuracy_in_range(self, day7_history):
        """Validation accuracy must be in [0, 1]."""
        assert (day7_history["val_accuracy"] >= 0).all(), "val_accuracy below 0"
        assert (day7_history["val_accuracy"] <= 1).all(), "val_accuracy above 1"

    def test_learning_rate_positive(self, day7_history):
        """All learning rates must be positive."""
        assert (day7_history["learning_rate"] > 0).all(), "Non-positive learning rate"

    def test_epoch_durations_positive(self, day7_history):
        """All epoch durations must be positive."""
        assert (day7_history["epoch_time_seconds"] > 0).all(), "Non-positive epoch duration"

    def test_no_nan_in_metrics(self, day7_history):
        """No NaN values in any metric column."""
        numeric_cols = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"]
        for col in numeric_cols:
            assert not day7_history[col].isnull().any(), f"NaN found in {col}"

    def test_day6_and_day7_history_match(self, day6_history, day7_history):
        """Day 6 and Day 7 histories must contain identical data."""
        pd.testing.assert_frame_equal(
            day6_history, day7_history,
            check_exact=True,
            obj="Day 6 vs Day 7 training history",
        )


# ═════════════════════════════════════════════════════════════════════
# 2. CHECKPOINT TESTS
# ═════════════════════════════════════════════════════════════════════


class TestCheckpoint:
    """Verify best checkpoint integrity and reloadability."""

    def test_checkpoint_exists(self, checkpoint_path):
        """Best checkpoint file must exist."""
        assert checkpoint_path.exists()

    def test_checkpoint_contains_required_keys(self, checkpoint_path):
        """Checkpoint must contain all required metadata."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        required_keys = [
            "epoch", "model_state_dict", "optimizer_state_dict",
            "val_loss", "val_accuracy", "checkpoint_type",
            "architecture", "num_classes",
        ]
        for key in required_keys:
            assert key in ckpt, f"Missing checkpoint key: {key}"

    def test_checkpoint_type_is_trained(self, checkpoint_path):
        """Checkpoint must be labeled as trained_best_model."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        assert ckpt["checkpoint_type"] == "trained_best_model"

    def test_checkpoint_architecture_correct(self, checkpoint_path):
        """Checkpoint architecture must be efficientnet_b0."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        assert ckpt["architecture"] == "efficientnet_b0"

    def test_checkpoint_num_classes_correct(self, checkpoint_path):
        """Checkpoint must have 7 classes."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        assert ckpt["num_classes"] == 7

    def test_checkpoint_epoch_matches_best(self, checkpoint_path, day7_history):
        """Checkpoint epoch must match best val_loss epoch in history."""
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        best_epoch = int(day7_history.loc[day7_history["val_loss"].idxmin(), "epoch"])
        assert ckpt["epoch"] == best_epoch, \
            f"Checkpoint epoch ({ckpt['epoch']}) != best val_loss epoch ({best_epoch})"


# ═════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT RELOAD TESTS
# ═════════════════════════════════════════════════════════════════════


class TestCheckpointReload:
    """Verify that a fresh model can load the checkpoint and produce valid outputs."""

    def test_reload_produces_correct_shape(self, config, checkpoint_path):
        """Fresh model with loaded checkpoint produces [B, 7] output."""
        model = build_model(config)
        model.eval()
        load_training_checkpoint(checkpoint_path, model, device="cpu")

        with torch.no_grad():
            logits = model(torch.randn(4, 3, 224, 224))
        assert logits.shape == (4, 7), f"Expected (4, 7), got {logits.shape}"

    def test_reload_logits_finite(self, config, checkpoint_path):
        """Loaded model must produce finite logits (no NaN/Inf)."""
        model = build_model(config)
        model.eval()
        load_training_checkpoint(checkpoint_path, model, device="cpu")

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))
        assert torch.isfinite(logits).all(), "Loaded model produced non-finite logits"

    def test_reload_no_nan(self, config, checkpoint_path):
        """Loaded model must not produce NaN values."""
        model = build_model(config)
        model.eval()
        load_training_checkpoint(checkpoint_path, model, device="cpu")

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))
        assert not torch.isnan(logits).any(), "NaN detected in reloaded model output"

    def test_reload_no_inf(self, config, checkpoint_path):
        """Loaded model must not produce Inf values."""
        model = build_model(config)
        model.eval()
        load_training_checkpoint(checkpoint_path, model, device="cpu")

        with torch.no_grad():
            logits = model(torch.randn(2, 3, 224, 224))
        assert not torch.isinf(logits).any(), "Inf detected in reloaded model output"

    def test_class_mapping_unchanged(self):
        """Class mapping must remain [akiec, bcc, bkl, df, mel, nv, vasc]."""
        expected = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        assert CLASS_NAMES == expected
        assert NUM_CLASSES == 7


# ═════════════════════════════════════════════════════════════════════
# 4. SPLIT CSV INTEGRITY TESTS
# ═════════════════════════════════════════════════════════════════════


class TestSplitIntegrity:
    """Verify split CSVs remain unchanged."""

    def test_train_split_exists(self):
        """Training split CSV must exist."""
        assert (PROJECT_ROOT / "reports" / "data" / "split_train.csv").exists()

    def test_val_split_exists(self):
        """Validation split CSV must exist."""
        assert (PROJECT_ROOT / "reports" / "data" / "split_val.csv").exists()

    def test_test_split_exists(self):
        """Test split CSV must exist."""
        assert (PROJECT_ROOT / "reports" / "data" / "split_test.csv").exists()

    def test_split_sizes_correct(self):
        """Split sizes must match expected counts."""
        train_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_val.csv")
        test_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_test.csv")

        assert len(train_df) == 6982, f"Expected 6982 train samples, got {len(train_df)}"
        assert len(val_df) == 1521, f"Expected 1521 val samples, got {len(val_df)}"
        assert len(test_df) == 1512, f"Expected 1512 test samples, got {len(test_df)}"

    def test_total_sample_count(self):
        """Total sample count must equal 10015."""
        train_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_val.csv")
        test_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_test.csv")
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 10015, f"Expected 10015 total, got {total}"

    def test_no_overlap_between_splits(self):
        """No image_id must appear in multiple splits."""
        train_ids = set(pd.read_csv(
            PROJECT_ROOT / "reports" / "data" / "split_train.csv"
        )["image_id"])
        val_ids = set(pd.read_csv(
            PROJECT_ROOT / "reports" / "data" / "split_val.csv"
        )["image_id"])
        test_ids = set(pd.read_csv(
            PROJECT_ROOT / "reports" / "data" / "split_test.csv"
        )["image_id"])

        assert len(train_ids & val_ids) == 0, "Train-val overlap detected"
        assert len(train_ids & test_ids) == 0, "Train-test overlap detected"
        assert len(val_ids & test_ids) == 0, "Val-test overlap detected"


# ═════════════════════════════════════════════════════════════════════
# 5. TEST SET PROTECTION TESTS
# ═════════════════════════════════════════════════════════════════════


class TestSetProtection:
    """Verify that NO test set evaluation has been performed."""

    def test_no_test_columns_in_history(self, day7_history):
        """Training history must not contain any test-related columns."""
        test_cols = [c for c in day7_history.columns if "test" in c.lower()]
        assert len(test_cols) == 0, f"Test columns found in history: {test_cols}"

    def test_no_test_results_files(self):
        """No test result files should exist for Day 7."""
        reports_dir = PROJECT_ROOT / "reports"
        test_files = [
            p for p in reports_dir.rglob("*")
            if p.is_file() and "test_result" in p.name.lower() and "day7" in p.name.lower()
        ]
        assert len(test_files) == 0, f"Test result files found: {test_files}"

    def test_training_code_no_test_evaluation(self):
        """Training pipeline must not contain test set evaluation calls."""
        trainer_path = PROJECT_ROOT / "src" / "training" / "trainer.py"
        content = trainer_path.read_text(encoding="utf-8")

        # The train_model function should not reference test_loader or test evaluation
        # (it's OK if comments mention test set protection)
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            if "test_loader" in stripped and "=" in stripped and "loaders" in stripped:
                pytest.fail(
                    f"Line {i} in trainer.py uses test_loader in assignment: {stripped}"
                )


# ═════════════════════════════════════════════════════════════════════
# 6. TRAINING CURVES TESTS
# ═════════════════════════════════════════════════════════════════════


class TestTrainingCurves:
    """Verify training curve figures were generated."""

    def test_day7_loss_curve_exists(self):
        """Day 7 loss curve must exist."""
        path = PROJECT_ROOT / "reports" / "figures" / "day7_loss_curve.png"
        assert path.exists(), f"Day 7 loss curve not found at {path}"

    def test_day7_accuracy_curve_exists(self):
        """Day 7 accuracy curve must exist."""
        path = PROJECT_ROOT / "reports" / "figures" / "day7_accuracy_curve.png"
        assert path.exists(), f"Day 7 accuracy curve not found at {path}"

    def test_day7_lr_schedule_exists(self):
        """Day 7 LR schedule plot must exist."""
        path = PROJECT_ROOT / "reports" / "figures" / "day7_lr_schedule.png"
        assert path.exists(), f"Day 7 LR schedule not found at {path}"

    def test_curve_files_not_empty(self):
        """All curve files must have non-zero size."""
        figures_dir = PROJECT_ROOT / "reports" / "figures"
        for name in ["day7_loss_curve.png", "day7_accuracy_curve.png", "day7_lr_schedule.png"]:
            path = figures_dir / name
            if path.exists():
                assert path.stat().st_size > 0, f"{name} is empty (0 bytes)"


# ═════════════════════════════════════════════════════════════════════
# 7. CONFIGURATION TESTS
# ═════════════════════════════════════════════════════════════════════


class TestConfiguration:
    """Verify training configuration is correct and consistent."""

    def test_seed_is_42(self, config):
        """Seed must be 42 for reproducibility."""
        assert config.get("seed") == 42

    def test_model_architecture(self, config):
        """Model architecture must be efficientnet_b0."""
        assert config.get("model", {}).get("architecture") == "efficientnet_b0"

    def test_num_classes_is_7(self, config):
        """Number of classes must be 7."""
        assert config.get("model", {}).get("num_classes") == 7
        assert config.get("classes", {}).get("num_classes") == 7

    def test_backbone_frozen(self, config):
        """Backbone must be frozen for Day 6/7 baseline."""
        assert config.get("model", {}).get("freeze_backbone") is True

    def test_loss_is_unweighted(self, config):
        """Day 6/7 baseline must use unweighted CrossEntropyLoss."""
        # Verify the training code uses nn.CrossEntropyLoss() without weights
        trainer_path = PROJECT_ROOT / "src" / "training" / "trainer.py"
        content = trainer_path.read_text(encoding="utf-8")
        assert "nn.CrossEntropyLoss()" in content, \
            "Expected unweighted CrossEntropyLoss() in trainer.py"

    def test_optimizer_is_adam(self, config):
        """Optimizer must be Adam."""
        assert config.get("training", {}).get("optimizer") == "adam"

    def test_split_by_lesion_id(self, config):
        """Split must be by lesion_id to prevent leakage."""
        assert config.get("split", {}).get("group_column") == "lesion_id"
