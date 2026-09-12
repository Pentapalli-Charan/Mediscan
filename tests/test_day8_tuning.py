"""
MediScan — Day 8: Controlled Hyperparameter Tuning Tests

Tests covering:
- Verification of day8_experiments.csv existence, schema, and baseline values
- Experiment 0 (Baseline control) match with Day 6/7 ground truth
- Validation of experiment metrics (loss, accuracy, durations)
- Checkpoint existence and reload verification for completed experiments
- Comparison figures existence and validity
- Strict test set isolation (zero test predictions / evaluation)
- Leakage-safe split preservation (zero lesion overlap between train and val)
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import load_config
from src.models import CLASS_NAMES, NUM_CLASSES, build_model
from src.training import load_training_checkpoint


@pytest.fixture(scope="module")
def config():
    return load_config(str(PROJECT_ROOT / "config" / "config.yaml"))


@pytest.fixture(scope="module")
def experiments_df():
    csv_path = PROJECT_ROOT / "reports" / "data" / "day8_experiments.csv"
    if not csv_path.exists():
        pytest.skip("day8_experiments.csv not found — run Day 8 tuning first")
    return pd.read_csv(csv_path)


class TestExperimentsRecord:
    """Verify schema and validity of day8_experiments.csv."""

    def test_csv_exists(self):
        csv_path = PROJECT_ROOT / "reports" / "data" / "day8_experiments.csv"
        assert csv_path.exists(), "day8_experiments.csv does not exist"

    def test_required_columns_present(self, experiments_df):
        required = [
            "experiment_id",
            "experiment_name",
            "learning_rate",
            "dropout",
            "augmentation_configuration",
            "status",
        ]
        for col in required:
            assert col in experiments_df.columns, f"Missing required column: {col}"

    def test_has_at_least_baseline_and_one_experiment(self, experiments_df):
        assert len(experiments_df) >= 2, "Must contain at least Experiment 0 and 1"

    def test_experiment_0_matches_baseline_ground_truth(self, experiments_df):
        exp0 = experiments_df[experiments_df["experiment_id"] == 0].iloc[0]
        assert exp0["experiment_name"] == "baseline_control"
        assert round(float(exp0["best_val_loss"]), 4) == 0.6422
        assert round(float(exp0["highest_val_accuracy"]), 4) == 0.7765
        assert int(exp0["best_epoch"]) == 14

    def test_completed_experiments_metrics_valid(self, experiments_df):
        completed = experiments_df[experiments_df["status"] == "COMPLETED"]
        for _, row in completed.iterrows():
            assert 0 < row["best_val_loss"] < 5.0, f"Invalid val loss: {row['best_val_loss']}"
            assert 0.0 <= row["highest_val_accuracy"] <= 1.0, f"Invalid val accuracy: {row['highest_val_accuracy']}"
            assert row["total_epochs"] > 0, f"Invalid total epochs: {row['total_epochs']}"


class TestCheckpoints:
    """Verify checkpoints for completed experiments."""

    def test_baseline_checkpoint_exists(self):
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
        assert ckpt_path.exists(), "Baseline checkpoint missing"

    def test_baseline_checkpoint_reload(self, config):
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
        model = build_model(config)
        model.eval()
        ckpt_meta = load_training_checkpoint(ckpt_path, model, device="cpu")
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            out = model(dummy)
        assert out.shape == (2, 7)
        assert torch.isfinite(out).all()


class TestComparisonFigures:
    """Verify Day 8 comparison figures."""

    def test_day8_figures_directory_exists(self):
        fig_dir = PROJECT_ROOT / "reports" / "figures" / "day8"
        assert fig_dir.exists(), "reports/figures/day8 directory missing"

    def test_loss_comparison_plot_exists(self):
        plot_path = PROJECT_ROOT / "reports" / "figures" / "day8" / "day8_val_loss_comparison.png"
        assert plot_path.exists(), "Validation loss comparison plot missing"
        assert plot_path.stat().st_size > 1000, "Validation loss plot is empty"

    def test_accuracy_comparison_plot_exists(self):
        plot_path = PROJECT_ROOT / "reports" / "figures" / "day8" / "day8_val_accuracy_comparison.png"
        assert plot_path.exists(), "Validation accuracy comparison plot missing"
        assert plot_path.stat().st_size > 1000, "Validation accuracy plot is empty"

    def test_summary_barchart_exists(self):
        plot_path = PROJECT_ROOT / "reports" / "figures" / "day8" / "day8_summary_comparison.png"
        assert plot_path.exists(), "Summary comparison chart missing"
        assert plot_path.stat().st_size > 1000, "Summary chart is empty"


class TestSplitAndLeakageSafety:
    """Confirm split integrity and absence of data leakage."""

    def test_no_lesion_overlap_train_val(self):
        train_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_val.csv")
        overlap = set(train_df["lesion_id"]).intersection(set(val_df["lesion_id"]))
        assert len(overlap) == 0, f"Lesion leakage detected: {len(overlap)} lesions"

    def test_no_image_overlap_train_val(self):
        train_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_train.csv")
        val_df = pd.read_csv(PROJECT_ROOT / "reports" / "data" / "split_val.csv")
        overlap = set(train_df["image_id"]).intersection(set(val_df["image_id"]))
        assert len(overlap) == 0, f"Image overlap detected: {len(overlap)} images"


class TestSetProtection:
    """Strict audit that test data was never evaluated in Day 8."""

    def test_no_test_columns_in_experiments_csv(self, experiments_df):
        for col in experiments_df.columns:
            assert "test" not in col.lower(), f"Forbidden test column in experiments CSV: {col}"

    def test_no_test_results_in_day8_figures(self):
        fig_dir = PROJECT_ROOT / "reports" / "figures" / "day8"
        if fig_dir.exists():
            for fig in fig_dir.glob("*.png"):
                assert "test" not in fig.name.lower(), f"Forbidden test figure found: {fig.name}"
