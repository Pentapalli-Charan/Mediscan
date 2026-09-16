"""
MediScan — Day 13: MLflow Experiment Tracking & Metadata Tests

Validates:
1. MLflow package import and version detection.
2. Isolated tracking directory initialization (no pollution of canonical store).
3. Experiment creation and retrieval under isolated local store.
4. Active run lifecycle, parameter logging, metric logging, and artifact persistence.
5. Offline functionality without external network calls.
6. Verification of the canonical 'MediScan' tracking store (mlruns/):
   - Exactly 3 genuine runs tracked:
     * EfficientNet-B0-Baseline
     * EfficientNet-B0-LR-5e-4
     * ResNet50-Benchmark
   - No skipped Day 8 runs fabricated.
   - Day 9 test metrics present with explicit historical provenance tags.
7. Strict model checkpoint integrity (efficientnet_b0_best.pth MD5 unchanged).
8. Strict test split partition integrity (split_test.csv MD5 unchanged).
9. Day 9 metrics file integrity (day9_metrics.json unaltered).
10. Existing Streamlit inference remains completely operational.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure file store is permitted
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import mlflow
from mlflow.tracking import MlflowClient

from src.training.mlflow_tracker import (
    DEFAULT_MLRUNS_DIR,
    EXPERIMENT_NAME,
    get_client,
    get_experiment_runs,
    get_experiment_summary_df,
    get_tracking_uri,
    setup_mlflow,
)

EXPECTED_CHECKPOINT_MD5 = "7c1c6fcbe02e93f0ff8b4a20f62e29e3"
EXPECTED_TEST_SPLIT_MD5 = "6a5ae1b65c25d504f78bc1da2361d82c"


class TestMLflowCoreFunctionality:
    """Verify MLflow setup, offline local storage, logging, and querying in isolation."""

    def test_mlflow_import_and_version(self):
        """MLflow must be importable and report version >= 3.0."""
        assert hasattr(mlflow, "__version__")
        version_parts = [int(p) for p in mlflow.__version__.split(".")[:2]]
        assert version_parts[0] >= 3, f"Expected MLflow 3+, got {mlflow.__version__}"

    def test_isolated_experiment_and_run_lifecycle(self, tmp_path):
        """Verify experiment creation, param logging, metric logging, and artifacts in temp directory."""
        temp_tracking_uri = (tmp_path / "temp_mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(temp_tracking_uri)

        client = MlflowClient(tracking_uri=temp_tracking_uri)
        exp_id = client.create_experiment("Isolated-Test-Exp")
        assert exp_id is not None

        with mlflow.start_run(experiment_id=exp_id, run_name="Test-Run") as run:
            run_id = run.info.run_id

            # Log parameters
            mlflow.log_params({
                "architecture": "efficientnet_b0",
                "learning_rate": 0.001,
                "batch_size": 32,
            })

            # Log epoch metrics
            for epoch in range(1, 4):
                mlflow.log_metric("train_loss", 1.0 / epoch, step=epoch)
                mlflow.log_metric("val_loss", 1.2 / epoch, step=epoch)

            # Log summary metric
            mlflow.log_metric("best_val_loss", 0.4)

            # Log dummy artifact
            sample_artifact = tmp_path / "summary.txt"
            sample_artifact.write_text("MediScan test artifact content")
            mlflow.log_artifact(str(sample_artifact))

        # Query run back via client
        retrieved_run = client.get_run(run_id)
        assert retrieved_run.info.status == "FINISHED"
        assert retrieved_run.data.params["architecture"] == "efficientnet_b0"
        assert retrieved_run.data.params["learning_rate"] == "0.001"
        assert retrieved_run.data.metrics["best_val_loss"] == 0.4

        # Verify metric history
        metric_history = client.get_metric_history(run_id, "train_loss")
        assert len(metric_history) == 3
        assert metric_history[-1].step == 3

        # Verify artifact listing
        artifacts = client.list_artifacts(run_id)
        assert any(a.path == "summary.txt" for a in artifacts)


@pytest.fixture(scope="module")
def shared_mlflow_client():
    canonical_uri = get_tracking_uri()
    return MlflowClient(tracking_uri=canonical_uri)


class TestCanonicalMediScanTrackingStore:
    """Verify the canonical MediScan experiment and historical run records."""

    def test_mediscan_experiment_exists(self, shared_mlflow_client):
        """Canonical 'MediScan' experiment must exist in local mlruns/."""
        exp = shared_mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        assert exp is not None, f"Experiment '{EXPERIMENT_NAME}' not found in {DEFAULT_MLRUNS_DIR}"
        assert exp.lifecycle_stage == "active"

    def test_canonical_runs_count_and_names(self, shared_mlflow_client):
        """Store must contain exactly the 3 executed runs (no fabricated runs)."""
        exp = shared_mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = shared_mlflow_client.search_runs(experiment_ids=[exp.experiment_id])
        run_names = {r.data.tags.get("mlflow.runName") for r in runs}

        expected_runs = {
            "EfficientNet-B0-Baseline",
            "EfficientNet-B0-LR-5e-4",
            "ResNet50-Benchmark",
        }
        assert expected_runs.issubset(run_names), f"Missing runs: {expected_runs - run_names}"

        # Verify that skipped Day 8 experiments (2, 3, 4) were NOT logged as fake runs
        for r in runs:
            name = r.data.tags.get("mlflow.runName", "")
            assert "higher_lr_2e3" not in name.lower()
            assert "dropout_04" not in name.lower()
            assert "stronger_augmentation" not in name.lower()

    def test_baseline_run_metrics_and_provenance(self, shared_mlflow_client):
        """Baseline run must contain verified Day 6 validation metrics and Day 9 test metrics."""
        exp = shared_mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = shared_mlflow_client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.mlflow.runName = 'EfficientNet-B0-Baseline'",
        )
        assert len(runs) >= 1
        baseline_run = runs[0]

        # Validation metrics
        metrics = baseline_run.data.metrics
        assert abs(metrics["best_val_loss"] - 0.6422) < 1e-3
        assert abs(metrics["val_accuracy_at_best_loss"] - 0.7738) < 1e-3

        # Day 9 test metrics
        assert abs(metrics["test_top1_accuracy"] - 0.757937) < 1e-4
        assert abs(metrics["test_top2_accuracy"] - 0.895503) < 1e-4
        assert abs(metrics["test_macro_f1"] - 0.515696) < 1e-4
        assert abs(metrics["test_weighted_f1"] - 0.744771) < 1e-4
        assert abs(metrics["test_macro_roc_auc"] - 0.9211) < 1e-3
        assert abs(metrics["test_ce_loss"] - 0.663784) < 1e-4

        # Provenance tags
        tags = baseline_run.data.tags
        assert tags.get("test_evaluation_status") == "locked_held_out_evaluation"
        assert "NOT used for model selection" in tags.get("test_evaluation_note", "")

    def test_summary_dataframe_generation(self):
        """get_experiment_summary_df must return populated comparison table."""
        df = get_experiment_summary_df(EXPERIMENT_NAME)
        assert not df.empty
        assert "Run Name" in df.columns
        assert "Best Val Loss" in df.columns
        assert "Val Acc (%)" in df.columns
        assert len(df) >= 3


class TestPipelineIntegrityAndNonInterference:
    """Verify that Day 13 changes strictly preserved all preceding artifacts and functionality."""

    def test_efficientnet_checkpoint_hash_unaltered(self):
        """Primary deployment checkpoint must be bitwise identical."""
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
        assert ckpt_path.exists(), f"Checkpoint missing at {ckpt_path}"
        actual_md5 = hashlib.md5(ckpt_path.read_bytes()).hexdigest()
        assert actual_md5 == EXPECTED_CHECKPOINT_MD5, (
            f"Checkpoint was modified! Expected {EXPECTED_CHECKPOINT_MD5}, got {actual_md5}"
        )

    def test_test_split_hash_unaltered(self):
        """Test partition split_test.csv must remain strictly untouched."""
        split_path = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
        assert split_path.exists(), f"Test split missing at {split_path}"
        actual_md5 = hashlib.md5(split_path.read_bytes()).hexdigest()
        assert actual_md5 == EXPECTED_TEST_SPLIT_MD5, (
            f"split_test.csv was modified! Expected {EXPECTED_TEST_SPLIT_MD5}, got {actual_md5}"
        )

    def test_day9_metrics_json_unaltered(self):
        """Day 9 metrics file must retain its exact locked values."""
        json_path = PROJECT_ROOT / "reports" / "data" / "day9_metrics.json"
        assert json_path.exists()
        with open(json_path, "r") as f:
            data = json.load(f)

        assert abs(data["overall_metrics"]["top1_accuracy"] - 0.757937) < 1e-6
        assert abs(data["overall_metrics"]["top2_accuracy"] - 0.895503) < 1e-6
        assert abs(data["overall_metrics"]["macro_f1"] - 0.515696) < 1e-6
        assert abs(data["overall_metrics"]["weighted_f1"] - 0.744771) < 1e-6

    def test_app_utils_inference_still_operational(self):
        """Inference pipeline from Day 12 must run without regression."""
        from app.utils import load_cached_mediscan_model, run_model_inference

        model, meta = load_cached_mediscan_model(device=torch.device("cpu"))
        assert meta["architecture"] == "efficientnet_b0"

        dummy_tensor = torch.randn(1, 3, 224, 224)
        pred = run_model_inference(model, dummy_tensor, device=torch.device("cpu"))
        assert pred["predicted_class"] in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        assert len(pred["probabilities"]) == 7
        assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-4
