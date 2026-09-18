"""
MediScan — MLflow Experiment Tracking & Metadata Management (Day 13)

Integrates MLflow for reproducible experiment tracking across training runs,
hyperparameter evaluations, and model benchmarking:
- Configures local file-based tracking store (mlruns/)
- Organizes runs under the canonical "MediScan" experiment
- Logs structured hyperparameters, epoch-level metrics, and model artifacts
- Populates and synchronizes historical experiment runs faithfully without retraining
- Supports offline querying of experiment comparisons
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Silence prototype agent hint in MLflow 3.16+
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import numpy as np
import pandas as pd
import torch
import yaml

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# Canonical project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MLRUNS_DIR = PROJECT_ROOT / "mlruns"
EXPERIMENT_NAME = "MediScan"


def get_tracking_uri(custom_uri: Optional[Union[str, Path]] = None) -> str:
    """
    Return the normalized local tracking URI for MLflow.
    Defaults to file:///<PROJECT_ROOT>/mlruns.

    Args:
        custom_uri: Optional path or URI override.

    Returns:
        str: Absolute file URI or tracking URI.
    """
    if custom_uri is not None:
        p = Path(custom_uri)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve().as_uri()

    return DEFAULT_MLRUNS_DIR.resolve().as_uri()


def setup_mlflow(
    tracking_uri: Optional[Union[str, Path]] = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> Tuple[str, str]:
    """
    Configure MLflow tracking store and initialize/retrieve the target experiment.

    Args:
        tracking_uri: Optional tracking URI or folder. Defaults to mlruns/.
        experiment_name: Name of the experiment (default: "MediScan").

    Returns:
        tuple: (resolved_tracking_uri, experiment_id)
    """
    uri = get_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(uri)

    client = MlflowClient(tracking_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = client.create_experiment(
            name=experiment_name,
            tags={
                "project": "MediScan",
                "dataset": "HAM10000",
                "task": "Dermoscopic Skin Lesion Classification",
                "framework": "PyTorch",
            },
        )
        logger.info(f"Created MLflow experiment '{experiment_name}' with ID {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' (ID: {experiment_id})")

    mlflow.set_experiment(experiment_name)
    return uri, str(experiment_id)


def get_client(tracking_uri: Optional[Union[str, Path]] = None) -> MlflowClient:
    """Return an instantiated MlflowClient pointed at the tracking URI."""
    uri = get_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=uri)


def get_experiment_runs(
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: Optional[Union[str, Path]] = None,
) -> List[Run]:
    """
    Retrieve all non-deleted runs for an experiment.

    Args:
        experiment_name: Name of the experiment.
        tracking_uri: Tracking URI.

    Returns:
        List of mlflow.entities.Run objects.
    """
    client = get_client(tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []
    return client.search_runs(experiment_ids=[exp.experiment_id])


def get_experiment_summary_df(
    experiment_name: str = EXPERIMENT_NAME,
    tracking_uri: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Extract a clean tabular comparison DataFrame of runs in the experiment.

    Args:
        experiment_name: Name of the experiment.
        tracking_uri: Tracking URI.

    Returns:
        pd.DataFrame: Table of runs with key parameters and metrics.
    """
    runs = get_experiment_runs(experiment_name, tracking_uri)
    if not runs:
        return pd.DataFrame()

    records = []
    for r in runs:
        row = {
            "Run ID": r.info.run_id[:8],
            "Run Name": r.data.tags.get("mlflow.runName", "Unnamed"),
            "Status": r.info.status,
            "Architecture": r.data.params.get("architecture", "N/A"),
            "Learning Rate": r.data.params.get("learning_rate", "N/A"),
            "Dropout": r.data.params.get("dropout", "N/A"),
            "Batch Size": r.data.params.get("batch_size", "N/A"),
            "Best Epoch": r.data.metrics.get("best_epoch", "N/A"),
            "Best Val Loss": round(r.data.metrics.get("best_val_loss", 0.0), 4)
            if "best_val_loss" in r.data.metrics
            else "N/A",
            "Val Acc (%)": round(r.data.metrics.get("val_accuracy_at_best_loss", 0.0) * 100, 2)
            if "val_accuracy_at_best_loss" in r.data.metrics
            else "N/A",
            "Test Top-1 Acc (%)": round(r.data.metrics.get("test_top1_accuracy", 0.0) * 100, 2)
            if "test_top1_accuracy" in r.data.metrics
            else "N/A",
            "Milestone": r.data.tags.get("milestone", "N/A"),
        }
        records.append(row)

    df = pd.DataFrame(records)
    return df


def populate_historical_experiments(
    tracking_uri: Optional[Union[str, Path]] = None,
    experiment_name: str = EXPERIMENT_NAME,
    force_relog: bool = False,
) -> Dict[str, str]:
    """
    Faithfully ingest actual verified historical experiment runs from Days 6, 8, 9, 11
    into the local MLflow store.

    Runs Ingested:
    1. EfficientNet-B0-Baseline (Day 6 / Day 8 Exp 0 + Day 9 locked held-out test evaluation)
    2. EfficientNet-B0-LR-5e-4 (Day 8 Exp 1)
    3. ResNet50-Benchmark (Day 11 controlled CPU benchmark)

    Preserves strict scientific integrity:
    - Skipped Day 8 runs (Exp 2-4) are NOT fabricated.
    - Day 9 held-out test evaluation is explicitly marked as historical final evaluation.
    - Zero retraining is performed.

    Args:
        tracking_uri: Local tracking directory or URI.
        experiment_name: Target MLflow experiment.
        force_relog: Whether to re-log runs if they already exist in the experiment.

    Returns:
        Dict[str, str]: Mapping of run name to MLflow run ID.
    """
    uri, exp_id = setup_mlflow(tracking_uri, experiment_name)
    client = MlflowClient(tracking_uri=uri)

    # Check existing runs to avoid duplicate logging
    existing_runs = client.search_runs(experiment_ids=[exp_id])
    existing_run_names = {r.data.tags.get("mlflow.runName"): r.info.run_id for r in existing_runs}

    run_ids: Dict[str, str] = {}

    # Common system parameters
    py_version = "3.14.5"
    torch_version = torch.__version__
    torchvision_version = "0.28.0+cpu"
    git_commit = "07c463cfb146197e06d5f483e7a4ce85048193f8"

    # ═════════════════════════════════════════════════════════════════
    # 1. RUN 1: EfficientNet-B0-Baseline (Day 6 / Day 8 Exp 0)
    # ═════════════════════════════════════════════════════════════════
    run_name_1 = "EfficientNet-B0-Baseline"
    if run_name_1 in existing_run_names and not force_relog:
        logger.info(f"Run '{run_name_1}' already tracked in MLflow (ID: {existing_run_names[run_name_1]}).")
        run_ids[run_name_1] = existing_run_names[run_name_1]
    else:
        with mlflow.start_run(
            run_name=run_name_1,
            description=(
                "Primary MediScan baseline model. EfficientNet-B0 backbone with frozen feature extractor "
                "and Linear(1280, 7) classification head trained with Adam (lr=1e-3, dropout=0.2) for 15 epochs. "
                "Retained as production deployment architecture."
            ),
        ) as run1:
            run_id_1 = run1.info.run_id
            run_ids[run_name_1] = run_id_1

            # Tags
            mlflow.set_tags({
                "milestone": "Day 6 / Day 8 Exp 0",
                "model_role": "Production Deployment Model",
                "model_retained": "True",
                "status": "COMPLETED",
                "checkpoint_canonical_path": "models/checkpoints/efficientnet_b0_best.pth",
                "checkpoint_md5": "7c1c6fcbe02e93f0ff8b4a20f62e29e3",
                "gradcam_target_layer": "model.features[8]",
                "test_evaluation_status": "locked_held_out_evaluation",
                "test_evaluation_note": "Day 9 final locked test evaluation. NOT used for model selection.",
            })

            # Parameters
            mlflow.log_params({
                "architecture": "efficientnet_b0",
                "pretrained_weights": "IMAGENET1K_V1",
                "num_classes": 7,
                "classifier_head": "Dropout(p=0.2) -> Linear(1280, 7)",
                "dropout": 0.2,
                "backbone_frozen": True,
                "total_parameters": 4016515,
                "trainable_parameters": 8967,
                "dataset_name": "HAM10000",
                "train_samples": 6982,
                "val_samples": 1521,
                "test_samples": 1512,
                "split_strategy": "stratified_group_lesion_id",
                "input_resolution": "224x224",
                "normalize_mean": "[0.485, 0.456, 0.406]",
                "normalize_std": "[0.229, 0.224, 0.225]",
                "augmentation": "baseline (hflip, vflip, rotate30, shiftscalerotate, brightness_contrast)",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_epochs": 15,
                "scheduler": "ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)",
                "early_stopping_patience": 3,
                "loss_function": "CrossEntropyLoss",
                "random_seed": 42,
                "device": "cpu",
                "python_version": py_version,
                "pytorch_version": torch_version,
                "torchvision_version": torchvision_version,
                "git_commit": git_commit,
            })

            # Epoch-level metrics from day6_training_history.csv
            hist_file_1 = PROJECT_ROOT / "reports" / "data" / "day6_training_history.csv"
            if hist_file_1.exists():
                df_hist_1 = pd.read_csv(hist_file_1)
                for _, row in df_hist_1.iterrows():
                    ep = int(row["epoch"])
                    mlflow.log_metric("train_loss", float(row["train_loss"]), step=ep)
                    mlflow.log_metric("train_accuracy", float(row["train_accuracy"]), step=ep)
                    mlflow.log_metric("val_loss", float(row["val_loss"]), step=ep)
                    mlflow.log_metric("val_accuracy", float(row["val_accuracy"]), step=ep)
                    mlflow.log_metric("learning_rate", float(row["learning_rate"]), step=ep)
                    if "epoch_time_seconds" in row:
                        mlflow.log_metric("epoch_time_seconds", float(row["epoch_time_seconds"]), step=ep)

            # Summary Validation Metrics
            mlflow.log_metrics({
                "best_epoch": 14,
                "best_val_loss": 0.6422,
                "val_accuracy_at_best_loss": 0.7738,
                "peak_val_accuracy": 0.7765,
                "peak_val_accuracy_epoch": 15,
                "total_training_duration_seconds": 3724.62,
            })

            # Locked Day 9 Test Metrics (Provenance clearly marked)
            mlflow.log_metrics({
                "test_top1_accuracy": 0.757937,
                "test_top2_accuracy": 0.895503,
                "test_macro_precision": 0.574358,
                "test_macro_recall": 0.478688,
                "test_macro_f1": 0.515696,
                "test_weighted_precision": 0.738198,
                "test_weighted_recall": 0.757937,
                "test_weighted_f1": 0.744771,
                "test_macro_roc_auc": 0.9211,
                "test_weighted_roc_auc": 0.9145,
                "test_macro_pr_auc": 0.5646,
                "test_ce_loss": 0.663784,
            })

            # Artifacts logging
            artifacts_to_log = [
                PROJECT_ROOT / "reports" / "data" / "day6_training_history.csv",
                PROJECT_ROOT / "reports" / "data" / "day9_classification_report.csv",
                PROJECT_ROOT / "reports" / "data" / "day9_metrics.json",
                PROJECT_ROOT / "config" / "config.yaml",
            ]
            for art in artifacts_to_log:
                if art.exists():
                    mlflow.log_artifact(str(art))

    # ═════════════════════════════════════════════════════════════════
    # 2. RUN 2: EfficientNet-B0-LR-5e-4 (Day 8 Experiment 1)
    # ═════════════════════════════════════════════════════════════════
    run_name_2 = "EfficientNet-B0-LR-5e-4"
    if run_name_2 in existing_run_names and not force_relog:
        logger.info(f"Run '{run_name_2}' already tracked in MLflow (ID: {existing_run_names[run_name_2]}).")
        run_ids[run_name_2] = existing_run_names[run_name_2]
    else:
        with mlflow.start_run(
            run_name=run_name_2,
            description=(
                "Day 8 Hyperparameter Tuning Experiment 1: Evaluated lower learning rate (5e-4 vs baseline 1e-3) "
                "on EfficientNet-B0 for 5 epochs. Converged slower with higher validation loss (0.7283 vs 0.6422)."
            ),
        ) as run2:
            run_id_2 = run2.info.run_id
            run_ids[run_name_2] = run_id_2

            mlflow.set_tags({
                "milestone": "Day 8 Exp 1",
                "model_role": "Hyperparameter Candidate",
                "model_retained": "False (Baseline 1e-3 outperformed 5e-4)",
                "status": "COMPLETED",
                "checkpoint_canonical_path": "models/checkpoints/day8_exp1_lower_lr_5e4_best.pth",
                "skipped_experiments_context": (
                    "Day 8 Experiments 2, 3, 4 were skipped per CPU compute budget guidelines. "
                    "Only genuine executed runs are tracked."
                ),
            })

            mlflow.log_params({
                "architecture": "efficientnet_b0",
                "pretrained_weights": "IMAGENET1K_V1",
                "num_classes": 7,
                "classifier_head": "Dropout(p=0.2) -> Linear(1280, 7)",
                "dropout": 0.2,
                "backbone_frozen": True,
                "optimizer": "Adam",
                "learning_rate": 0.0005,
                "batch_size": 32,
                "max_epochs": 5,
                "scheduler": "ReduceLROnPlateau(mode='min', factor=0.5, patience=2, min_lr=1e-6)",
                "early_stopping_patience": 3,
                "loss_function": "CrossEntropyLoss",
                "random_seed": 42,
                "device": "cpu",
                "python_version": py_version,
                "pytorch_version": torch_version,
                "torchvision_version": torchvision_version,
                "git_commit": git_commit,
            })

            # Epoch-level metrics from day8_history_exp1.csv
            hist_file_2 = PROJECT_ROOT / "reports" / "data" / "day8_history_exp1.csv"
            if hist_file_2.exists():
                df_hist_2 = pd.read_csv(hist_file_2)
                for _, row in df_hist_2.iterrows():
                    ep = int(row["epoch"])
                    mlflow.log_metric("train_loss", float(row["train_loss"]), step=ep)
                    mlflow.log_metric("train_accuracy", float(row["train_accuracy"]), step=ep)
                    mlflow.log_metric("val_loss", float(row["val_loss"]), step=ep)
                    mlflow.log_metric("val_accuracy", float(row["val_accuracy"]), step=ep)
                    mlflow.log_metric("learning_rate", float(row["learning_rate"]), step=ep)
                    if "epoch_time_seconds" in row:
                        mlflow.log_metric("epoch_time_seconds", float(row["epoch_time_seconds"]), step=ep)

            # Summary Metrics
            mlflow.log_metrics({
                "best_epoch": 5,
                "best_val_loss": 0.7283,
                "val_accuracy_at_best_loss": 0.7521,
                "peak_val_accuracy": 0.7521,
                "total_training_duration_seconds": 1159.99,
            })

            # Artifacts
            artifacts_exp1 = [
                PROJECT_ROOT / "reports" / "data" / "day8_history_exp1.csv",
                PROJECT_ROOT / "reports" / "data" / "day8_experiments.csv",
            ]
            for art in artifacts_exp1:
                if art.exists():
                    mlflow.log_artifact(str(art))

    # ═════════════════════════════════════════════════════════════════
    # 3. RUN 3: ResNet50-Benchmark (Day 11 Architecture Comparison)
    # ═════════════════════════════════════════════════════════════════
    run_name_3 = "ResNet50-Benchmark"
    if run_name_3 in existing_run_names and not force_relog:
        logger.info(f"Run '{run_name_3}' already tracked in MLflow (ID: {existing_run_names[run_name_3]}).")
        run_ids[run_name_3] = existing_run_names[run_name_3]
    else:
        with mlflow.start_run(
            run_name=run_name_3,
            description=(
                "Day 11 Architecture Comparison: Pretrained ResNet-50 evaluated under controlled matched protocol "
                "with frozen backbone and Linear(2048, 7) head. Epoch duration 5265.4s (20.3x slower on CPU). "
                "EfficientNet-B0 was retained for edge deployment."
            ),
        ) as run3:
            run_id_3 = run3.info.run_id
            run_ids[run_name_3] = run_id_3

            mlflow.set_tags({
                "milestone": "Day 11 Benchmark",
                "model_role": "Architectural Comparison Baseline",
                "model_retained": "False (EfficientNet-B0 retained: 5.86x smaller, 20.3x faster)",
                "status": "COMPLETED",
                "checkpoint_canonical_path": "models/checkpoints/resnet50_best.pth",
                "gradcam_target_layer": "model.layer4[-1]",
                "compute_constraint_note": (
                    "Controlled 1-epoch benchmark executed under CPU budget constraint. "
                    "Full 15-epoch test evaluation was NOT performed for ResNet-50."
                ),
            })

            mlflow.log_params({
                "architecture": "resnet50",
                "pretrained_weights": "ResNet50_Weights.IMAGENET1K_V1",
                "num_classes": 7,
                "classifier_head": "Dropout(p=0.2) -> Linear(2048, 7)",
                "dropout": 0.2,
                "backbone_frozen": True,
                "total_parameters": 23522375,
                "trainable_parameters": 14343,
                "dataset_name": "HAM10000",
                "train_samples": 6982,
                "val_samples": 1521,
                "test_samples": 1512,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "batch_size": 32,
                "max_epochs": 1,
                "loss_function": "CrossEntropyLoss",
                "random_seed": 42,
                "device": "cpu",
                "python_version": py_version,
                "pytorch_version": torch_version,
                "torchvision_version": torchvision_version,
                "git_commit": git_commit,
            })

            # Epoch 1 metric from day11_resnet50_training_history.csv
            hist_file_3 = PROJECT_ROOT / "reports" / "data" / "day11_resnet50_training_history.csv"
            if hist_file_3.exists():
                df_hist_3 = pd.read_csv(hist_file_3)
                for _, row in df_hist_3.iterrows():
                    ep = int(row["epoch"])
                    mlflow.log_metric("train_loss", float(row["train_loss"]), step=ep)
                    mlflow.log_metric("train_accuracy", float(row["train_accuracy"]), step=ep)
                    mlflow.log_metric("val_loss", float(row["val_loss"]), step=ep)
                    mlflow.log_metric("val_accuracy", float(row["val_accuracy"]), step=ep)
                    mlflow.log_metric("learning_rate", float(row["learning_rate"]), step=ep)
                    dur_col = "epoch_duration_seconds" if "epoch_duration_seconds" in row else "epoch_time_seconds"
                    if dur_col in row:
                        mlflow.log_metric("epoch_duration_seconds", float(row[dur_col]), step=ep)

            # Summary Metrics
            mlflow.log_metrics({
                "best_epoch": 1,
                "best_val_loss": 0.7646,
                "val_accuracy_at_best_loss": 0.7252,
                "total_training_duration_seconds": 5265.4,
                "cpu_throughput_slowdown_ratio_vs_efficientnet": 20.3,
            })

            # Artifacts
            artifacts_resnet = [
                PROJECT_ROOT / "reports" / "data" / "day11_resnet50_training_history.csv",
                PROJECT_ROOT / "reports" / "data" / "day11_model_comparison.csv",
            ]
            fig_dir = PROJECT_ROOT / "reports" / "figures" / "day11"
            if fig_dir.is_dir():
                for fig in fig_dir.glob("*.png"):
                    artifacts_resnet.append(fig)

            for art in artifacts_resnet:
                if art.exists():
                    mlflow.log_artifact(str(art))

    return run_ids
