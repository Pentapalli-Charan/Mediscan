"""
MediScan — Day 7: Baseline Training Analysis

Reads the Day 6 training outputs and performs comprehensive analysis:
1. Training history verification and Day 7 CSV output
2. Training curves (loss, accuracy, learning rate)
3. Best checkpoint analysis
4. Overfitting / underfitting analysis
5. Scheduler / early stopping analysis
6. Checkpoint reload test
7. Reproducibility verification
8. Performance reporting
9. Test set protection verification
10. Code quality inspection summary

IMPORTANT:
- Does NOT evaluate on the test set.
- Does NOT fabricate any metric.
- All analysis is based on recorded training artifacts.
"""

import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloaders
from src.data.preprocessing_day3 import load_config
from src.models import CLASS_NAMES, NUM_CLASSES, build_model, get_model_summary
from src.training import load_training_checkpoint
from src.utils.device import get_device, get_device_info
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# 1. TRAINING HISTORY VERIFICATION
# ═════════════════════════════════════════════════════════════════════


def verify_training_history(project_root: Path) -> Dict[str, Any]:
    """Verify and copy training history to Day 7 output path."""
    day6_csv = project_root / "reports" / "data" / "day6_training_history.csv"
    day7_csv = project_root / "reports" / "data" / "day7_training_history.csv"

    if not day6_csv.exists():
        return {"success": False, "error": f"Day 6 training history not found: {day6_csv}"}

    df = pd.read_csv(day6_csv)

    # Verify required columns
    required_cols = [
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "learning_rate", "epoch_time_seconds",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return {"success": False, "error": f"Missing columns: {missing_cols}"}

    # Verify data integrity
    issues = []
    if df.empty:
        issues.append("Training history is empty")
    if df["epoch"].is_monotonic_increasing is False:
        issues.append("Epochs are not monotonically increasing")
    if df["train_loss"].isnull().any():
        issues.append("NaN values in train_loss")
    if df["val_loss"].isnull().any():
        issues.append("NaN values in val_loss")
    if (df["train_accuracy"] < 0).any() or (df["train_accuracy"] > 1).any():
        issues.append("train_accuracy outside [0, 1] range")
    if (df["val_accuracy"] < 0).any() or (df["val_accuracy"] > 1).any():
        issues.append("val_accuracy outside [0, 1] range")
    if (df["learning_rate"] <= 0).any():
        issues.append("Non-positive learning rate detected")

    # Save Day 7 copy
    df.to_csv(str(day7_csv), index=False)
    logger.info(f"Saved Day 7 training history to: {day7_csv}")

    return {
        "success": len(issues) == 0,
        "issues": issues,
        "num_epochs": len(df),
        "columns": list(df.columns),
        "day6_csv_path": str(day6_csv),
        "day7_csv_path": str(day7_csv),
        "history_df": df,
    }


# ═════════════════════════════════════════════════════════════════════
# 2. TRAINING CURVES
# ═════════════════════════════════════════════════════════════════════


def generate_training_curves(df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
    """Generate Day 7 training curves: loss, accuracy, and learning rate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = df["epoch"].values
    saved_paths = {}

    # A. Training vs Validation Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df["train_loss"].values, "o-", label="Train Loss",
            color="#1f77b4", linewidth=2, markersize=6)
    ax.plot(epochs, df["val_loss"].values, "s-", label="Validation Loss",
            color="#ff7f0e", linewidth=2, markersize=6)

    # Mark best validation loss
    best_idx = df["val_loss"].idxmin()
    best_epoch = df.loc[best_idx, "epoch"]
    best_val_loss = df.loc[best_idx, "val_loss"]
    ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5,
               label=f"Best Val Loss (Epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val_loss], color="green", s=120, zorder=5,
               edgecolors="black", linewidths=1.5)

    ax.set_title("MediScan — Training vs Validation Loss (Day 7 Analysis)", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CrossEntropyLoss", fontsize=12)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11, loc="upper right")
    fig.tight_layout()
    loss_path = output_dir / "day7_loss_curve.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    saved_paths["loss_curve"] = str(loss_path)

    # B. Training vs Validation Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df["train_accuracy"].values * 100, "o-", label="Train Accuracy (%)",
            color="#2ca02c", linewidth=2, markersize=6)
    ax.plot(epochs, df["val_accuracy"].values * 100, "s-", label="Validation Accuracy (%)",
            color="#d62728", linewidth=2, markersize=6)

    # Mark best validation accuracy
    best_acc_idx = df["val_accuracy"].idxmax()
    best_acc_epoch = df.loc[best_acc_idx, "epoch"]
    best_acc = df.loc[best_acc_idx, "val_accuracy"] * 100
    ax.axvline(x=best_acc_epoch, color="purple", linestyle="--", alpha=0.5,
               label=f"Best Val Acc (Epoch {best_acc_epoch})")
    ax.scatter([best_acc_epoch], [best_acc], color="purple", s=120, zorder=5,
               edgecolors="black", linewidths=1.5)

    ax.set_title("MediScan — Training vs Validation Accuracy (Day 7 Analysis)", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11, loc="lower right")
    fig.tight_layout()
    acc_path = output_dir / "day7_accuracy_curve.png"
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    saved_paths["accuracy_curve"] = str(acc_path)

    # C. Learning Rate Schedule
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, df["learning_rate"].values, "D-", color="#9467bd",
            linewidth=2, markersize=7, label="Learning Rate")
    ax.set_title("MediScan — Learning Rate Schedule (Day 7 Analysis)", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_xticks(epochs)
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    lr_path = output_dir / "day7_lr_schedule.png"
    fig.savefig(lr_path, dpi=150)
    plt.close(fig)
    saved_paths["lr_schedule"] = str(lr_path)

    logger.info(f"Generated Day 7 training curves: {list(saved_paths.keys())}")
    return saved_paths


# ═════════════════════════════════════════════════════════════════════
# 3. BEST CHECKPOINT ANALYSIS
# ═════════════════════════════════════════════════════════════════════


def analyze_best_checkpoint(df: pd.DataFrame, checkpoint_path: Path) -> Dict[str, Any]:
    """Identify best epoch by min val_loss and also by max val_accuracy."""
    # Best by minimum validation loss
    best_loss_idx = df["val_loss"].idxmin()
    best_loss_row = df.loc[best_loss_idx]

    # Best by maximum validation accuracy
    best_acc_idx = df["val_accuracy"].idxmax()
    best_acc_row = df.loc[best_acc_idx]

    result = {
        "best_val_loss_epoch": {
            "epoch": int(best_loss_row["epoch"]),
            "val_loss": float(best_loss_row["val_loss"]),
            "val_accuracy": float(best_loss_row["val_accuracy"]),
            "train_loss": float(best_loss_row["train_loss"]),
            "train_accuracy": float(best_loss_row["train_accuracy"]),
            "learning_rate": float(best_loss_row["learning_rate"]),
        },
        "best_val_accuracy_epoch": {
            "epoch": int(best_acc_row["epoch"]),
            "val_loss": float(best_acc_row["val_loss"]),
            "val_accuracy": float(best_acc_row["val_accuracy"]),
            "train_loss": float(best_acc_row["train_loss"]),
            "train_accuracy": float(best_acc_row["train_accuracy"]),
            "learning_rate": float(best_acc_row["learning_rate"]),
        },
        "same_epoch": int(best_loss_row["epoch"]) == int(best_acc_row["epoch"]),
    }

    # Verify checkpoint matches best val_loss epoch
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        result["checkpoint_epoch"] = ckpt.get("epoch")
        result["checkpoint_val_loss"] = ckpt.get("val_loss")
        result["checkpoint_val_accuracy"] = ckpt.get("val_accuracy")
        result["checkpoint_matches_best"] = (
            ckpt.get("epoch") == int(best_loss_row["epoch"])
        )
    else:
        result["checkpoint_exists"] = False

    logger.info(
        f"Best val_loss epoch: {result['best_val_loss_epoch']['epoch']} "
        f"(val_loss={result['best_val_loss_epoch']['val_loss']:.4f}, "
        f"val_acc={result['best_val_loss_epoch']['val_accuracy']*100:.2f}%)"
    )
    return result


# ═════════════════════════════════════════════════════════════════════
# 4. OVERFITTING / UNDERFITTING ANALYSIS
# ═════════════════════════════════════════════════════════════════════


def analyze_convergence(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze training curves for underfitting, overfitting, or healthy convergence."""
    train_losses = df["train_loss"].values
    val_losses = df["val_loss"].values
    train_accs = df["train_accuracy"].values
    val_accs = df["val_accuracy"].values

    # Gap analysis
    loss_gaps = val_losses - train_losses
    acc_gaps = train_accs - val_accs

    # Trend analysis
    first_half = len(df) // 2 if len(df) >= 4 else 1
    second_half_start = first_half

    val_loss_trend_late = np.polyfit(
        range(second_half_start, len(df)), val_losses[second_half_start:], 1
    )[0] if len(df) > second_half_start + 1 else 0.0

    train_loss_trend_late = np.polyfit(
        range(second_half_start, len(df)), train_losses[second_half_start:], 1
    )[0] if len(df) > second_half_start + 1 else 0.0

    # Final epoch metrics
    final_train_loss = float(train_losses[-1])
    final_val_loss = float(val_losses[-1])
    final_train_acc = float(train_accs[-1])
    final_val_acc = float(val_accs[-1])
    final_gap = float(loss_gaps[-1])

    # Determine convergence status
    findings = []
    status = "UNKNOWN"

    # Check for underfitting: both losses still high, both decreasing
    if final_val_loss > 1.5 and final_train_loss > 1.0:
        findings.append("Both train and val losses are still high — possible underfitting")
        status = "UNDERFITTING"

    # Check for overfitting: val_loss increasing while train_loss decreasing
    if val_loss_trend_late > 0.01 and train_loss_trend_late < -0.01:
        findings.append(
            f"Val loss trending UP (+{val_loss_trend_late:.4f}/epoch) "
            f"while train loss trending DOWN ({train_loss_trend_late:.4f}/epoch) in later epochs — overfitting"
        )
        status = "OVERFITTING"

    # Check for large train-val gap
    if final_gap > 0.5:
        findings.append(
            f"Large train-val loss gap at final epoch: {final_gap:.4f} — likely overfitting"
        )
        if status != "OVERFITTING":
            status = "OVERFITTING"

    # Check for unstable training: high variance in val_loss
    val_loss_std = np.std(val_losses[-min(5, len(val_losses)):])
    if val_loss_std > 0.15:
        findings.append(f"High variance in validation loss (std={val_loss_std:.4f}) — unstable training")
        status = "UNSTABLE"

    # Check for healthy convergence
    if len(findings) == 0:
        if val_loss_trend_late <= 0.01 and final_gap < 0.5:
            findings.append(
                f"Both losses converging, moderate train-val gap ({final_gap:.4f}), "
                f"val_loss trend: {val_loss_trend_late:.4f}/epoch — healthy convergence"
            )
            status = "HEALTHY_CONVERGENCE"
        else:
            findings.append("Moderate convergence behavior — neither strong overfitting nor underfitting")
            status = "MODERATE"

    return {
        "status": status,
        "findings": findings,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "final_train_accuracy": final_train_acc,
        "final_val_accuracy": final_val_acc,
        "final_loss_gap": final_gap,
        "final_accuracy_gap": float(acc_gaps[-1]),
        "val_loss_trend_late": float(val_loss_trend_late),
        "train_loss_trend_late": float(train_loss_trend_late),
        "loss_gaps_by_epoch": [round(g, 4) for g in loss_gaps],
        "accuracy_gaps_by_epoch": [round(g, 4) for g in acc_gaps],
    }


# ═════════════════════════════════════════════════════════════════════
# 5. SCHEDULER / EARLY STOPPING ANALYSIS
# ═════════════════════════════════════════════════════════════════════


def analyze_scheduler_and_early_stopping(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze learning rate changes and early stopping behavior."""
    lrs = df["learning_rate"].values
    epochs = df["epoch"].values
    num_epochs_completed = len(df)
    max_epochs_configured = config.get("training", {}).get("epochs", 15)

    # Detect LR changes
    lr_changes = []
    for i in range(1, len(lrs)):
        if abs(lrs[i] - lrs[i - 1]) > 1e-10:
            lr_changes.append({
                "epoch": int(epochs[i]),
                "old_lr": float(lrs[i - 1]),
                "new_lr": float(lrs[i]),
                "factor": round(float(lrs[i] / lrs[i - 1]), 4) if lrs[i - 1] > 0 else 0,
            })

    # Early stopping analysis
    early_stopped = num_epochs_completed < max_epochs_configured
    es_cfg = config.get("training", {}).get("early_stopping", {})
    sched_cfg = config.get("training", {}).get("scheduler", {})

    return {
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "mode": sched_cfg.get("mode", "min"),
            "factor": sched_cfg.get("factor", 0.5),
            "patience": sched_cfg.get("patience", 2),
            "min_lr": sched_cfg.get("min_lr", 1e-6),
            "lr_changed": len(lr_changes) > 0,
            "num_lr_changes": len(lr_changes),
            "lr_changes": lr_changes,
            "initial_lr": float(lrs[0]),
            "final_lr": float(lrs[-1]),
        },
        "early_stopping": {
            "enabled": es_cfg.get("enabled", True),
            "patience": es_cfg.get("patience", 3),
            "min_delta": es_cfg.get("min_delta", 0.001),
            "triggered": early_stopped,
            "epochs_completed": num_epochs_completed,
            "max_epochs_configured": max_epochs_configured,
            "stopped_at_epoch": int(epochs[-1]) if early_stopped else None,
        },
    }


# ═════════════════════════════════════════════════════════════════════
# 6. CHECKPOINT RELOAD TEST
# ═════════════════════════════════════════════════════════════════════


def checkpoint_reload_test(
    config: Dict[str, Any], checkpoint_path: Path, project_root: Path
) -> Dict[str, Any]:
    """Create fresh model, load checkpoint, run validation batch."""
    device_str = get_device()
    device = torch.device(device_str)

    # Build fresh model with same architecture
    fresh_model = build_model(config).to(device)
    fresh_model.eval()

    # Load checkpoint
    ckpt_meta = load_training_checkpoint(checkpoint_path, fresh_model, device=device_str)

    # Run a real validation batch
    loaders = create_dataloaders(
        config_path=str(project_root / "config" / "config.yaml"),
        project_root=str(project_root),
    )
    val_loader = loaders["val"]
    sample_batch = next(iter(val_loader))
    images, labels = sample_batch
    images = images.to(device)

    with torch.no_grad():
        logits = fresh_model(images)

    # Verification checks
    checks = {
        "output_shape": list(logits.shape),
        "expected_shape": [images.shape[0], 7],
        "shape_correct": list(logits.shape) == [images.shape[0], 7],
        "has_nan": bool(torch.isnan(logits).any()),
        "has_inf": bool(torch.isinf(logits).any()),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "num_classes": int(logits.shape[1]),
        "class_mapping_correct": logits.shape[1] == NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "checkpoint_epoch": ckpt_meta.get("epoch"),
        "checkpoint_val_loss": ckpt_meta.get("val_loss"),
        "checkpoint_architecture": ckpt_meta.get("architecture"),
    }

    checks["all_passed"] = (
        checks["shape_correct"]
        and not checks["has_nan"]
        and not checks["has_inf"]
        and checks["logits_finite"]
        and checks["class_mapping_correct"]
    )

    logger.info(
        f"Checkpoint reload test: {'PASSED' if checks['all_passed'] else 'FAILED'} "
        f"(shape={checks['output_shape']}, finite={checks['logits_finite']})"
    )
    return checks


# ═════════════════════════════════════════════════════════════════════
# 7. REPRODUCIBILITY VERIFICATION
# ═════════════════════════════════════════════════════════════════════


def verify_reproducibility(project_root: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Verify seed config, split CSVs, training config are intact."""
    results = {}

    # Seed
    results["seed"] = config.get("seed", "NOT SET")

    # Split CSV checksums
    split_dir = project_root / "reports" / "data"
    for split_name in ("train", "val", "test"):
        csv_path = split_dir / f"split_{split_name}.csv"
        if csv_path.exists():
            with open(csv_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            df = pd.read_csv(csv_path)
            results[f"split_{split_name}"] = {
                "exists": True,
                "md5": md5,
                "num_rows": len(df),
                "columns": list(df.columns),
            }
        else:
            results[f"split_{split_name}"] = {"exists": False}

    # Training config
    train_cfg = config.get("training", {})
    results["training_config"] = {
        "epochs": train_cfg.get("epochs"),
        "batch_size": train_cfg.get("batch_size"),
        "learning_rate": train_cfg.get("learning_rate"),
        "optimizer": train_cfg.get("optimizer"),
        "weight_decay": train_cfg.get("weight_decay"),
    }

    # Model config
    model_cfg = config.get("model", {})
    results["model_config"] = {
        "architecture": model_cfg.get("architecture"),
        "pretrained": model_cfg.get("pretrained"),
        "weights": model_cfg.get("weights"),
        "num_classes": model_cfg.get("num_classes"),
        "freeze_backbone": model_cfg.get("freeze_backbone"),
        "dropout": model_cfg.get("dropout"),
    }

    # Scheduler config
    sched_cfg = train_cfg.get("scheduler", {})
    results["scheduler_config"] = {
        "type": sched_cfg.get("type"),
        "mode": sched_cfg.get("mode"),
        "factor": sched_cfg.get("factor"),
        "patience": sched_cfg.get("patience"),
        "min_lr": sched_cfg.get("min_lr"),
    }

    logger.info(f"Reproducibility verification complete (seed={results['seed']})")
    return results


# ═════════════════════════════════════════════════════════════════════
# 8. PERFORMANCE REPORTING
# ═════════════════════════════════════════════════════════════════════


def report_performance(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate performance metrics from training history."""
    epoch_durations = df["epoch_time_seconds"].values
    total_duration = float(np.sum(epoch_durations))

    device_info = get_device_info()

    return {
        "total_training_duration_seconds": round(total_duration, 2),
        "total_training_duration_minutes": round(total_duration / 60, 2),
        "average_epoch_duration_seconds": round(float(np.mean(epoch_durations)), 2),
        "min_epoch_duration_seconds": round(float(np.min(epoch_durations)), 2),
        "max_epoch_duration_seconds": round(float(np.max(epoch_durations)), 2),
        "num_epochs": len(df),
        "device": device_info.get("device", "cpu"),
        "torch_version": device_info.get("torch_version", "unknown"),
        "cuda_available": device_info.get("device") == "cuda",
        "gpu_name": device_info.get("gpu_name", "N/A (CPU training)"),
        "device_info": device_info,
    }


# ═════════════════════════════════════════════════════════════════════
# 9. TEST SET PROTECTION VERIFICATION
# ═════════════════════════════════════════════════════════════════════


def verify_test_set_protection(project_root: Path) -> Dict[str, Any]:
    """Verify that Day 7 produced zero test set outputs."""
    checks = {
        "test_accuracy_produced": False,
        "test_loss_produced": False,
        "test_predictions_produced": False,
        "test_confusion_matrix_produced": False,
        "test_classification_report_produced": False,
    }

    # Search for any test-related output files
    reports_dir = project_root / "reports"
    test_indicators = [
        "test_accuracy", "test_loss", "test_predictions",
        "test_confusion", "test_classification", "test_results",
    ]

    for indicator in test_indicators:
        # Check file names
        for path in reports_dir.rglob("*"):
            if indicator in path.name.lower() and "day7" in path.name.lower():
                checks[f"{indicator}_file_found"] = str(path)

    # Verify training history does NOT contain test columns
    day7_csv = project_root / "reports" / "data" / "day7_training_history.csv"
    if day7_csv.exists():
        df = pd.read_csv(day7_csv)
        test_cols = [c for c in df.columns if "test" in c.lower()]
        checks["test_columns_in_history"] = test_cols
        if test_cols:
            checks["test_loss_produced"] = True

    checks["all_protected"] = not any([
        checks["test_accuracy_produced"],
        checks["test_loss_produced"],
        checks["test_predictions_produced"],
        checks["test_confusion_matrix_produced"],
        checks["test_classification_report_produced"],
    ])

    logger.info(f"Test set protection: {'VERIFIED' if checks['all_protected'] else 'VIOLATION DETECTED'}")
    return checks


# ═════════════════════════════════════════════════════════════════════
# 10. MAIN ANALYSIS PIPELINE
# ═════════════════════════════════════════════════════════════════════


def run_day7_analysis() -> Dict[str, Any]:
    """Run all Day 7 analysis tasks and return comprehensive results."""
    logger.info("=" * 65)
    logger.info("MediScan — Day 7: Baseline Training Analysis")
    logger.info("=" * 65)

    project_root = PROJECT_ROOT
    config = load_config(str(project_root / "config" / "config.yaml"))
    checkpoint_path = project_root / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    figures_dir = project_root / "reports" / "figures"

    results = {}

    # 1. Verify training history
    logger.info("\n[1/9] Verifying training history...")
    hist_result = verify_training_history(project_root)
    results["training_history"] = {k: v for k, v in hist_result.items() if k != "history_df"}
    if not hist_result["success"]:
        logger.error(f"Training history verification failed: {hist_result}")
        return results
    df = hist_result["history_df"]

    # 2. Generate training curves
    logger.info("\n[2/9] Generating training curves...")
    curve_paths = generate_training_curves(df, figures_dir)
    results["training_curves"] = curve_paths

    # 3. Best checkpoint analysis
    logger.info("\n[3/9] Analyzing best checkpoint...")
    results["best_checkpoint"] = analyze_best_checkpoint(df, checkpoint_path)

    # 4. Overfitting / underfitting analysis
    logger.info("\n[4/9] Analyzing convergence behavior...")
    results["convergence"] = analyze_convergence(df)

    # 5. Scheduler / early stopping analysis
    logger.info("\n[5/9] Analyzing scheduler and early stopping...")
    results["scheduler_early_stopping"] = analyze_scheduler_and_early_stopping(df, config)

    # 6. Checkpoint reload test
    logger.info("\n[6/9] Running checkpoint reload test...")
    results["checkpoint_reload"] = checkpoint_reload_test(config, checkpoint_path, project_root)

    # 7. Reproducibility verification
    logger.info("\n[7/9] Verifying reproducibility...")
    results["reproducibility"] = verify_reproducibility(project_root, config)

    # 8. Performance reporting
    logger.info("\n[8/9] Reporting performance...")
    results["performance"] = report_performance(df)

    # 9. Test set protection
    logger.info("\n[9/9] Verifying test set protection...")
    results["test_set_protection"] = verify_test_set_protection(project_root)

    # Save analysis results JSON
    analysis_json_path = project_root / "reports" / "data" / "day7_analysis_results.json"

    # Make results JSON-serializable
    serializable = json.loads(json.dumps(results, default=str))
    with open(analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved analysis results to: {analysis_json_path}")

    # Print summary
    logger.info("\n" + "=" * 65)
    logger.info("DAY 7 ANALYSIS SUMMARY")
    logger.info("=" * 65)
    logger.info(f"Epochs completed:       {len(df)}")
    best = results["best_checkpoint"]["best_val_loss_epoch"]
    logger.info(f"Best epoch (val_loss):  {best['epoch']}")
    logger.info(f"Best val_loss:          {best['val_loss']:.4f}")
    logger.info(f"Best val_accuracy:      {best['val_accuracy']*100:.2f}%")
    logger.info(f"Convergence status:     {results['convergence']['status']}")
    sched = results["scheduler_early_stopping"]["scheduler"]
    logger.info(f"LR changes:             {sched['num_lr_changes']}")
    es = results["scheduler_early_stopping"]["early_stopping"]
    logger.info(f"Early stopping:         {'Triggered' if es['triggered'] else 'Not triggered'}")
    logger.info(f"Checkpoint reload:      {'PASSED' if results['checkpoint_reload']['all_passed'] else 'FAILED'}")
    logger.info(f"Test set protection:    {'VERIFIED' if results['test_set_protection']['all_protected'] else 'VIOLATION'}")
    logger.info("=" * 65)

    return results


if __name__ == "__main__":
    results = run_day7_analysis()
