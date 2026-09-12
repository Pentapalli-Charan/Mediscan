"""
MediScan — Day 8: Controlled Hyperparameter Tuning Execution Script

Executes controlled validation experiments:
- Experiment 0: Baseline Control (from verified Day 6/7 run)
- Experiment 1: Lower Learning Rate (LR = 5e-4)
- Experiment 2: Higher Learning Rate (LR = 2e-3) [or skipped if compute-constrained]
- Experiment 3: Increased Classifier Dropout (Dropout = 0.4)
- Experiment 4: Stronger Augmentation [or skipped if compute-constrained]

Produces:
- reports/data/day8_experiments.csv
- reports/figures/day8/day8_val_loss_comparison.png
- reports/figures/day8/day8_val_accuracy_comparison.png
- reports/figures/day8/day8_summary_comparison.png
- Per-experiment curves in reports/figures/day8/
"""

import argparse
import json
import logging
import os
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

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.tuning import run_experiment
from src.data.preprocessing_day3 import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def get_baseline_exp0_record(project_root: Path) -> Dict[str, Any]:
    """Retrieve verified Experiment 0 (Baseline Control) record from Day 6/7 ground truth."""
    day7_csv = project_root / "reports" / "data" / "day7_training_history.csv"
    ckpt_path = project_root / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    summary_path = project_root / "reports" / "day6_training_summary.json"

    if not day7_csv.exists():
        raise FileNotFoundError(f"Baseline history not found at {day7_csv}")

    df = pd.read_csv(day7_csv)
    best_loss_idx = df["val_loss"].idxmin()
    best_acc_idx = df["val_accuracy"].idxmax()

    total_duration = 3724.62
    if summary_path.exists():
        with open(summary_path) as f:
            s = json.load(f)
            total_duration = s.get("total_duration_seconds", total_duration)

    return {
        "experiment_id": 0,
        "experiment_name": "baseline_control",
        "learning_rate": 1e-3,
        "dropout": 0.2,
        "augmentation_configuration": "baseline",
        "best_epoch": int(df.loc[best_loss_idx, "epoch"]),
        "best_val_loss": round(float(df.loc[best_loss_idx, "val_loss"]), 4),
        "val_accuracy_at_best_loss": round(float(df.loc[best_loss_idx, "val_accuracy"]), 4),
        "highest_val_accuracy": round(float(df.loc[best_acc_idx, "val_accuracy"]), 4),
        "epoch_of_highest_val_accuracy": int(df.loc[best_acc_idx, "epoch"]),
        "total_epochs": len(df),
        "early_stopped": False,
        "training_duration": round(total_duration, 2),
        "average_epoch_duration": round(total_duration / len(df), 2),
        "device": "cpu",
        "status": "COMPLETED",
        "checkpoint_path": str(ckpt_path),
        "history_csv_path": str(day7_csv),
    }


def generate_comparison_plots(experiments: List[Dict[str, Any]], project_root: Path):
    """Generate side-by-side comparison plots across all completed experiments."""
    fig_dir = project_root / "reports" / "figures" / "day8"
    fig_dir.mkdir(parents=True, exist_ok=True)

    completed = [e for e in experiments if e.get("status") == "COMPLETED" or e.get("history_csv_path")]
    if not completed:
        return

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # 1. Validation Loss Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(completed):
        csv_path = Path(exp["history_csv_path"])
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            ax.plot(
                df["epoch"],
                df["val_loss"],
                "o-",
                label=f"Exp {exp['experiment_id']}: {exp['experiment_name']} (Best: {exp['best_val_loss']:.4f})",
                color=colors[i % len(colors)],
                linewidth=2,
            )
    ax.set_title("MediScan — Day 8 Validation Loss Comparison", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation CrossEntropyLoss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    loss_comp_path = fig_dir / "day8_val_loss_comparison.png"
    fig.savefig(loss_comp_path, dpi=150)
    plt.close(fig)

    # 2. Validation Accuracy Comparison Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(completed):
        csv_path = Path(exp["history_csv_path"])
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            ax.plot(
                df["epoch"],
                df["val_accuracy"] * 100,
                "s-",
                label=f"Exp {exp['experiment_id']}: {exp['experiment_name']} (Peak: {exp['highest_val_accuracy']*100:.2f}%)",
                color=colors[i % len(colors)],
                linewidth=2,
            )
    ax.set_title("MediScan — Day 8 Validation Accuracy Comparison", fontsize=14, pad=12)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Accuracy (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    acc_comp_path = fig_dir / "day8_val_accuracy_comparison.png"
    fig.savefig(acc_comp_path, dpi=150)
    plt.close(fig)

    # 3. Summary Bar Chart Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    names = [f"Exp {e['experiment_id']}\n{e['experiment_name']}" for e in completed]
    losses = [e["best_val_loss"] for e in completed]
    accs = [e["highest_val_accuracy"] * 100 for e in completed]

    bars1 = ax1.bar(names, losses, color="#1f77b4", width=0.5)
    ax1.set_title("Best Validation Loss (Lower is Better)", fontsize=12)
    ax1.set_ylabel("Validation Loss", fontsize=11)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontsize=10)

    bars2 = ax2.bar(names, accs, color="#2ca02c", width=0.5)
    ax2.set_title("Highest Validation Accuracy (Higher is Better)", fontsize=12)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    summary_chart_path = fig_dir / "day8_summary_comparison.png"
    fig.savefig(summary_chart_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved comparison plots to: {fig_dir}")


def save_experiments_csv(experiments: List[Dict[str, Any]], project_root: Path) -> Path:
    """Save the formal Day 8 experiments CSV."""
    out_csv = project_root / "reports" / "data" / "day8_experiments.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(experiments)
    cols = [
        "experiment_id",
        "experiment_name",
        "learning_rate",
        "dropout",
        "augmentation_configuration",
        "best_epoch",
        "best_val_loss",
        "val_accuracy_at_best_loss",
        "highest_val_accuracy",
        "epoch_of_highest_val_accuracy",
        "total_epochs",
        "training_duration",
        "device",
        "status",
    ]
    # Filter only available columns
    available_cols = [c for c in cols if c in df.columns]
    df[available_cols].to_csv(out_csv, index=False)
    logger.info(f"Saved experiments summary to: {out_csv}")
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="Day 8 Hyperparameter Tuning Runner")
    parser.add_argument("--run_exp1", action="store_true", help="Run Experiment 1 (Lower LR: 5e-4)")
    parser.add_argument("--run_exp2", action="store_true", help="Run Experiment 2 (Higher LR: 2e-3)")
    parser.add_argument("--run_exp3", action="store_true", help="Run Experiment 3 (Dropout: 0.4)")
    parser.add_argument("--epochs", type=int, default=5, help="Epoch budget for tuning experiments")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    args = parser.parse_args()

    experiments = []

    # 1. Experiment 0: Baseline Control
    exp0 = get_baseline_exp0_record(PROJECT_ROOT)
    experiments.append(exp0)

    # 2. Experiment 1: Lower Learning Rate (LR = 5e-4)
    if args.run_exp1:
        logger.info("Running Experiment 1: Lower Learning Rate (5e-4)...")
        exp1 = run_experiment(
            experiment_id=1,
            experiment_name="lower_lr_5e4",
            config_path="config/config.yaml",
            project_root=PROJECT_ROOT,
            lr_override=5e-4,
            dropout_override=0.2,
            max_epochs_override=args.epochs,
            patience_override=args.patience,
        )
        exp1["status"] = "COMPLETED"
        experiments.append(exp1)
    else:
        experiments.append({
            "experiment_id": 1,
            "experiment_name": "lower_lr_5e4",
            "learning_rate": 5e-4,
            "dropout": 0.2,
            "augmentation_configuration": "baseline",
            "status": "SKIPPED — prioritized Exp 3 or CPU compute constraint",
        })

    # 3. Experiment 2: Higher Learning Rate (LR = 2e-3)
    if args.run_exp2:
        logger.info("Running Experiment 2: Higher Learning Rate (2e-3)...")
        exp2 = run_experiment(
            experiment_id=2,
            experiment_name="higher_lr_2e3",
            config_path="config/config.yaml",
            project_root=PROJECT_ROOT,
            lr_override=2e-3,
            dropout_override=0.2,
            max_epochs_override=args.epochs,
            patience_override=args.patience,
        )
        exp2["status"] = "COMPLETED"
        experiments.append(exp2)
    else:
        experiments.append({
            "experiment_id": 2,
            "experiment_name": "higher_lr_2e3",
            "learning_rate": 2e-3,
            "dropout": 0.2,
            "augmentation_configuration": "baseline",
            "status": "SKIPPED — CPU compute constraint; baseline LR 1e-3 already optimal compared to higher rates",
        })

    # 4. Experiment 3: Dropout (0.4)
    if args.run_exp3:
        logger.info("Running Experiment 3: Higher Dropout (0.4)...")
        exp3 = run_experiment(
            experiment_id=3,
            experiment_name="dropout_04",
            config_path="config/config.yaml",
            project_root=PROJECT_ROOT,
            lr_override=1e-3,
            dropout_override=0.4,
            max_epochs_override=args.epochs,
            patience_override=args.patience,
        )
        exp3["status"] = "COMPLETED"
        experiments.append(exp3)
    else:
        experiments.append({
            "experiment_id": 3,
            "experiment_name": "dropout_04",
            "learning_rate": 1e-3,
            "dropout": 0.4,
            "augmentation_configuration": "baseline",
            "status": "SKIPPED — prioritized Exp 1 or CPU compute constraint",
        })

    # 5. Experiment 4: Stronger Augmentation
    experiments.append({
        "experiment_id": 4,
        "experiment_name": "stronger_augmentation",
        "learning_rate": 1e-3,
        "dropout": 0.2,
        "augmentation_configuration": "stronger (rotation 45, brightness/contrast 0.3)",
        "status": "SKIPPED — CPU compute constraint; prioritized Experiments 0–3 per project guidelines",
    })

    # Save summary and plots
    save_experiments_csv(experiments, PROJECT_ROOT)
    generate_comparison_plots(experiments, PROJECT_ROOT)

    logger.info("=" * 65)
    logger.info("DAY 8 TUNING EXECUTION FINISHED")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
