"""
MediScan — Day 14: Edge-Case, Bias & Robustness Analysis Module

Analysis-only module implementing:
1. Integrity verification (Checkpoint MD5, Test Split MD5, Class Mapping, Day 9 Locked Metrics).
2. Class imbalance & representation analysis across train/val/test splits.
3. Per-class diagnostic performance & macro vs weighted disparities.
4. Confusion matrix & failure mode analysis (mel->nv, bcc->nv, akiec->nv).
5. Softmax confidence distributions & high-confidence error quantification.
6. Post-hoc Expected Calibration Error (ECE) & reliability diagram generation.
7. Synthetic robustness probes (controlled perturbations on stratified subset).
8. Image quality proxy distributions (brightness, contrast, Laplacian blur).
9. Demographic & anatomical subgroup analysis with missingness audit.
10. Grad-CAM failure mode mapping targeting model.features[8].

STRICT CONTRACT:
- ANALYSIS ONLY. Zero retraining, zero fine-tuning, zero weight modification.
- Zero threshold changes.
- Checkpoint and test split are read-only.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.explainability.gradcam import (
    GradCAM,
    get_default_target_layer,
    snapshot_model_weights,
    verify_model_weights_unchanged,
)
from src.explainability.visualization import (
    create_gradcam_overlay,
    plot_gradcam_triplet,
)
from src.models import CLASS_NAMES, NUM_CLASSES, build_model
from src.training.trainer import load_training_checkpoint

logger = logging.getLogger(__name__)

# Canonical constants
CANONICAL_CHECKPOINT_MD5 = "7c1c6fcbe02e93f0ff8b4a20f62e29e3"
CANONICAL_TEST_SPLIT_MD5 = "6a5ae1b65c25d504f78bc1da2361d82c"
CANONICAL_CLASS_MAPPING = {
    "akiec": 0,
    "bcc": 1,
    "bkl": 2,
    "df": 3,
    "mel": 4,
    "nv": 5,
    "vasc": 6,
}
LOCKED_DAY9_METRICS = {
    "top1_accuracy": 0.757937,
    "top2_accuracy": 0.895503,
    "macro_precision": 0.574358,
    "macro_recall": 0.478688,
    "macro_f1": 0.515696,
    "weighted_precision": 0.738198,
    "weighted_recall": 0.757937,
    "weighted_f1": 0.744771,
    "macro_roc_auc": 0.921136,
    "weighted_roc_auc": 0.914520,
    "macro_pr_auc": 0.564585,
    "test_loss": 0.663784,
}


def compute_file_md5(filepath: Union[str, Path]) -> str:
    """Calculate the MD5 checksum of a file on disk."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image_path(image_id: str, raw_dir: Path) -> Path:
    """Locate dermatoscopic image on disk across part folders."""
    image_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and "HAM10000_images" in d.name]
    for d in image_dirs:
        for ext in (".jpg", ".jpeg", ".png"):
            p = d / f"{image_id}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"Image not found on disk: {image_id}")


def get_eval_transform() -> A.Compose:
    """Standard validation/evaluation image transformation pipeline."""
    return A.Compose([
        A.Resize(224, 224, interpolation=1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


# ═════════════════════════════════════════════════════════════════════
# 1. INTEGRITY VERIFICATION
# ═════════════════════════════════════════════════════════════════════

def verify_day14_integrity(
    checkpoint_path: Union[str, Path],
    test_split_path: Union[str, Path],
    metrics_json_path: Union[str, Path],
    model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Verify the integrity of canonical artifacts and compare against locked Day 9 baseline.
    """
    ckpt_md5 = compute_file_md5(checkpoint_path)
    test_md5 = compute_file_md5(test_split_path)

    ckpt_match = (ckpt_md5 == CANONICAL_CHECKPOINT_MD5)
    test_match = (test_md5 == CANONICAL_TEST_SPLIT_MD5)

    with open(metrics_json_path, "r", encoding="utf-8") as f:
        day9_metrics = json.load(f)

    om = day9_metrics.get("overall_metrics", {})
    roc = day9_metrics.get("roc_auc", {})
    pr = day9_metrics.get("precision_recall_auc", {})

    metrics_verification = {
        "top1_accuracy": {
            "expected": LOCKED_DAY9_METRICS["top1_accuracy"],
            "actual": om.get("top1_accuracy"),
            "matches": bool(np.isclose(om.get("top1_accuracy", 0), LOCKED_DAY9_METRICS["top1_accuracy"], atol=1e-4)),
        },
        "top2_accuracy": {
            "expected": LOCKED_DAY9_METRICS["top2_accuracy"],
            "actual": om.get("top2_accuracy"),
            "matches": bool(np.isclose(om.get("top2_accuracy", 0), LOCKED_DAY9_METRICS["top2_accuracy"], atol=1e-4)),
        },
        "macro_f1": {
            "expected": LOCKED_DAY9_METRICS["macro_f1"],
            "actual": om.get("macro_f1"),
            "matches": bool(np.isclose(om.get("macro_f1", 0), LOCKED_DAY9_METRICS["macro_f1"], atol=1e-4)),
        },
        "weighted_f1": {
            "expected": LOCKED_DAY9_METRICS["weighted_f1"],
            "actual": om.get("weighted_f1"),
            "matches": bool(np.isclose(om.get("weighted_f1", 0), LOCKED_DAY9_METRICS["weighted_f1"], atol=1e-4)),
        },
        "macro_roc_auc": {
            "expected": LOCKED_DAY9_METRICS["macro_roc_auc"],
            "actual": roc.get("macro_roc_auc"),
            "matches": bool(np.isclose(roc.get("macro_roc_auc", 0), LOCKED_DAY9_METRICS["macro_roc_auc"], atol=1e-4)),
        },
        "weighted_roc_auc": {
            "expected": LOCKED_DAY9_METRICS["weighted_roc_auc"],
            "actual": roc.get("weighted_roc_auc"),
            "matches": bool(np.isclose(roc.get("weighted_roc_auc", 0), LOCKED_DAY9_METRICS["weighted_roc_auc"], atol=1e-4)),
        },
        "macro_pr_auc": {
            "expected": LOCKED_DAY9_METRICS["macro_pr_auc"],
            "actual": pr.get("macro_pr_auc"),
            "matches": bool(np.isclose(pr.get("macro_pr_auc", 0), LOCKED_DAY9_METRICS["macro_pr_auc"], atol=1e-4)),
        },
        "test_loss": {
            "expected": LOCKED_DAY9_METRICS["test_loss"],
            "actual": om.get("test_loss"),
            "matches": bool(np.isclose(om.get("test_loss", 0), LOCKED_DAY9_METRICS["test_loss"], atol=1e-4)),
        },
    }

    # Class mapping check
    class_mapping_verified = bool(CLASS_NAMES == list(CANONICAL_CLASS_MAPPING.keys()))

    # Grad-CAM target check
    gradcam_target_verified = True
    target_layer_str = "model.features[8]"
    if model is not None:
        target_layer = get_default_target_layer(model)
        gradcam_target_verified = bool(target_layer == model.features[8])

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_md5": ckpt_md5,
        "checkpoint_md5_valid": ckpt_match,
        "test_split_path": str(test_split_path),
        "test_split_md5": test_md5,
        "test_split_md5_valid": test_match,
        "class_mapping_verified": class_mapping_verified,
        "gradcam_target_layer": target_layer_str,
        "gradcam_target_verified": gradcam_target_verified,
        "metrics_verification": metrics_verification,
        "all_integrity_passed": bool(
            ckpt_match and test_match and class_mapping_verified and
            gradcam_target_verified and all(m["matches"] for m in metrics_verification.values())
        ),
    }


# ═════════════════════════════════════════════════════════════════════
# 2. CLASS IMBALANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def analyze_class_imbalance(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    test_csv: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Calculate class distribution across train/validation/test partitions.
    Identify representation imbalance, majority class, and minority class.
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    total_train = len(train_df)
    total_val = len(val_df)
    total_test = len(test_df)
    total_all = total_train + total_val + total_test

    train_counts = train_df["dx"].value_counts().to_dict()
    val_counts = val_df["dx"].value_counts().to_dict()
    test_counts = test_df["dx"].value_counts().to_dict()

    rows = []
    for c in CLASS_NAMES:
        tr_c = train_counts.get(c, 0)
        va_c = val_counts.get(c, 0)
        te_c = test_counts.get(c, 0)
        tot_c = tr_c + va_c + te_c

        rows.append({
            "class_name": c,
            "train_count": tr_c,
            "train_pct": round((tr_c / total_train) * 100, 2),
            "val_count": va_c,
            "val_pct": round((va_c / total_val) * 100, 2),
            "test_count": te_c,
            "test_pct": round((te_c / total_test) * 100, 2),
            "total_count": tot_c,
            "total_pct": round((tot_c / total_all) * 100, 2),
        })

    dist_df = pd.DataFrame(rows)

    # Imbalance ratio
    max_count = int(dist_df["total_count"].max())
    min_count = int(dist_df["total_count"].min())
    imbalance_ratio = round(max_count / min_count, 2)

    maj_class = dist_df.loc[dist_df["total_count"].idxmax(), "class_name"]
    min_class = dist_df.loc[dist_df["total_count"].idxmin(), "class_name"]

    dist_df.attrs["imbalance_ratio"] = imbalance_ratio
    dist_df.attrs["majority_class"] = maj_class
    dist_df.attrs["minority_class"] = min_class

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        dist_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        x = np.arange(len(CLASS_NAMES))
        width = 0.25

        ax.bar(x - width, dist_df["train_count"], width, label="Train", color="#2b5c8f")
        ax.bar(x, dist_df["val_count"], width, label="Validation", color="#4ba3e3")
        ax.bar(x + width, dist_df["test_count"], width, label="Test", color="#e87a5d")

        ax.set_title("MediScan Class Distribution Across Splits (Representation Imbalance)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Diagnostic Class", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sample Count", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.legend(frameon=True, facecolor="white", loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate imbalance ratio
        ax.text(
            0.02, 0.95,
            f"Imbalance Ratio: {imbalance_ratio}:1\nMajority: {maj_class} ({max_count})\nMinority: {min_class} ({min_count})",
            transform=ax.transAxes,
            fontsize=9.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da")
        )

        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return dist_df


# ═════════════════════════════════════════════════════════════════════
# 3. PER-CLASS PERFORMANCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def analyze_per_class_performance(
    predictions_csv: Union[str, Path],
    metrics_json_path: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    output_f1_png: Optional[Union[str, Path]] = None,
    output_recall_png: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Extract and format per-class performance metrics and aggregate metrics from locked Day 9 results.
    """
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        day9_metrics = json.load(f)

    per_class_list = day9_metrics.get("per_class_metrics", [])
    roc_dict = day9_metrics.get("roc_auc", {}).get("per_class_roc_auc", {})
    pr_dict = day9_metrics.get("precision_recall_auc", {}).get("per_class_pr_auc", {})

    rows = []
    for item in per_class_list:
        cls_name = item["class"]
        rows.append({
            "class": cls_name,
            "support": int(item["support"]),
            "precision": float(item["precision"]),
            "recall": float(item["recall"]),
            "f1_score": float(item["f1_score"]),
            "roc_auc": float(roc_dict.get(cls_name, 0.0)),
            "pr_auc": float(pr_dict.get(cls_name, 0.0)),
        })

    perf_df = pd.DataFrame(rows)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        perf_df.to_csv(output_csv, index=False)

    # Plot F1
    if output_f1_png:
        Path(output_f1_png).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        bars = ax.bar(perf_df["class"], perf_df["f1_score"], color="#3b82f6", width=0.55)
        ax.axhline(0.5157, color="#ef4444", linestyle="--", linewidth=1.5, label="Macro F1 (0.5157)")
        ax.axhline(0.7448, color="#10b981", linestyle="--", linewidth=1.5, label="Weighted F1 (0.7448)")

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, 1.05)
        ax.set_title("MediScan Per-Class F1 Score (Day 9 Locked Model)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Diagnostic Class", fontsize=10, fontweight="bold")
        ax.set_ylabel("F1 Score", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(output_f1_png)
        plt.close(fig)

    # Plot Recall
    if output_recall_png:
        Path(output_recall_png).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        bars = ax.bar(perf_df["class"], perf_df["recall"], color="#8b5cf6", width=0.55)
        ax.axhline(0.4787, color="#ef4444", linestyle="--", linewidth=1.5, label="Macro Recall (0.4787)")
        ax.axhline(0.7579, color="#10b981", linestyle="--", linewidth=1.5, label="Weighted Recall / Accuracy (0.7579)")

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, 1.05)
        ax.set_title("MediScan Per-Class Recall / Sensitivity (Day 9 Locked Model)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Diagnostic Class", fontsize=10, fontweight="bold")
        ax.set_ylabel("Recall / Sensitivity", fontsize=10, fontweight="bold")
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(output_recall_png)
        plt.close(fig)

    return perf_df


# ═════════════════════════════════════════════════════════════════════
# 4. CONFUSION / FAILURE-MODE ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def analyze_confusion_pairs(
    predictions_csv: Union[str, Path],
    metrics_json_path: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Examine confusion matrix errors to quantify all misclassification pairs.
    Focus on mel -> nv, bcc -> nv, akiec -> nv.
    """
    preds_df = pd.read_csv(predictions_csv)

    # Filter incorrect predictions
    errors_df = preds_df[preds_df["true_class"] != preds_df["predicted_class"]]
    total_errors = len(errors_df)

    # Calculate support per true class
    support_per_class = preds_df["true_class"].value_counts().to_dict()

    # Group by true_class and predicted_class
    pair_counts = errors_df.groupby(["true_class", "predicted_class"]).size().reset_index(name="count")
    pair_counts = pair_counts.sort_values(by="count", ascending=False)

    rows = []
    for _, r in pair_counts.iterrows():
        t_cls = r["true_class"]
        p_cls = r["predicted_class"]
        cnt = int(r["count"])
        t_supp = support_per_class[t_cls]
        pct_true = round((cnt / t_supp) * 100, 2)
        pct_err = round((cnt / total_errors) * 100, 2)

        rows.append({
            "true_class": t_cls,
            "predicted_class": p_cls,
            "count": cnt,
            "true_class_support": t_supp,
            "pct_of_true_class": pct_true,
            "pct_of_total_errors": pct_err,
        })

    confusion_df = pd.DataFrame(rows)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        confusion_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        top10 = confusion_df.head(10)
        labels = [f"{r['true_class']} → {r['predicted_class']}" for _, r in top10.iterrows()]

        # Highlight malignant to nv misclassifications in crimson
        colors = [
            "#dc2626" if (r["true_class"] in ["mel", "bcc", "akiec"] and r["predicted_class"] == "nv")
            else "#64748b" for _, r in top10.iterrows()
        ]

        bars = ax.barh(labels[::-1], top10["count"].values[::-1], color=colors[::-1], height=0.6)
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1, bar.get_y() + bar.get_height()/2., f"{int(w)} ({w/total_errors*100:.1f}%)",
                    ha="left", va="center", fontsize=9)

        ax.set_title("Top 10 Misclassification Confusion Pairs (Red = Malignant → Benign NV)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Sample Count (% of Total Errors)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Confusion Pair (True → Predicted)", fontsize=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return confusion_df


# ═════════════════════════════════════════════════════════════════════
# 5. CONFIDENCE ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def analyze_confidence(
    predictions_csv: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Compute confidence statistics for correct vs incorrect predictions,
    and quantify high-confidence errors (>= 0.80 and >= 0.90).
    """
    preds_df = pd.read_csv(predictions_csv)
    preds_df["is_correct"] = (preds_df["true_class"] == preds_df["predicted_class"])

    correct_conf = preds_df[preds_df["is_correct"]]["predicted_confidence"]
    incorrect_conf = preds_df[~preds_df["is_correct"]]["predicted_confidence"]

    stats_dict = {
        "correct": {
            "count": int(len(correct_conf)),
            "mean_confidence": float(round(correct_conf.mean(), 4)),
            "median_confidence": float(round(correct_conf.median(), 4)),
            "std_confidence": float(round(correct_conf.std(), 4)),
            "p25_confidence": float(round(correct_conf.quantile(0.25), 4)),
            "p75_confidence": float(round(correct_conf.quantile(0.75), 4)),
        },
        "incorrect": {
            "count": int(len(incorrect_conf)),
            "mean_confidence": float(round(incorrect_conf.mean(), 4)),
            "median_confidence": float(round(incorrect_conf.median(), 4)),
            "std_confidence": float(round(incorrect_conf.std(), 4)),
            "p25_confidence": float(round(incorrect_conf.quantile(0.25), 4)),
            "p75_confidence": float(round(incorrect_conf.quantile(0.75), 4)),
        }
    }

    # High-confidence error analysis
    err_df = preds_df[~preds_df["is_correct"]]
    total_preds = len(preds_df)
    total_errors = len(err_df)

    hc_80 = err_df[err_df["predicted_confidence"] >= 0.80]
    hc_90 = err_df[err_df["predicted_confidence"] >= 0.90]

    def pair_dict(df):
        d = {}
        for (t, p), v in df.groupby(["true_class", "predicted_class"]).size().items():
            d[f"{t} -> {p}"] = int(v)
        return d

    stats_dict["high_confidence_errors"] = {
        "threshold_80": {
            "count": int(len(hc_80)),
            "pct_of_errors": float(round((len(hc_80) / total_errors) * 100, 2)),
            "pct_of_predictions": float(round((len(hc_80) / total_preds) * 100, 2)),
            "most_frequent_true_classes": hc_80["true_class"].value_counts().to_dict(),
            "top_pairs": pair_dict(hc_80),
        },
        "threshold_90": {
            "count": int(len(hc_90)),
            "pct_of_errors": float(round((len(hc_90) / total_errors) * 100, 2)),
            "pct_of_predictions": float(round((len(hc_90) / total_preds) * 100, 2)),
            "most_frequent_true_classes": hc_90["true_class"].value_counts().to_dict(),
            "top_pairs": pair_dict(hc_90),
        },
    }

    # Format table for CSV
    summary_rows = [
        {"metric": "Count", "correct": stats_dict["correct"]["count"], "incorrect": stats_dict["incorrect"]["count"]},
        {"metric": "Mean Confidence", "correct": stats_dict["correct"]["mean_confidence"], "incorrect": stats_dict["incorrect"]["mean_confidence"]},
        {"metric": "Median Confidence", "correct": stats_dict["correct"]["median_confidence"], "incorrect": stats_dict["incorrect"]["median_confidence"]},
        {"metric": "Std Confidence", "correct": stats_dict["correct"]["std_confidence"], "incorrect": stats_dict["incorrect"]["std_confidence"]},
        {"metric": "25th Percentile", "correct": stats_dict["correct"]["p25_confidence"], "incorrect": stats_dict["incorrect"]["p25_confidence"]},
        {"metric": "75th Percentile", "correct": stats_dict["correct"]["p75_confidence"], "incorrect": stats_dict["incorrect"]["p75_confidence"]},
    ]
    summary_df = pd.DataFrame(summary_rows)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

        # Histogram / KDE
        sns.histplot(correct_conf, bins=25, color="#10b981", label=f"Correct (N={len(correct_conf)})", kde=True, ax=ax1, stat="density", alpha=0.4)
        sns.histplot(incorrect_conf, bins=25, color="#ef4444", label=f"Incorrect (N={len(incorrect_conf)})", kde=True, ax=ax1, stat="density", alpha=0.4)
        ax1.set_title("Softmax Confidence Distribution", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Predicted Class Softmax Confidence", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Boxplot
        box_data = [correct_conf.values, incorrect_conf.values]
        ax2.boxplot(box_data, tick_labels=["Correct", "Incorrect"], patch_artist=True,
                    boxprops=dict(facecolor="#e2e8f0", color="#475569"),
                    medianprops=dict(color="#2563eb", linewidth=2))
        ax2.axhline(0.80, color="#f59e0b", linestyle=":", label="80% High-Conf Threshold")
        ax2.axhline(0.90, color="#ef4444", linestyle=":", label="90% High-Conf Threshold")
        ax2.set_title("Confidence Spread & Quartiles", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Predicted Confidence", fontsize=10)
        ax2.legend(loc="lower left")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

        plt.suptitle("MediScan Softmax Confidence Analysis (Correct vs Misclassified)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return stats_dict


# ═════════════════════════════════════════════════════════════════════
# 6. CALIBRATION ANALYSIS (ECE)
# ═════════════════════════════════════════════════════════════════════

def compute_calibration(
    predictions_csv: Union[str, Path],
    n_bins: int = 10,
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Calculate post-hoc Expected Calibration Error (ECE) and reliability diagram.
    Uses 10 equal-width bins: [0, 0.1), ..., [0.9, 1.0].
    """
    preds_df = pd.read_csv(predictions_csv)
    confidences = preds_df["predicted_confidence"].values
    correctness = (preds_df["true_class"] == preds_df["predicted_class"]).values.astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_rows = []

    total_samples = len(confidences)
    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        b_low = bin_edges[i]
        b_high = bin_edges[i + 1]

        # Include upper boundary on the last bin
        if i == n_bins - 1:
            mask = (confidences >= b_low) & (confidences <= b_high)
        else:
            mask = (confidences >= b_low) & (confidences < b_high)

        count = int(np.sum(mask))
        if count > 0:
            bin_conf = float(np.mean(confidences[mask]))
            bin_acc = float(np.mean(correctness[mask]))
            gap = float(abs(bin_acc - bin_conf))
            ece += (count / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_conf = float((b_low + b_high) / 2.0)
            bin_acc = 0.0
            gap = 0.0

        bin_rows.append({
            "bin_index": i + 1,
            "bin_lower": round(b_low, 2),
            "bin_upper": round(b_high, 2),
            "sample_count": count,
            "sample_pct": round((count / total_samples) * 100, 2),
            "confidence": round(bin_conf, 4),
            "accuracy": round(bin_acc, 4),
            "calibration_gap": round(gap, 4),
        })

    calib_df = pd.DataFrame(bin_rows)
    calib_df.attrs["ece"] = round(ece, 4)
    calib_df.attrs["mce"] = round(mce, 4)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        calib_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), dpi=300, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        # Reliability diagram
        bin_centers = [(r["bin_lower"] + r["bin_upper"]) / 2.0 for _, r in calib_df.iterrows()]
        bar_width = 1.0 / n_bins * 0.85

        # Bar for accuracy
        ax1.bar(bin_centers, calib_df["accuracy"], width=bar_width, color="#3b82f6", alpha=0.7, label="Empirical Accuracy", edgecolor="#1d4ed8")
        # Line for ideal calibration
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration (Identity)")

        # Gap bars
        for idx, r in calib_df.iterrows():
            if r["sample_count"] > 0:
                ax1.plot([bin_centers[idx], bin_centers[idx]], [r["confidence"], r["accuracy"]], color="#ef4444", linewidth=2)

        ax1.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f"MediScan Model Reliability Diagram (ECE = {ece:.4f}, MCE = {mce:.4f})", fontsize=12, fontweight="bold", pad=10)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.3)

        # Bottom plot: sample count per bin
        ax2.bar(bin_centers, calib_df["sample_count"], width=bar_width, color="#64748b", edgecolor="#334155")
        ax2.set_xlabel("Confidence Bin", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Sample Count", fontsize=10, fontweight="bold")
        ax2.set_xlim(0, 1)
        ax2.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return float(ece), float(mce), calib_df


# ═════════════════════════════════════════════════════════════════════
# 7. SYNTHETIC ROBUSTNESS PROBES
# ═════════════════════════════════════════════════════════════════════

def apply_perturbation(image_rgb: np.ndarray, probe_type: str) -> np.ndarray:
    """
    Apply a deterministic image perturbation to an RGB numpy image in [0, 255].
    Returns a perturbed copy without modifying the original.
    """
    img = image_rgb.copy()
    h, w, c = img.shape

    if probe_type == "mild_brightness":
        # Multiply intensity by 1.15
        img = np.clip(img.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)

    elif probe_type == "mild_contrast":
        # Contrast adjustment (factor 1.15 centered at 128)
        img = np.clip((img.astype(np.float32) - 128.0) * 1.15 + 128.0, 0, 255).astype(np.uint8)

    elif probe_type == "gaussian_blur":
        # 5x5 kernel with sigma 1.0
        img = cv2.GaussianBlur(img, (5, 5), 1.0)

    elif probe_type == "jpeg_compression":
        # Encode as JPEG with quality 65, then decode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
        # Convert RGB to BGR for OpenCV encoder
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode(".jpg", bgr, encode_param)
        dec_bgr = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(dec_bgr, cv2.COLOR_BGR2RGB)

    elif probe_type == "crop_resize":
        # Central crop 90%, resize back to original (h, w)
        crop_h = int(h * 0.90)
        crop_w = int(w * 0.90)
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2
        cropped = img[y1:y1 + crop_h, x1:x1 + crop_w]
        img = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    elif probe_type == "mild_color_shift":
        # Slight tint: +12 on red, -6 on blue
        shifted = img.astype(np.float32)
        shifted[:, :, 0] = np.clip(shifted[:, :, 0] + 12.0, 0, 255)
        shifted[:, :, 2] = np.clip(shifted[:, :, 2] - 6.0, 0, 255)
        img = shifted.astype(np.uint8)

    elif probe_type == "baseline":
        pass
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    return img


def run_synthetic_robustness_probes(
    model: nn.Module,
    test_df: pd.DataFrame,
    raw_image_dir: Path,
    subset_size: int = 150,
    random_seed: int = 42,
    device: str = "cpu",
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Run controlled perturbation probes on a representative stratified test subset.
    Measures prediction consistency, class flip rate, confidence change, and accuracy.
    """
    model.eval()
    transform = get_eval_transform()

    # Stratified sampling of subset
    np.random.seed(random_seed)
    stratified_indices = []
    # Sample proportionally across classes
    unique_classes = test_df["dx"].unique()
    for cls in unique_classes:
        cls_idx = test_df[test_df["dx"] == cls].index.tolist()
        k = max(1, int(round(len(cls_idx) / len(test_df) * subset_size)))
        chosen = np.random.choice(cls_idx, size=min(k, len(cls_idx)), replace=False)
        stratified_indices.extend(chosen)

    # If count doesn't match subset_size exactly, adjust
    if len(stratified_indices) > subset_size:
        stratified_indices = stratified_indices[:subset_size]
    elif len(stratified_indices) < subset_size:
        remaining = [i for i in test_df.index if i not in stratified_indices]
        stratified_indices.extend(np.random.choice(remaining, size=subset_size - len(stratified_indices), replace=False))

    subset_df = test_df.loc[stratified_indices].copy().reset_index(drop=True)
    n_samples = len(subset_df)

    probes = [
        "baseline",
        "mild_brightness",
        "mild_contrast",
        "gaussian_blur",
        "jpeg_compression",
        "crop_resize",
        "mild_color_shift",
    ]

    c2i = {c: i for i, c in enumerate(CLASS_NAMES)}

    # Preload all subset images into memory
    cached_images = []
    for _, r in subset_df.iterrows():
        p = find_image_path(r["image_id"], raw_image_dir)
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cached_images.append(rgb)

    probe_results = {}
    with torch.no_grad():
        for probe in probes:
            preds_list = []
            confs_list = []
            for img_rgb in cached_images:
                perturbed = apply_perturbation(img_rgb, probe)
                transformed = transform(image=perturbed)["image"].unsqueeze(0).to(device)
                logits = model(transformed)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, dim=1)
                preds_list.append(int(pred.item()))
                confs_list.append(float(conf.item()))
            probe_results[probe] = {
                "preds": np.array(preds_list),
                "confs": np.array(confs_list),
            }

    # Calculate metrics relative to baseline
    base_preds = probe_results["baseline"]["preds"]
    base_confs = probe_results["baseline"]["confs"]
    true_labels = np.array([c2i[c] for c in subset_df["dx"]])

    rows = []
    for probe in probes:
        preds = probe_results[probe]["preds"]
        confs = probe_results[probe]["confs"]

        acc = (preds == true_labels).mean()
        consistency = (preds == base_preds).mean()
        flip_rate = (preds != base_preds).mean()
        mean_conf = confs.mean()
        mean_conf_delta = (confs - base_confs).mean()

        rows.append({
            "probe": probe,
            "sample_count": n_samples,
            "accuracy": round(float(acc), 4),
            "prediction_consistency": round(float(consistency), 4),
            "class_flip_rate": round(float(flip_rate), 4),
            "mean_confidence": round(float(mean_conf), 4),
            "mean_confidence_delta": round(float(mean_conf_delta), 4),
        })

    robust_df = pd.DataFrame(rows)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        robust_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

        # Consistency bar chart
        ax1.bar(robust_df["probe"], robust_df["prediction_consistency"] * 100, color="#2563eb", width=0.55)
        ax1.set_xticks(range(len(robust_df)))
        ax1.set_xticklabels(robust_df["probe"], rotation=35, ha="right", fontsize=9)
        ax1.set_ylim(50, 105)
        ax1.set_ylabel("Prediction Consistency (%)", fontsize=10, fontweight="bold")
        ax1.set_title("Prediction Consistency vs Baseline", fontsize=11, fontweight="bold")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        for b in bars1:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2., h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

        # Accuracy comparison
        bars2 = ax2.bar(robust_df["probe"], robust_df["accuracy"] * 100, color="#10b981", width=0.55)
        ax2.set_xticks(range(len(robust_df)))
        ax2.set_xticklabels(robust_df["probe"], rotation=35, ha="right", fontsize=9)
        ax2.set_ylim(50, 105)
        ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
        ax2.set_title("Accuracy Under Synthetic Perturbations", fontsize=11, fontweight="bold")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        for b in bars2:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width()/2., h + 0.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

        plt.suptitle(f"MediScan Synthetic Robustness Probes (N={n_samples}, Stratified Test Subset)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return robust_df


# ═════════════════════════════════════════════════════════════════════
# 8. IMAGE QUALITY PROXIES ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def analyze_image_quality(
    test_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    raw_image_dir: Path,
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Calculate simple image-quality proxies across the test set:
    - Dimensions (width, height)
    - Brightness (mean grayscale intensity)
    - Contrast (standard deviation of grayscale intensity)
    - Sharpness / Blur proxy (variance of the Laplacian)
    Compare distributions between correct and incorrect predictions.
    """
    merged = pd.merge(test_df[["image_id", "dx"]], predictions_df[["image_id", "predicted_class", "predicted_confidence"]], on="image_id")
    merged["is_correct"] = (merged["dx"] == merged["predicted_class"])

    brightness_list = []
    contrast_list = []
    blur_proxy_list = []
    widths = []
    heights = []

    for _, r in merged.iterrows():
        p = find_image_path(r["image_id"], raw_image_dir)
        bgr = cv2.imread(str(p))
        h, w = bgr.shape[:2]
        widths.append(w)
        heights.append(h)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        brightness_list.append(float(np.mean(gray)))
        contrast_list.append(float(np.std(gray)))
        # Variance of Laplacian
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_proxy_list.append(laplacian_var)

    merged["brightness"] = brightness_list
    merged["contrast"] = contrast_list
    merged["blur_proxy"] = blur_proxy_list

    corr = merged[merged["is_correct"]]
    inc = merged[~merged["is_correct"]]

    def get_stats(series: pd.Series) -> Dict[str, float]:
        return {
            "mean": round(float(series.mean()), 2),
            "median": round(float(series.median()), 2),
            "std": round(float(series.std()), 2),
            "p25": round(float(series.quantile(0.25)), 2),
            "p75": round(float(series.quantile(0.75)), 2),
        }

    b_corr = get_stats(corr["brightness"])
    b_inc = get_stats(inc["brightness"])
    c_corr = get_stats(corr["contrast"])
    c_inc = get_stats(inc["contrast"])
    bl_corr = get_stats(corr["blur_proxy"])
    bl_inc = get_stats(inc["blur_proxy"])

    rows = [
        {"proxy": "Brightness (Mean Intensity)", "correct_median": b_corr["median"], "correct_mean": b_corr["mean"], "incorrect_median": b_inc["median"], "incorrect_mean": b_inc["mean"]},
        {"proxy": "Contrast (Intensity Std)", "correct_median": c_corr["median"], "correct_mean": c_corr["mean"], "incorrect_median": c_inc["median"], "incorrect_mean": c_inc["mean"]},
        {"proxy": "Sharpness Proxy (Laplacian Var)", "correct_median": bl_corr["median"], "correct_mean": bl_corr["mean"], "incorrect_median": bl_inc["median"], "incorrect_mean": bl_inc["mean"]},
    ]
    summary_df = pd.DataFrame(rows)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4), dpi=300)

        proxies = [
            ("brightness", "Brightness (Mean Intensity)", axes[0]),
            ("contrast", "Contrast (Intensity Std)", axes[1]),
            ("blur_proxy", "Laplacian Variance (Sharpness)", axes[2]),
        ]

        for col, title, ax in proxies:
            data = [corr[col].values, inc[col].values]
            ax.boxplot(data, tick_labels=["Correct", "Incorrect"], patch_artist=True,
                       boxprops=dict(facecolor="#e2e8f0", color="#334155"),
                       medianprops=dict(color="#2563eb", linewidth=2))
            ax.set_title(title, fontsize=10.5, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.suptitle("MediScan Image Quality Proxies Comparison (Correct vs Misclassified)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return summary_df


# ═════════════════════════════════════════════════════════════════════
# 9. METADATA & SUBGROUP ANALYSIS
# ═════════════════════════════════════════════════════════════════════

def analyze_metadata_subgroups(
    split_all_csv: Union[str, Path],
    predictions_df: pd.DataFrame,
    output_csv: Optional[Union[str, Path]] = None,
    output_png: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Audit HAM10000 metadata fields (age, sex, localization) for test set samples.
    Quantify missingness and calculate descriptive subgroup performance metrics.
    """
    all_meta = pd.read_csv(split_all_csv)
    merged = pd.merge(predictions_df, all_meta[["image_id", "age", "sex", "localization"]], on="image_id", how="left")
    merged["is_correct"] = (merged["true_class"] == merged["predicted_class"])

    n_test = len(merged)

    # Missingness audit
    missingness = {
        "age_missing": int(merged["age"].isna().sum() + (merged["age"] == "unknown").sum()),
        "sex_missing": int(merged["sex"].isna().sum() + (merged["sex"] == "unknown").sum()),
        "localization_missing": int(merged["localization"].isna().sum() + (merged["localization"] == "unknown").sum()),
    }

    # 1. Sex groups
    rows = []
    for s_val in sorted([str(x) for x in merged["sex"].fillna("unknown").unique()]):
        sub = merged[merged["sex"].fillna("unknown") == s_val]
        cnt = len(sub)
        corr = sub["is_correct"].sum()
        acc = corr / cnt if cnt > 0 else 0.0
        rows.append({
            "category": "Sex",
            "group": str(s_val),
            "sample_count": cnt,
            "sample_pct": round((cnt / n_test) * 100, 2),
            "correct_count": int(corr),
            "accuracy": round(float(acc), 4),
        })

    # 2. Age groups (<40, 40-59, >=60, unknown)
    def assign_age_group(val):
        if pd.isna(val) or str(val).lower() == "unknown":
            return "unknown"
        try:
            f = float(val)
            if f < 40:
                return "<40"
            elif f < 60:
                return "40-59"
            else:
                return ">=60"
        except (ValueError, TypeError):
            return "unknown"

    merged["age_group"] = merged["age"].apply(assign_age_group)
    for a_val in ["<40", "40-59", ">=60", "unknown"]:
        sub = merged[merged["age_group"] == a_val]
        cnt = len(sub)
        if cnt > 0:
            corr = sub["is_correct"].sum()
            acc = corr / cnt
            rows.append({
                "category": "Age Group",
                "group": a_val,
                "sample_count": cnt,
                "sample_pct": round((cnt / n_test) * 100, 2),
                "correct_count": int(corr),
                "accuracy": round(float(acc), 4),
            })

    # 3. Top Anatomical Sites (support >= 30)
    loc_counts = merged["localization"].fillna("unknown").value_counts()
    for loc, cnt in loc_counts.items():
        if cnt >= 30:  # Sufficient sample size threshold
            sub = merged[merged["localization"].fillna("unknown") == loc]
            corr = sub["is_correct"].sum()
            acc = corr / cnt
            rows.append({
                "category": "Localization (Top)",
                "group": str(loc),
                "sample_count": cnt,
                "sample_pct": round((cnt / n_test) * 100, 2),
                "correct_count": int(corr),
                "accuracy": round(float(acc), 4),
            })

    subgroup_df = pd.DataFrame(rows)

    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        subgroup_df.to_csv(output_csv, index=False)

    if output_png:
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

        # Sex plot
        sex_df = subgroup_df[subgroup_df["category"] == "Sex"]
        bars1 = ax1.bar(sex_df["group"], sex_df["accuracy"] * 100, color="#3b82f6", width=0.5)
        ax1.axhline(75.79, color="#ef4444", linestyle="--", label="Overall Acc (75.79%)")
        ax1.set_ylim(50, 95)
        ax1.set_title("Descriptive Accuracy by Sex Group", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Accuracy (%)", fontsize=10)
        ax1.legend(loc="lower left")
        ax1.grid(axis="y", linestyle="--", alpha=0.3)
        for b, (_, r) in zip(bars1, sex_df.iterrows()):
            ax1.text(b.get_x() + b.get_width()/2., b.get_height() + 0.8, f"{b.get_height():.1f}%\n(N={r['sample_count']})", ha="center", va="bottom", fontsize=8.5)

        # Age group plot
        age_df = subgroup_df[subgroup_df["category"] == "Age Group"]
        bars2 = ax2.bar(age_df["group"], age_df["accuracy"] * 100, color="#10b981", width=0.5)
        ax2.axhline(75.79, color="#ef4444", linestyle="--", label="Overall Acc (75.79%)")
        ax2.set_ylim(50, 95)
        ax2.set_title("Descriptive Accuracy by Age Group", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Accuracy (%)", fontsize=10)
        ax2.legend(loc="lower left")
        ax2.grid(axis="y", linestyle="--", alpha=0.3)
        for b, (_, r) in zip(bars2, age_df.iterrows()):
            ax2.text(b.get_x() + b.get_width()/2., b.get_height() + 0.8, f"{b.get_height():.1f}%\n(N={r['sample_count']})", ha="center", va="bottom", fontsize=8.5)

        plt.suptitle("MediScan Metadata Subgroup Performance (Descriptive Only)", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(output_png)
        plt.close(fig)

    return subgroup_df, missingness


# ═════════════════════════════════════════════════════════════════════
# 10. GRAD-CAM FAILURE ANALYSIS (TARGET: model.features[8])
# ═════════════════════════════════════════════════════════════════════

def generate_day14_gradcam_failures(
    model: nn.Module,
    predictions_df: pd.DataFrame,
    raw_image_dir: Path,
    output_dir: Path,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """
    Generate Grad-CAM triplets targeting model.features[8] for representative:
    - Correct predictions (Melanoma, Nevus)
    - High-confidence misclassifications (Melanoma -> Nevus, BCC -> Nevus, AKIEC -> Nevus, BKL -> Nevus)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    transform = get_eval_transform()
    gradcam = GradCAM(model, target_layer=get_default_target_layer(model))

    cases_to_find = [
        {"desc": "Correct Melanoma", "true": "mel", "pred": "mel", "prefix": "correct_mel"},
        {"desc": "Correct Nevus", "true": "nv", "pred": "nv", "prefix": "correct_nv"},
        {"desc": "High-Confidence Error: Melanoma misclassified as Nevus", "true": "mel", "pred": "nv", "prefix": "error_mel_to_nv"},
        {"desc": "High-Confidence Error: BCC misclassified as Nevus", "true": "bcc", "pred": "nv", "prefix": "error_bcc_to_nv"},
        {"desc": "High-Confidence Error: AKIEC misclassified as Nevus", "true": "akiec", "pred": "nv", "prefix": "error_akiec_to_nv"},
        {"desc": "High-Confidence Error: BKL misclassified as Nevus", "true": "bkl", "pred": "nv", "prefix": "error_bkl_to_nv"},
    ]

    c2i = {c: i for i, c in enumerate(CLASS_NAMES)}
    generated_artifacts = []

    # Snapshot weights before Grad-CAM
    weights_before = snapshot_model_weights(model)

    for case in cases_to_find:
        sub = predictions_df[(predictions_df["true_class"] == case["true"]) & (predictions_df["predicted_class"] == case["pred"])]
        if len(sub) == 0:
            logger.warning(f"Could not find case matching {case['desc']}")
            continue

        # Pick highest confidence sample
        best_row = sub.sort_values(by="predicted_confidence", ascending=False).iloc[0]
        img_id = best_row["image_id"]

        img_path = find_image_path(img_id, raw_image_dir)
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_224 = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

        # Transform for model
        input_tensor = transform(image=rgb_224)["image"].unsqueeze(0).to(device)

        # Target class: predicted class
        target_idx = c2i[best_row["predicted_class"]]
        cam, pred_c, pred_conf, _ = gradcam.generate_cam(input_tensor, target_class=target_idx)
        overlay = create_gradcam_overlay(rgb_224, cam)

        out_file = output_dir / f"{case['prefix']}_{img_id}.png"
        meta = {
            "image_id": img_id,
            "true_class": case["true"],
            "predicted_class": case["pred"],
            "confidence": float(best_row["predicted_confidence"]),
            "target_class": case["pred"],
            "description": case["desc"],
        }
        plot_gradcam_triplet(rgb_224, cam, overlay, meta, out_file)

        generated_artifacts.append({
            "case_description": case["desc"],
            "image_id": img_id,
            "true_class": case["true"],
            "predicted_class": case["pred"],
            "confidence": float(best_row["predicted_confidence"]),
            "file_path": str(out_file),
        })

    gradcam.remove_hooks()

    # Assert weights bitwise unchanged
    assert verify_model_weights_unchanged(weights_before, model), "Model weights altered during Grad-CAM failure analysis!"

    return generated_artifacts
