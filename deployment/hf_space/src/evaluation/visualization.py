"""
MediScan — Presentation-Quality Evaluation Visualizations (Day 9)

Generates publication-ready diagnostic figures:
- 7x7 Raw-Count Confusion Matrix Heatmap
- 7x7 Row-Normalized Confusion Matrix Heatmap
- Multi-class One-vs-Rest ROC Curves with AUC Scores
- One-vs-Rest Precision-Recall Curves with Average Precision Scores
- Representative Misclassifications Image Grid with Diagnostic Annotations
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_dir: Union[str, Path],
) -> Tuple[Path, Path, np.ndarray]:
    """
    Generate and save raw-count and row-normalized 7x7 confusion matrix heatmaps.

    Args:
        y_true: True class indices [N].
        y_pred: Predicted class indices [N].
        class_names: Ordered list of diagnostic class labels.
        output_dir: Directory where figures will be saved.

    Returns:
        tuple: (raw_matrix_path, normalized_matrix_path, raw_confusion_matrix_array)
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cm_raw = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(divide="ignore", invalid="ignore"):
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm_raw / row_sums, 0.0)

    # 1. Raw Confusion Matrix
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    sns.heatmap(
        cm_raw,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Sample Count"},
        linewidths=0.75,
        linecolor="#e0e0e0",
        ax=ax,
    )
    ax.set_title(
        "MediScan Baseline (EfficientNet-B0) — Raw Confusion Matrix\nHeld-Out Test Set (N = 1,512)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted Diagnostic Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Diagnostic Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    raw_path = out_dir / "day9_confusion_matrix.png"
    plt.savefig(raw_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Normalized Confusion Matrix
    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Proportion (Recall per Class)"},
        linewidths=0.75,
        linecolor="#e0e0e0",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )
    ax.set_title(
        "MediScan Baseline (EfficientNet-B0) — Normalized Confusion Matrix\nHeld-Out Test Set (Row-Normalized / Sensitivity)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Predicted Diagnostic Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Diagnostic Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.xticks(rotation=0, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    norm_path = out_dir / "day9_confusion_matrix_normalized.png"
    plt.savefig(norm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return raw_path, norm_path, cm_raw


def plot_roc_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> Path:
    """
    Generate One-vs-Rest ROC curves for all classes with individual and macro AUC annotations.

    Args:
        y_true: True class indices [N].
        probs: Softmax probabilities [N, num_classes].
        class_names: Ordered list of diagnostic classes.
        output_path: Destination filepath.

    Returns:
        Path: Path to saved plot.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    y_onehot = np.zeros((len(y_true), num_classes), dtype=int)
    for i, c in enumerate(y_true):
        y_onehot[i, c] = 1

    palette = sns.color_palette("tab10", n_colors=num_classes)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)

    macro_auc_list = []
    for c_idx, c_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_onehot[:, c_idx], probs[:, c_idx])
        roc_auc_val = auc(fpr, tpr)
        macro_auc_list.append(roc_auc_val)
        ax.plot(
            fpr,
            tpr,
            label=f"{c_name} (AUC = {roc_auc_val:.3f})",
            color=palette[c_idx],
            linewidth=2.0,
        )

    macro_auc = float(np.mean(macro_auc_list))

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.7, label="Chance Level (AUC = 0.500)")

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"MediScan Baseline — Multi-Class ROC Curves (One-vs-Rest)\nMacro-Averaged ROC-AUC = {macro_auc:.3f} | Test Set N = 1,512",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.5)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_precision_recall_curves(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> Path:
    """
    Generate One-vs-Rest Precision-Recall curves for all classes with Average Precision scores.

    Args:
        y_true: True class indices [N].
        probs: Softmax probabilities [N, num_classes].
        class_names: Ordered list of diagnostic classes.
        output_path: Destination filepath.

    Returns:
        Path: Path to saved plot.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    y_onehot = np.zeros((len(y_true), num_classes), dtype=int)
    for i, c in enumerate(y_true):
        y_onehot[i, c] = 1

    palette = sns.color_palette("tab10", n_colors=num_classes)

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=300)

    macro_ap_list = []
    for c_idx, c_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_onehot[:, c_idx], probs[:, c_idx])
        ap_val = auc(recall, precision)
        macro_ap_list.append(ap_val)
        ax.plot(
            recall,
            precision,
            label=f"{c_name} (AP = {ap_val:.3f})",
            color=palette[c_idx],
            linewidth=2.0,
        )

    macro_ap = float(np.mean(macro_ap_list))

    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        f"MediScan Baseline — Precision-Recall Curves (One-vs-Rest)\nMacro Average Precision = {macro_ap:.3f} | Supplementary Metric",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=9.5)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_misclassifications_grid(
    misclassified_df: pd.DataFrame,
    image_base_dir: Union[str, Path],
    output_path: Union[str, Path],
    n_samples: int = 16,
) -> Path:
    """
    Generate a formatted grid of representative misclassifications.

    Args:
        misclassified_df: DataFrame containing misclassified samples with image_id,
            true_class, predicted_class, predicted_confidence.
        image_base_dir: Directory containing HAM10000 image folders.
        output_path: Destination filepath.
        n_samples: Number of samples to visualize (default: 16 in 4x4 grid).

    Returns:
        Path: Path to saved grid figure.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_dir = Path(image_base_dir)
    image_dirs = [d for d in base_dir.iterdir() if d.is_dir() and "HAM10000_images" in d.name]

    def find_img(img_id: str) -> Optional[Path]:
        for d in image_dirs:
            for ext in (".jpg", ".jpeg", ".png"):
                p = d / f"{img_id}{ext}"
                if p.exists():
                    return p
        return None

    selected = misclassified_df.head(n_samples)
    actual_n = len(selected)
    if actual_n == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.text(0.5, 0.5, "No misclassifications to display", ha="center", va="center")
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    cols = 4
    rows = int(np.ceil(actual_n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]
        if i < actual_n:
            row = selected.iloc[i]
            img_id = row["image_id"]
            img_file = find_img(img_id)

            if img_file and img_file.exists():
                img = Image.open(img_file).convert("RGB")
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "Image Not Found", ha="center", va="center")

            title_color = "#b71c1c" if row["true_class"] in ["mel", "bcc", "akiec"] else "#333333"
            ax.set_title(
                f"ID: {img_id}\nTrue: {row['true_class']} → Pred: {row['predicted_class']}\nConf: {row['predicted_confidence']:.1%}",
                fontsize=9,
                fontweight="bold",
                color=title_color,
                pad=6,
            )
        ax.axis("off")

    plt.suptitle(
        f"MediScan Baseline (EfficientNet-B0) — Representative Misclassified Test Samples (N = {actual_n})\n"
        "(Red titles highlight critical malignant/premalignant true categories misclassified)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path
