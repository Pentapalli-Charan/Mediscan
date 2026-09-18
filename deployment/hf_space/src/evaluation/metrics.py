"""
MediScan — Quantitative Evaluation Metrics (Day 9)

Computes comprehensive diagnostic metrics for multi-class classification:
- Top-1 and Top-2 accuracy
- Macro & Weighted Precision, Recall, F1-score
- Per-class classification breakdown (Precision, Recall, F1, Support)
- Multi-class One-vs-Rest ROC-AUC (per-class, macro, weighted)
- One-vs-Rest Average Precision / PR-AUC (per-class, macro)
- Held-out test CrossEntropyLoss
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch
import torch.nn as nn


def compute_top_k_accuracy(
    y_true: np.ndarray,
    probs_or_logits: np.ndarray,
    k: int = 2,
) -> float:
    """
    Compute genuine top-k accuracy across samples.

    A sample is considered correct if the true class index is among the k classes
    with the highest predicted probability or logit value.

    Args:
        y_true: True class indices [N].
        probs_or_logits: Array of shape [N, num_classes].
        k: Rank threshold (default: 2).

    Returns:
        float: Top-k accuracy as a float in [0.0, 1.0].
    """
    if len(y_true) == 0:
        return 0.0

    # Get indices of the top-k predictions along class dimension
    # np.argpartition or sort
    top_k_indices = np.argsort(probs_or_logits, axis=1)[:, -k:]
    correct = np.any(top_k_indices == y_true[:, None], axis=1)
    return float(np.mean(correct))


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    logits: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate summary diagnostic metrics across all test samples.

    Args:
        y_true: True class indices [N].
        y_pred: Predicted class indices [N].
        probs: Softmax probabilities [N, num_classes].
        logits: Raw logits [N, num_classes] (optional, for test loss).

    Returns:
        dict: Top-1, Top-2, macro, and weighted precision, recall, and F1 scores.
    """
    top1_acc = float(accuracy_score(y_true, y_pred))
    top2_acc = compute_top_k_accuracy(y_true, probs, k=2)

    # Macro metrics (unweighted average across all classes)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Weighted metrics (weighted by true class support)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "top1_accuracy": round(top1_acc, 6),
        "top2_accuracy": round(top2_acc, 6),
        "macro_precision": round(float(macro_p), 6),
        "macro_recall": round(float(macro_r), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_p), 6),
        "weighted_recall": round(float(weighted_r), 6),
        "weighted_f1": round(float(weighted_f1), 6),
    }

    if logits is not None:
        metrics["test_loss"] = compute_test_loss(logits, y_true)

    return metrics


def compute_per_class_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> pd.DataFrame:
    """
    Generate per-class precision, recall, F1, and support table.

    Args:
        y_true: True class indices [N].
        y_pred: Predicted class indices [N].
        class_names: List of ordered class names.

    Returns:
        pd.DataFrame: Table indexed by class name with metrics columns.
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    records = []
    for idx, c_name in enumerate(class_names):
        records.append({
            "class": c_name,
            "precision": round(float(precision[idx]), 6),
            "recall": round(float(recall[idx]), 6),
            "f1_score": round(float(f1[idx]), 6),
            "support": int(support[idx]),
        })

    return pd.DataFrame(records)


def compute_roc_auc_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute One-vs-Rest ROC-AUC scores for each individual class, macro-average, and weighted-average.

    Args:
        y_true: True class indices [N].
        probs: Softmax probabilities [N, num_classes].
        class_names: List of ordered class names.

    Returns:
        dict: Per-class ROC-AUC values, macro ROC-AUC, and weighted ROC-AUC.
    """
    num_classes = len(class_names)
    per_class_auc: Dict[str, Union[float, str]] = {}
    valid_aucs: List[float] = []
    supports: List[int] = []

    # One-hot encode true labels
    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=int)
    for i, c in enumerate(y_true):
        y_true_onehot[i, c] = 1

    for c_idx, c_name in enumerate(class_names):
        support_c = int(np.sum(y_true_onehot[:, c_idx]))
        supports.append(support_c)

        # Check if positive samples exist and negative samples exist
        if support_c == 0 or support_c == len(y_true):
            per_class_auc[c_name] = "NOT AVAILABLE — single class present in split"
        else:
            try:
                auc_val = float(roc_auc_score(y_true_onehot[:, c_idx], probs[:, c_idx]))
                per_class_auc[c_name] = round(auc_val, 6)
                valid_aucs.append(auc_val)
            except ValueError as e:
                per_class_auc[c_name] = f"NOT AVAILABLE — {str(e)}"

    if len(valid_aucs) == num_classes:
        macro_auc = float(np.mean(valid_aucs))
        weighted_auc = float(
            np.average(valid_aucs, weights=supports)
        )
    else:
        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else 0.0
        weighted_auc = 0.0

    return {
        "per_class_roc_auc": per_class_auc,
        "macro_roc_auc": round(macro_auc, 6),
        "weighted_roc_auc": round(weighted_auc, 6),
    }


def compute_pr_auc_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute One-vs-Rest Average Precision (PR-AUC) scores for each class and macro-average.

    Args:
        y_true: True class indices [N].
        probs: Softmax probabilities [N, num_classes].
        class_names: List of ordered class names.

    Returns:
        dict: Per-class Average Precision and macro Average Precision.
    """
    num_classes = len(class_names)
    per_class_ap: Dict[str, Union[float, str]] = {}
    valid_aps: List[float] = []

    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=int)
    for i, c in enumerate(y_true):
        y_true_onehot[i, c] = 1

    for c_idx, c_name in enumerate(class_names):
        support_c = int(np.sum(y_true_onehot[:, c_idx]))
        if support_c == 0:
            per_class_ap[c_name] = "NOT AVAILABLE — zero positive samples in split"
        else:
            try:
                ap_val = float(average_precision_score(y_true_onehot[:, c_idx], probs[:, c_idx]))
                per_class_ap[c_name] = round(ap_val, 6)
                valid_aps.append(ap_val)
            except ValueError as e:
                per_class_ap[c_name] = f"NOT AVAILABLE — {str(e)}"

    macro_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

    return {
        "per_class_pr_auc": per_class_ap,
        "macro_pr_auc": round(macro_ap, 6),
    }


def compute_test_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate unweighted CrossEntropyLoss on the held-out test set.

    Args:
        logits: Raw unnormalized model outputs [N, num_classes].
        y_true: True class integer labels [N].

    Returns:
        float: Mean cross-entropy loss.
    """
    criterion = nn.CrossEntropyLoss()
    logits_t = torch.from_numpy(logits).float()
    labels_t = torch.from_numpy(y_true).long()
    loss = criterion(logits_t, labels_t)
    return round(float(loss.item()), 6)
