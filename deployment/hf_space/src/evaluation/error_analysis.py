"""
MediScan — Error & Clinical Diagnostic Analysis (Day 9)

Provides targeted error decomposition for medical image classification:
- Critical malignant/premalignant error tracking (mel, bcc, akiec)
- Specific count of malignant-to-benign nevi confusion (mel->nv, bcc->nv, akiec->nv)
- Diagnostic confusion pair frequency ranking
- Model confidence dynamics (correct vs. incorrect distributions, high-confidence errors)
- Representative sample extraction for visualization
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def analyze_malignant_errors(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze error profiles on malignant and premalignant categories (mel, bcc, akiec).

    Args:
        predictions_df: DataFrame containing per-sample predictions.

    Returns:
        dict: Detailed breakdown of false negatives and malignant->nv confusion.
    """
    malignant_classes = ["mel", "bcc", "akiec"]
    results: Dict[str, Any] = {}

    for cls in malignant_classes:
        subset = predictions_df[predictions_df["true_class"] == cls]
        total = len(subset)
        correct = int((subset["predicted_class"] == cls).sum())
        false_negatives = total - correct
        mis_as_nv = int((subset["predicted_class"] == "nv").sum())
        recall = round(correct / total, 4) if total > 0 else 0.0
        mis_nv_pct = round(mis_as_nv / total * 100, 2) if total > 0 else 0.0

        # Detailed confusion distribution
        confusion_counts = subset[subset["predicted_class"] != cls]["predicted_class"].value_counts().to_dict()

        results[cls] = {
            "total_support": total,
            "correct": correct,
            "false_negatives": false_negatives,
            "recall": recall,
            "misclassified_as_nv": mis_as_nv,
            "pct_misclassified_as_nv": mis_nv_pct,
            "misclassification_breakdown": confusion_counts,
        }

    # Combined malignant summary
    all_mal = predictions_df[predictions_df["true_class"].isin(malignant_classes)]
    total_mal = len(all_mal)
    correct_mal = int((all_mal["predicted_class"] == all_mal["true_class"]).sum())
    fn_mal = total_mal - correct_mal
    mal_as_nv = int((all_mal["predicted_class"] == "nv").sum())

    results["summary"] = {
        "total_malignant_support": total_mal,
        "correct_malignant": correct_mal,
        "total_false_negatives": fn_mal,
        "malignant_sensitivity": round(correct_mal / total_mal, 4) if total_mal > 0 else 0.0,
        "total_malignant_misclassified_as_nv": mal_as_nv,
        "pct_malignant_misclassified_as_nv": round(mal_as_nv / total_mal * 100, 2) if total_mal > 0 else 0.0,
    }

    return results


def analyze_prediction_confidence(predictions_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute confidence statistics for correct vs incorrect predictions,
    and identify high-confidence incorrect predictions (confidence >= 0.80).

    Args:
        predictions_df: DataFrame containing per-sample predictions.

    Returns:
        dict: Confidence statistics and high-confidence error details.
    """
    correct_mask = predictions_df["true_class"] == predictions_df["predicted_class"]
    correct_conf = predictions_df.loc[correct_mask, "predicted_confidence"].values
    incorrect_conf = predictions_df.loc[~correct_mask, "predicted_confidence"].values

    stats = {
        "correct": {
            "count": int(len(correct_conf)),
            "mean_confidence": round(float(np.mean(correct_conf)), 4) if len(correct_conf) > 0 else 0.0,
            "median_confidence": round(float(np.median(correct_conf)), 4) if len(correct_conf) > 0 else 0.0,
            "std_confidence": round(float(np.std(correct_conf)), 4) if len(correct_conf) > 0 else 0.0,
            "min_confidence": round(float(np.min(correct_conf)), 4) if len(correct_conf) > 0 else 0.0,
            "max_confidence": round(float(np.max(correct_conf)), 4) if len(correct_conf) > 0 else 0.0,
        },
        "incorrect": {
            "count": int(len(incorrect_conf)),
            "mean_confidence": round(float(np.mean(incorrect_conf)), 4) if len(incorrect_conf) > 0 else 0.0,
            "median_confidence": round(float(np.median(incorrect_conf)), 4) if len(incorrect_conf) > 0 else 0.0,
            "std_confidence": round(float(np.std(incorrect_conf)), 4) if len(incorrect_conf) > 0 else 0.0,
            "min_confidence": round(float(np.min(incorrect_conf)), 4) if len(incorrect_conf) > 0 else 0.0,
            "max_confidence": round(float(np.max(incorrect_conf)), 4) if len(incorrect_conf) > 0 else 0.0,
        },
    }

    # High-confidence incorrect predictions (conf >= 0.80)
    high_conf_errors_df = predictions_df[(~correct_mask) & (predictions_df["predicted_confidence"] >= 0.80)]
    stats["high_confidence_errors"] = {
        "threshold": 0.80,
        "count": int(len(high_conf_errors_df)),
        "pct_of_all_errors": round(len(high_conf_errors_df) / len(incorrect_conf) * 100, 2) if len(incorrect_conf) > 0 else 0.0,
        "most_frequent_classes": high_conf_errors_df["true_class"].value_counts().to_dict(),
    }

    return stats


def get_confusion_pair_ranking(predictions_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Rank all confusion pairs (True -> Predicted) by error frequency.

    Args:
        predictions_df: DataFrame with predictions.

    Returns:
        list of dicts: Ranked confusion pairs.
    """
    errors_df = predictions_df[predictions_df["true_class"] != predictions_df["predicted_class"]]
    pair_counts = (
        errors_df.groupby(["true_class", "predicted_class"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )

    pairs = []
    total_errors = len(errors_df)
    for _, row in pair_counts.iterrows():
        pairs.append({
            "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "count": int(row["count"]),
            "pct_of_total_errors": round(row["count"] / total_errors * 100, 2) if total_errors > 0 else 0.0,
        })

    return pairs


def extract_representative_errors(
    predictions_df: pd.DataFrame,
    n_samples: int = 16,
) -> pd.DataFrame:
    """
    Curate an informative subset of representative misclassifications.

    Prioritizes:
    1. Critical malignant-to-benign errors (mel -> nv, bcc -> nv, akiec -> nv)
    2. High-confidence errors (confidence >= 0.80)
    3. Minority class errors (df, vasc)
    4. Frequent benign confusions (bkl -> nv, nv -> bkl)

    Args:
        predictions_df: DataFrame containing predictions.
        n_samples: Total number of samples to select (default: 16).

    Returns:
        pd.DataFrame: Curated sample DataFrame.
    """
    errors = predictions_df[predictions_df["true_class"] != predictions_df["predicted_class"]].copy()

    # Buckets
    mel_to_nv = errors[(errors["true_class"] == "mel") & (errors["predicted_class"] == "nv")]
    bcc_to_nv = errors[(errors["true_class"] == "bcc") & (errors["predicted_class"] == "nv")]
    akiec_to_nv = errors[(errors["true_class"] == "akiec") & (errors["predicted_class"] == "nv")]
    high_conf = errors[errors["predicted_confidence"] >= 0.80]
    minority_errors = errors[errors["true_class"].isin(["df", "vasc"])]
    bkl_confusions = errors[(errors["true_class"] == "bkl") & (errors["predicted_class"] == "nv")]

    selected_indices: List[int] = []

    def add_from_df(sub_df: pd.DataFrame, limit: int):
        nonlocal selected_indices
        available = sub_df[~sub_df.index.isin(selected_indices)].sort_values(
            by="predicted_confidence", ascending=False
        )
        for idx in available.head(limit).index:
            if len(selected_indices) < n_samples:
                selected_indices.append(idx)

    # Balance across categories
    add_from_df(mel_to_nv, 4)
    add_from_df(bcc_to_nv, 3)
    add_from_df(akiec_to_nv, 2)
    add_from_df(high_conf, 3)
    add_from_df(minority_errors, 2)
    add_from_df(bkl_confusions, 2)

    # Fill remaining from all errors if needed
    if len(selected_indices) < n_samples:
        add_from_df(errors, n_samples - len(selected_indices))

    return errors.loc[selected_indices]
