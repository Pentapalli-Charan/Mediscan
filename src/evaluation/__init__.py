"""
MediScan — Model Evaluation & Diagnostic Analysis Package (Day 9)
"""

from src.evaluation.inference import run_test_inference, measure_single_image_latency
from src.evaluation.metrics import (
    compute_overall_metrics,
    compute_per_class_report,
    compute_roc_auc_metrics,
    compute_pr_auc_metrics,
    compute_top_k_accuracy,
    compute_test_loss,
)
from src.evaluation.visualization import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_misclassifications_grid,
)
from src.evaluation.error_analysis import (
    analyze_malignant_errors,
    analyze_prediction_confidence,
    extract_representative_errors,
)

__all__ = [
    "run_test_inference",
    "measure_single_image_latency",
    "compute_overall_metrics",
    "compute_per_class_report",
    "compute_roc_auc_metrics",
    "compute_pr_auc_metrics",
    "compute_top_k_accuracy",
    "compute_test_loss",
    "plot_confusion_matrices",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_misclassifications_grid",
    "analyze_malignant_errors",
    "analyze_prediction_confidence",
    "extract_representative_errors",
]
