"""
MediScan — Day 9: Comprehensive Baseline Model Evaluation Runner

Executes the formal, unblinded held-out test evaluation of the baseline
EfficientNet-B0 transfer learning model on HAM10000 test set (N = 1,512).

Workflow:
1. Validates test split integrity, lesion isolation, and image availability.
2. Loads the fixed checkpoint: models/checkpoints/efficientnet_b0_best.pth.
3. Benchmarks single-image latency and executes batch inference.
4. Records all predictions and 7-class probability distributions.
5. Computes overall and per-class diagnostic metrics, Top-1 and Top-2 accuracy.
6. Computes multi-class ROC-AUC and PR-AUC.
7. Conducts malignant/premalignant error decomposition and confidence profiling.
8. Generates publication-ready figures and structured artifacts.
"""

import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import (
    HAM10000Dataset,
    get_augmentation_transform,
    load_config,
)
from src.evaluation.error_analysis import (
    analyze_malignant_errors,
    analyze_prediction_confidence,
    extract_representative_errors,
    get_confusion_pair_ranking,
)
from src.evaluation.inference import (
    measure_single_image_latency,
    run_test_inference,
)
from src.evaluation.metrics import (
    compute_overall_metrics,
    compute_per_class_report,
    compute_pr_auc_metrics,
    compute_roc_auc_metrics,
    compute_test_loss,
)
from src.evaluation.visualization import (
    plot_confusion_matrices,
    plot_misclassifications_grid,
    plot_precision_recall_curves,
    plot_roc_curves,
)
from src.models import CLASS_NAMES, NUM_CLASSES, build_model
from src.training.trainer import load_training_checkpoint
from src.utils.device import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_day9")


def verify_test_integrity(splits_dir: Path, raw_dir: Path) -> Dict[str, Any]:
    """Verify test split file integrity, sample count, and lesion isolation."""
    logger.info("Verifying test split integrity and isolation...")
    test_path = splits_dir / "split_test.csv"
    train_path = splits_dir / "split_train.csv"
    val_path = splits_dir / "split_val.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Test split CSV not found: {test_path}")

    md5_test = hashlib.md5(test_path.read_bytes()).hexdigest()
    df_test = pd.read_csv(test_path)
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    # Counts
    n_test = len(df_test)
    n_unique_imgs = df_test["image_id"].nunique()
    n_unique_lesions = df_test["lesion_id"].nunique()

    if n_test != 1512:
        raise ValueError(f"Expected 1,512 test samples, got {n_test}")
    if n_unique_imgs != 1512:
        raise ValueError(f"Duplicate images found in test set: {n_test} rows vs {n_unique_imgs} unique")

    # Overlaps
    lesion_overlap_train = len(set(df_train["lesion_id"]) & set(df_test["lesion_id"]))
    lesion_overlap_val = len(set(df_val["lesion_id"]) & set(df_test["lesion_id"]))
    img_overlap_train = len(set(df_train["image_id"]) & set(df_test["image_id"]))
    img_overlap_val = len(set(df_val["image_id"]) & set(df_test["image_id"]))

    if lesion_overlap_train != 0 or lesion_overlap_val != 0:
        raise ValueError(f"Data leakage detected! Lesion overlap: train={lesion_overlap_train}, val={lesion_overlap_val}")
    if img_overlap_train != 0 or img_overlap_val != 0:
        raise ValueError(f"Image overlap detected! Train={img_overlap_train}, Val={img_overlap_val}")

    # Image files existence
    image_dirs = [d for d in raw_dir.iterdir() if d.is_dir() and "HAM10000_images" in d.name]
    missing_imgs = []
    for img_id in df_test["image_id"]:
        found = any((d / f"{img_id}.jpg").exists() for d in image_dirs)
        if not found:
            missing_imgs.append(img_id)

    if missing_imgs:
        raise FileNotFoundError(f"Missing {len(missing_imgs)} test images on disk!")

    logger.info(f"Test split verified: {n_test} samples, 0 missing, MD5={md5_test}, 0 overlap with train/val.")
    return {
        "md5_checksum": md5_test,
        "sample_count": n_test,
        "unique_images": n_unique_imgs,
        "unique_lesions": n_unique_lesions,
        "lesion_overlap_train": lesion_overlap_train,
        "lesion_overlap_val": lesion_overlap_val,
        "image_overlap_train": img_overlap_train,
        "image_overlap_val": img_overlap_val,
    }


def main():
    logger.info("Starting MediScan Day 9: Comprehensive Baseline Model Evaluation...")

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(str(config_path))

    data_raw_dir = PROJECT_ROOT / config.get("paths", {}).get("data_raw", "data/raw")
    reports_dir = PROJECT_ROOT / config.get("paths", {}).get("reports", "reports")
    reports_data_dir = reports_dir / "data"
    figures_dir = reports_dir / "figures"
    errors_figures_dir = figures_dir / "day9" / "errors"

    reports_data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    errors_figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Integrity Check
    integrity_report = verify_test_integrity(reports_data_dir, data_raw_dir)

    # 2. Compute Device
    device = get_device()
    logger.info(f"Target compute device: {device}")

    # 3. Test Dataset & DataLoader
    test_csv = reports_data_dir / "split_test.csv"
    test_transform = get_augmentation_transform(config, mode="test")
    test_dataset = HAM10000Dataset(
        split_csv_path=str(test_csv),
        image_base_dir=str(data_raw_dir),
        transform=test_transform,
        include_metadata=True,
    )

    batch_size = config.get("dataloader", {}).get("batch_size", 32)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Safe and clean on Windows
        pin_memory=False,
        drop_last=False,
    )
    logger.info(f"Constructed test DataLoader: {len(test_dataset)} samples, batch_size={batch_size}")

    # 4. Load Baseline Checkpoint
    checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Baseline checkpoint not found at: {checkpoint_path}")

    model = build_model(config, architecture="efficientnet_b0", freeze_backbone=True)
    ckpt_meta = load_training_checkpoint(checkpoint_path, model, device=str(device))
    model = model.to(device)
    model.eval()

    logger.info(
        f"Model initialized and loaded from checkpoint (Epoch: {ckpt_meta.get('epoch')}, "
        f"Val Loss: {ckpt_meta.get('val_loss'):.4f})"
    )

    # 5. Measure Single-Image Latency
    sample_img, _, _ = test_dataset[0]
    latency_stats = measure_single_image_latency(model, sample_img, device, num_warmup=5, num_runs=30)
    logger.info(
        f"Single-image latency: Mean={latency_stats['single_image_mean_latency_ms']:.2f}ms "
        f"({latency_stats['single_image_mean_latency_sec']:.4f}s), "
        f"Median={latency_stats['single_image_median_latency_ms']:.2f}ms"
    )

    # 6. Full Test Inference
    logger.info("Executing full test inference over 1,512 images...")
    predictions_df, all_logits, all_labels, timing_stats = run_test_inference(
        model=model,
        dataloader=test_loader,
        class_names=CLASS_NAMES,
        device=device,
    )
    logger.info(
        f"Inference completed in {timing_stats['total_inference_duration_sec']}s "
        f"({timing_stats['throughput_images_per_sec']} imgs/sec)"
    )

    # Save Predictions CSV
    pred_csv_path = reports_data_dir / "day9_test_predictions.csv"
    predictions_df.to_csv(pred_csv_path, index=False)
    logger.info(f"Saved full test predictions to: {pred_csv_path}")

    # 7. Overall Metrics & Per-Class Breakdown
    y_true = all_labels
    class_map = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    y_pred = predictions_df["predicted_class"].map(class_map).values
    probs = predictions_df[[f"prob_{c}" for c in CLASS_NAMES]].to_numpy()

    overall_metrics = compute_overall_metrics(y_true, y_pred, probs, logits=all_logits)
    logger.info(f"Top-1 Accuracy: {overall_metrics['top1_accuracy'] * 100:.2f}%")
    logger.info(f"Top-2 Accuracy: {overall_metrics['top2_accuracy'] * 100:.2f}%")
    logger.info(f"Macro F1-score: {overall_metrics['macro_f1']:.4f}")
    logger.info(f"Weighted F1-score: {overall_metrics['weighted_f1']:.4f}")
    logger.info(f"Test Loss: {overall_metrics.get('test_loss', 0.0):.4f}")

    per_class_df = compute_per_class_report(y_true, y_pred, CLASS_NAMES)
    class_report_csv_path = reports_data_dir / "day9_classification_report.csv"
    per_class_df.to_csv(class_report_csv_path, index=False)
    logger.info(f"Saved classification report to: {class_report_csv_path}")

    # 8. ROC-AUC & PR-AUC
    roc_metrics = compute_roc_auc_metrics(y_true, probs, CLASS_NAMES)
    pr_metrics = compute_pr_auc_metrics(y_true, probs, CLASS_NAMES)
    logger.info(f"Macro ROC-AUC: {roc_metrics['macro_roc_auc']:.4f}")
    logger.info(f"Weighted ROC-AUC: {roc_metrics['weighted_roc_auc']:.4f}")
    logger.info(f"Macro PR-AUC (Average Precision): {pr_metrics['macro_pr_auc']:.4f}")

    # 9. Error Analysis
    malignant_analysis = analyze_malignant_errors(predictions_df)
    confidence_analysis = analyze_prediction_confidence(predictions_df)
    confusion_pairs = get_confusion_pair_ranking(predictions_df)
    representative_errors_df = extract_representative_errors(predictions_df, n_samples=16)

    # 10. Generate Visualizations
    logger.info("Generating evaluation visualizations...")
    raw_cm_path, norm_cm_path, cm_raw = plot_confusion_matrices(y_true, y_pred, CLASS_NAMES, figures_dir)
    roc_plot_path = plot_roc_curves(y_true, probs, CLASS_NAMES, figures_dir / "day9_roc_curves.png")
    pr_plot_path = plot_precision_recall_curves(y_true, probs, CLASS_NAMES, figures_dir / "day9_precision_recall_curves.png")
    errors_grid_path = plot_misclassifications_grid(
        representative_errors_df,
        data_raw_dir,
        errors_figures_dir / "day9_misclassifications_grid.png",
        n_samples=16,
    )

    logger.info(f"Saved Confusion Matrix (Raw): {raw_cm_path}")
    logger.info(f"Saved Confusion Matrix (Norm): {norm_cm_path}")
    logger.info(f"Saved ROC Curves: {roc_plot_path}")
    logger.info(f"Saved PR Curves: {pr_plot_path}")
    logger.info(f"Saved Error Grid: {errors_grid_path}")

    # 11. Compile Complete Structured Metrics JSON
    full_metrics = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {
            "architecture": "efficientnet_b0",
            "weights": "IMAGENET1K_V1",
            "backbone_frozen": True,
            "classifier_head": "Dropout(p=0.2) -> Linear(1280, 7)",
            "checkpoint_loaded": str(checkpoint_path.name),
            "checkpoint_best_val_loss": ckpt_meta.get("val_loss"),
            "checkpoint_best_val_accuracy": ckpt_meta.get("val_accuracy"),
            "checkpoint_epoch": ckpt_meta.get("epoch"),
        },
        "test_integrity": integrity_report,
        "timing_and_latency": {
            **timing_stats,
            **latency_stats,
        },
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_df.to_dict(orient="records"),
        "roc_auc": roc_metrics,
        "precision_recall_auc": pr_metrics,
        "confusion_matrix_raw": cm_raw.tolist(),
        "malignant_error_analysis": malignant_analysis,
        "confidence_analysis": confidence_analysis,
        "top_confusion_pairs": confusion_pairs[:10],
    }

    metrics_json_path = reports_data_dir / "day9_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    logger.info(f"Saved full metrics JSON to: {metrics_json_path}")

    print("\n" + "=" * 70)
    print("           MEDISCAN DAY 9: EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Test Samples:           {timing_stats['total_samples']}")
    print(f"Top-1 Accuracy:         {overall_metrics['top1_accuracy'] * 100:.2f}%")
    print(f"Top-2 Accuracy:         {overall_metrics['top2_accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score:         {overall_metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score:      {overall_metrics['weighted_f1']:.4f}")
    print(f"Macro ROC-AUC:          {roc_metrics['macro_roc_auc']:.4f}")
    print(f"Weighted ROC-AUC:       {roc_metrics['weighted_roc_auc']:.4f}")
    print(f"Macro PR-AUC (AP):      {pr_metrics['macro_pr_auc']:.4f}")
    print(f"Test Loss (CE):         {overall_metrics.get('test_loss'):.4f}")
    print(f"Total Inference Time:   {timing_stats['total_inference_duration_sec']:.2f}s")
    print(f"Throughput:             {timing_stats['throughput_images_per_sec']:.1f} imgs/s")
    print(f"Single-Image Latency:   {latency_stats['single_image_mean_latency_ms']:.2f}ms (Mean)")
    print("-" * 70)
    print("Per-Class Diagnostic Breakdown:")
    for _, row in per_class_df.iterrows():
        c_name = row["class"]
        auc_val = roc_metrics["per_class_roc_auc"].get(c_name, "N/A")
        auc_str = f"{auc_val:.3f}" if isinstance(auc_val, (int, float)) else str(auc_val)
        print(
            f"  {c_name:6s} | P: {row['precision']:.3f} | R: {row['recall']:.3f} | "
            f"F1: {row['f1_score']:.3f} | AUC: {auc_str:>5s} | N: {row['support']}"
        )
    print("-" * 70)
    print("Malignant / Premalignant Summary:")
    print(
        f"  mel   (Melanoma):            Recall={malignant_analysis['mel']['recall']:.1%}, "
        f"Misclassified as nv={malignant_analysis['mel']['misclassified_as_nv']} / {malignant_analysis['mel']['total_support']} "
        f"({malignant_analysis['mel']['pct_misclassified_as_nv']}%)"
    )
    print(
        f"  bcc   (Basal Cell Carcinoma):Recall={malignant_analysis['bcc']['recall']:.1%}, "
        f"Misclassified as nv={malignant_analysis['bcc']['misclassified_as_nv']} / {malignant_analysis['bcc']['total_support']} "
        f"({malignant_analysis['bcc']['pct_misclassified_as_nv']}%)"
    )
    print(
        f"  akiec (Actinic Keratoses):   Recall={malignant_analysis['akiec']['recall']:.1%}, "
        f"Misclassified as nv={malignant_analysis['akiec']['misclassified_as_nv']} / {malignant_analysis['akiec']['total_support']} "
        f"({malignant_analysis['akiec']['pct_misclassified_as_nv']}%)"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
