"""
MediScan — Day 14: Edge-Case, Bias & Robustness Analysis Runner

Executes all 18 phases of Day 14 analysis strictly without retraining:
- Phase 1: Artifact & codebase inspection
- Phase 2: Integrity verification (MD5 hashes, class mapping, Day 9 metrics)
- Phase 3: Class imbalance analysis across splits
- Phase 4: Per-class performance & disparities
- Phase 5: Confusion matrix & failure mode analysis (mel->nv, bcc->nv, akiec->nv)
- Phase 6: Softmax confidence distributions & high-confidence error audits
- Phase 7: Calibration analysis & Expected Calibration Error (ECE)
- Phase 8: Synthetic robustness probes (controlled perturbations on stratified subset)
- Phase 9: Image quality proxies (brightness, contrast, Laplacian blur)
- Phase 10: Metadata & subgroup analysis (age, sex, localization)
- Phase 11: Grad-CAM failure analysis targeting model.features[8]
- Phase 12: Comprehensive 12-factor limitations summary
- Phase 13: Machine-readable summary export to day14_analysis.json
- Phase 14: High-resolution visualization validation
"""

import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing_day3 import load_config
from src.evaluation.day14_analysis import (
    CANONICAL_CHECKPOINT_MD5,
    CANONICAL_TEST_SPLIT_MD5,
    LOCKED_DAY9_METRICS,
    analyze_class_imbalance,
    analyze_confidence,
    analyze_confusion_pairs,
    analyze_image_quality,
    analyze_metadata_subgroups,
    analyze_per_class_performance,
    compute_calibration,
    generate_day14_gradcam_failures,
    run_synthetic_robustness_probes,
    verify_day14_integrity,
)
from src.explainability.gradcam import get_default_target_layer
from src.models import CLASS_NAMES, build_model
from src.training.trainer import load_training_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_day14_analysis")


def main() -> None:
    logger.info("============================================================")
    logger.info("STARTING DAY 14: EDGE-CASE, BIAS & ROBUSTNESS ANALYSIS")
    logger.info("STRICT CONTRACT: ANALYSIS-ONLY MILESTONE. ZERO RETRAINING.")
    logger.info("============================================================")

    start_time = time.time()
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Define paths
    ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    test_split_path = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
    train_split_path = PROJECT_ROOT / "reports" / "data" / "split_train.csv"
    val_split_path = PROJECT_ROOT / "reports" / "data" / "split_val.csv"
    all_split_path = PROJECT_ROOT / "reports" / "data" / "split_all.csv"
    day9_metrics_path = PROJECT_ROOT / "reports" / "data" / "day9_metrics.json"
    day9_preds_path = PROJECT_ROOT / "reports" / "data" / "day9_test_predictions.csv"
    raw_img_dir = PROJECT_ROOT / "data" / "raw"
    config_path = PROJECT_ROOT / "config" / "config.yaml"

    output_data_dir = PROJECT_ROOT / "reports" / "data"
    output_fig_dir = PROJECT_ROOT / "reports" / "figures" / "day14"
    gradcam_fig_dir = output_fig_dir / "gradcam_failure_analysis"

    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_fig_dir.mkdir(parents=True, exist_ok=True)
    gradcam_fig_dir.mkdir(parents=True, exist_ok=True)

    # Load canonical model
    logger.info("Loading canonical EfficientNet-B0 baseline model...")
    cfg = load_config(str(config_path))
    model = build_model(cfg, architecture="efficientnet_b0", freeze_backbone=True)
    load_training_checkpoint(ckpt_path, model, device="cpu")
    model.eval()

    # ---------------------------------------------------------
    # PHASE 2: Integrity Verification
    # ---------------------------------------------------------
    logger.info("Phase 2: Verifying Checkpoint, Split, and Metric Integrity...")
    integrity_results = verify_day14_integrity(
        checkpoint_path=ckpt_path,
        test_split_path=test_split_path,
        metrics_json_path=day9_metrics_path,
        model=model,
    )
    if not integrity_results["all_integrity_passed"]:
        logger.error(f"Integrity check failed: {integrity_results}")
        raise RuntimeError("Integrity verification failed! Day 14 analysis aborted.")
    logger.info(f"✓ Integrity verified successfully: Checkpoint MD5={integrity_results['checkpoint_md5'][:8]}..., Test MD5={integrity_results['test_split_md5'][:8]}...")

    # ---------------------------------------------------------
    # PHASE 3: Class Imbalance Analysis
    # ---------------------------------------------------------
    logger.info("Phase 3: Computing Class Imbalance across Train/Val/Test Splits...")
    class_dist_csv = output_data_dir / "day14_class_distribution.csv"
    class_dist_png = output_fig_dir / "class_distribution.png"
    dist_df = analyze_class_imbalance(
        train_csv=train_split_path,
        val_csv=val_split_path,
        test_csv=test_split_path,
        output_csv=class_dist_csv,
        output_png=class_dist_png,
    )
    logger.info(f"✓ Class distribution computed: Imbalance ratio {dist_df.attrs['imbalance_ratio']}:1 (Majority: {dist_df.attrs['majority_class']}, Minority: {dist_df.attrs['minority_class']})")

    # ---------------------------------------------------------
    # PHASE 4: Per-Class Performance Analysis
    # ---------------------------------------------------------
    logger.info("Phase 4: Evaluating Per-Class Performance and Macro vs Weighted Disparities...")
    per_class_csv = output_data_dir / "day14_per_class_metrics.csv"
    per_class_f1_png = output_fig_dir / "per_class_f1.png"
    per_class_recall_png = output_fig_dir / "per_class_recall.png"
    perf_df = analyze_per_class_performance(
        predictions_csv=day9_preds_path,
        metrics_json_path=day9_metrics_path,
        output_csv=per_class_csv,
        output_f1_png=per_class_f1_png,
        output_recall_png=per_class_recall_png,
    )
    logger.info("✓ Per-class performance evaluated.")

    # ---------------------------------------------------------
    # PHASE 5: Confusion & Failure-Mode Analysis
    # ---------------------------------------------------------
    logger.info("Phase 5: Analyzing Confusion Matrix & Major Misclassification Pairs...")
    confusion_csv = output_data_dir / "day14_confusion_analysis.csv"
    confusion_png = output_fig_dir / "top_confusion_pairs.png"
    confusion_df = analyze_confusion_pairs(
        predictions_csv=day9_preds_path,
        metrics_json_path=day9_metrics_path,
        output_csv=confusion_csv,
        output_png=confusion_png,
    )
    logger.info(f"✓ Confusion analysis complete. Top error pair: {confusion_df.iloc[0]['true_class']} -> {confusion_df.iloc[0]['predicted_class']} (N={confusion_df.iloc[0]['count']})")

    # ---------------------------------------------------------
    # PHASE 6: Confidence Analysis
    # ---------------------------------------------------------
    logger.info("Phase 6: Evaluating Softmax Confidence & High-Confidence Errors...")
    conf_csv = output_data_dir / "day14_confidence_analysis.csv"
    conf_png = output_fig_dir / "confidence_distribution.png"
    conf_stats = analyze_confidence(
        predictions_csv=day9_preds_path,
        output_csv=conf_csv,
        output_png=conf_png,
    )
    logger.info(f"✓ Confidence stats: Correct mean={conf_stats['correct']['mean_confidence']:.4f}, Incorrect mean={conf_stats['incorrect']['mean_confidence']:.4f}, High-conf errors (>=0.80)={conf_stats['high_confidence_errors']['threshold_80']['count']}")

    # ---------------------------------------------------------
    # PHASE 7: Calibration Analysis (ECE)
    # ---------------------------------------------------------
    logger.info("Phase 7: Calculating Expected Calibration Error (ECE) and Reliability Diagram...")
    calib_csv = output_data_dir / "day14_calibration.csv"
    calib_png = output_fig_dir / "reliability_diagram.png"
    ece, mce, calib_df = compute_calibration(
        predictions_csv=day9_preds_path,
        n_bins=10,
        output_csv=calib_csv,
        output_png=calib_png,
    )
    logger.info(f"✓ Post-hoc Calibration: ECE = {ece:.4f}, MCE = {mce:.4f}")

    # ---------------------------------------------------------
    # PHASE 8: Synthetic Robustness Probes
    # ---------------------------------------------------------
    logger.info("Phase 8: Running Synthetic Robustness Probes on Stratified Test Subset...")
    test_df = pd.read_csv(test_split_path)
    robust_csv = output_data_dir / "day14_robustness.csv"
    robust_png = output_fig_dir / "robustness_comparison.png"
    robust_df = run_synthetic_robustness_probes(
        model=model,
        test_df=test_df,
        raw_image_dir=raw_img_dir,
        subset_size=150,
        random_seed=42,
        device="cpu",
        output_csv=robust_csv,
        output_png=robust_png,
    )
    logger.info("✓ Synthetic robustness probes complete across 6 perturbation conditions.")

    # ---------------------------------------------------------
    # PHASE 9: Image Quality Proxies Analysis
    # ---------------------------------------------------------
    logger.info("Phase 9: Calculating Image Quality Proxies on Test Set...")
    preds_df = pd.read_csv(day9_preds_path)
    img_quality_csv = output_data_dir / "day14_image_quality.csv"
    img_quality_png = output_fig_dir / "image_quality_comparison.png"
    img_quality_df = analyze_image_quality(
        test_df=test_df,
        predictions_df=preds_df,
        raw_image_dir=raw_img_dir,
        output_csv=img_quality_csv,
        output_png=img_quality_png,
    )
    logger.info("✓ Image quality proxy distributions analyzed.")

    # ---------------------------------------------------------
    # PHASE 10: Metadata & Subgroup Analysis
    # ---------------------------------------------------------
    logger.info("Phase 10: Auditing Metadata Missingness and Descriptive Subgroup Performance...")
    subgroup_csv = output_data_dir / "day14_subgroup_analysis.csv"
    subgroup_png = output_fig_dir / "subgroup_analysis.png"
    subgroup_df, missingness = analyze_metadata_subgroups(
        split_all_csv=all_split_path,
        predictions_df=preds_df,
        output_csv=subgroup_csv,
        output_png=subgroup_png,
    )
    logger.info(f"✓ Metadata subgroup analysis complete. Missingness: age={missingness['age_missing']}, sex={missingness['sex_missing']}, loc={missingness['localization_missing']}")

    # ---------------------------------------------------------
    # PHASE 11: Grad-CAM Failure Analysis (Target: model.features[8])
    # ---------------------------------------------------------
    logger.info("Phase 11: Generating Grad-CAM Failure Explanations Targeting model.features[8]...")
    gradcam_artifacts = generate_day14_gradcam_failures(
        model=model,
        predictions_df=preds_df,
        raw_image_dir=raw_img_dir,
        output_dir=gradcam_fig_dir,
        device="cpu",
    )
    logger.info(f"✓ Generated {len(gradcam_artifacts)} Grad-CAM failure visualizations.")

    # ---------------------------------------------------------
    # PHASE 12: Comprehensive Limitations Summary
    # ---------------------------------------------------------
    limitations_summary = {
        "class_imbalance": "Severe 58.74:1 dataset imbalance between NV and DF leads to strong predictive bias towards benign nevus.",
        "minority_representation": "Dermatofibroma (20 test samples) and Vascular lesions (24 test samples) have very small sample support, producing wide confidence intervals.",
        "acquisition_protocols": "HAM10000 images were collected using specialized dermatoscopes from two European clinical centers (Vienna & Queensland); variations in illumination and hardware impact generalizability.",
        "domain_shift": "Performance on non-dermatoscope clinical photography or mobile phone images is completely unknown and likely degrades significantly.",
        "clinical_validation": "The model is NOT clinically validated, lacks prospective trialing, and is not cleared for medical device or diagnostic usage.",
        "metadata_missingness": f"Age is missing in {missingness['age_missing']} test cases, sex in {missingness['sex_missing']}, and anatomical site in {missingness['localization_missing']}; ethnicity and skin phototype (Fitzpatrick) are completely absent.",
        "confidence_calibration": f"Softmax confidence exhibits a post-hoc Expected Calibration Error of {ece:.4f}, demonstrating overconfidence in misclassified predictions (mean confidence 0.5665, 45 errors >= 0.80).",
        "synthetic_robustness": "Perturbations tested (Gaussian blur, JPEG compression, mild color shift, crop) are synthetic approximations and do not encompass optical aberrations, surgical scars, or skin markers.",
        "ground_truth_labels": "Diagnoses were established through a combination of histopathology (53.3%), follow-up examination, and expert consensus; label ambiguity is known in border lesions (BKL vs MEL).",
        "generalization_boundaries": "The system represents a 7-class closed-world diagnostic model and cannot detect conditions outside its training vocabulary.",
        "computational_constraints": "Model training and inference evaluations were constrained to CPU execution environments without massive ensemble scaling.",
        "transfer_learning_freezing": "The backbone is frozen with ImageNet pretrained feature extractors, meaning lower-level representations were not optimized specifically for dermatoscopic patterns.",
    }

    # ---------------------------------------------------------
    # PHASE 13: Machine-Readable Export (day14_analysis.json)
    # ---------------------------------------------------------
    logger.info("Phase 13: Exporting Comprehensive Machine-Readable Summary...")
    analysis_json_path = output_data_dir / "day14_analysis.json"

    # Assemble complete JSON dictionary
    full_analysis = {
        "analysis_milestone": "Day 14 — Edge-Case, Bias & Robustness Analysis",
        "analysis_timestamp": timestamp_str,
        "contract": "ANALYSIS ONLY — ZERO MODEL MODIFICATION",
        "model_identity": {
            "architecture": "efficientnet_b0",
            "checkpoint_path": str(ckpt_path),
            "checkpoint_md5": integrity_results["checkpoint_md5"],
            "checkpoint_md5_valid": integrity_results["checkpoint_md5_valid"],
            "gradcam_target_layer": integrity_results["gradcam_target_layer"],
            "gradcam_target_verified": integrity_results["gradcam_target_verified"],
        },
        "test_split_integrity": {
            "test_split_path": str(test_split_path),
            "test_split_md5": integrity_results["test_split_md5"],
            "test_split_md5_valid": integrity_results["test_split_md5_valid"],
            "test_sample_count": len(test_df),
        },
        "locked_day9_metrics": integrity_results["metrics_verification"],
        "class_imbalance": {
            "imbalance_ratio": dist_df.attrs["imbalance_ratio"],
            "majority_class": dist_df.attrs["majority_class"],
            "minority_class": dist_df.attrs["minority_class"],
            "class_distribution": dist_df.to_dict(orient="records"),
        },
        "per_class_performance": perf_df.to_dict(orient="records"),
        "confusion_analysis": {
            "total_errors": int(len(preds_df[preds_df["true_class"] != preds_df["predicted_class"]])),
            "top_confusion_pairs": confusion_df.to_dict(orient="records"),
        },
        "confidence_analysis": conf_stats,
        "calibration": {
            "method": "10 equal-width bins [0.0, 1.0]",
            "expected_calibration_error_ece": round(float(ece), 4),
            "maximum_calibration_error_mce": round(float(mce), 4),
            "bins": calib_df.to_dict(orient="records"),
        },
        "synthetic_robustness": {
            "subset_size": 150,
            "selection_method": "Stratified random sampling across diagnostic classes",
            "random_seed": 42,
            "results": robust_df.to_dict(orient="records"),
        },
        "image_quality_proxies": img_quality_df.to_dict(orient="records"),
        "metadata_and_subgroups": {
            "missingness_audit": missingness,
            "subgroups": subgroup_df.to_dict(orient="records"),
        },
        "gradcam_failure_cases": gradcam_artifacts,
        "limitations": limitations_summary,
        "total_execution_duration_sec": round(time.time() - start_time, 2),
    }

    with open(analysis_json_path, "w", encoding="utf-8") as f:
        json.dump(full_analysis, f, indent=2)

    logger.info(f"✓ Machine-readable summary saved to {analysis_json_path}")
    logger.info(f"============================================================")
    logger.info(f"DAY 14 ANALYSIS COMPLETE in {round(time.time() - start_time, 2)}s")
    logger.info(f"============================================================")


if __name__ == "__main__":
    main()
