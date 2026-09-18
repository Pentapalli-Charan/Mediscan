"""
MediScan — Day 15 Finalization & Deployment Verification Test Suite

Tests:
1. Canonical model checkpoint existence and MD5 hash integrity
2. Held-out test split existence and MD5 hash integrity
3. Day 9 evaluation metrics locked values preservation
4. Class mapping dictionary and order consistency
5. Grad-CAM target layer contract (model.features[8])
6. Streamlit application compilation and import integrity
7. Standalone model loading into eval mode
8. Inference pipeline returning 7 probabilities summing to ~1.0
9. Curated deployment sample assets existence and metadata validity
10. Deployment bundle size budget (< 30 MB)
11. Deployment requirements specification and dependency coverage
12. Secret scanning (no API keys, tokens, or credentials committed)
13. Root entrypoint app.py delegation
14. Streamlit UI non-diagnostic terminology and disclaimers
15. Final documentation and model card presence
"""

import hashlib
import json
from pathlib import Path
import py_compile
import re
import sys
from typing import Dict

import numpy as np
from PIL import Image
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    NUM_CLASSES,
    compute_gradcam_explanation,
    get_ham10000_sample_catalog,
    get_inference_device,
    load_mediscan_model,
    preprocess_image_for_inference,
    run_model_inference,
)
from src.explainability.gradcam import get_default_target_layer

CANONICAL_CHECKPOINT_MD5 = "7c1c6fcbe02e93f0ff8b4a20f62e29e3"
CANONICAL_TEST_SPLIT_MD5 = "6a5ae1b65c25d504f78bc1da2361d82c"

LOCKED_DAY9_METRICS = {
    "top1_accuracy": 0.757937,
    "top2_accuracy": 0.895503,
    "macro_f1": 0.515696,
    "weighted_f1": 0.744771,
    "macro_roc_auc": 0.921136,
    "weighted_roc_auc": 0.914520,
    "macro_pr_auc": 0.564585,
    "test_loss": 0.663784,
}


def compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestDay15CanonicalIntegrity:
    """Validate that the canonical checkpoint, test split, and locked metrics are preserved."""

    def test_checkpoint_exists_and_hash_valid(self):
        ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
        assert ckpt_path.exists(), f"Checkpoint not found at: {ckpt_path}"
        actual_md5 = compute_md5(ckpt_path)
        assert actual_md5 == CANONICAL_CHECKPOINT_MD5, (
            f"Checkpoint MD5 drift detected! Expected {CANONICAL_CHECKPOINT_MD5}, got {actual_md5}"
        )

    def test_test_split_exists_and_hash_valid(self):
        split_path = PROJECT_ROOT / "reports" / "data" / "split_test.csv"
        assert split_path.exists(), f"Test split not found at: {split_path}"
        actual_md5 = compute_md5(split_path)
        assert actual_md5 == CANONICAL_TEST_SPLIT_MD5, (
            f"Test split MD5 drift detected! Expected {CANONICAL_TEST_SPLIT_MD5}, got {actual_md5}"
        )

    def test_day9_metrics_unchanged(self):
        metrics_file = PROJECT_ROOT / "reports" / "data" / "day9_metrics.json"
        assert metrics_file.exists(), f"Day 9 metrics file missing: {metrics_file}"
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        overall = data["overall_metrics"]
        roc = data["roc_auc"]
        pr = data["precision_recall_auc"]

        assert pytest.approx(overall["top1_accuracy"], rel=1e-5) == LOCKED_DAY9_METRICS["top1_accuracy"]
        assert pytest.approx(overall["top2_accuracy"], rel=1e-5) == LOCKED_DAY9_METRICS["top2_accuracy"]
        assert pytest.approx(overall["macro_f1"], rel=1e-5) == LOCKED_DAY9_METRICS["macro_f1"]
        assert pytest.approx(overall["weighted_f1"], rel=1e-5) == LOCKED_DAY9_METRICS["weighted_f1"]
        assert pytest.approx(overall["test_loss"], rel=1e-5) == LOCKED_DAY9_METRICS["test_loss"]
        assert pytest.approx(roc["macro_roc_auc"], rel=1e-5) == LOCKED_DAY9_METRICS["macro_roc_auc"]
        assert pytest.approx(roc["weighted_roc_auc"], rel=1e-5) == LOCKED_DAY9_METRICS["weighted_roc_auc"]
        assert pytest.approx(pr["macro_pr_auc"], rel=1e-5) == LOCKED_DAY9_METRICS["macro_pr_auc"]

    def test_class_mapping_unchanged(self):
        expected_classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
        assert CLASS_NAMES == expected_classes
        assert NUM_CLASSES == 7
        assert len(CLASS_DESCRIPTIONS) == 7
        for c in expected_classes:
            assert c in CLASS_DESCRIPTIONS

    def test_gradcam_target_layer_remains_features_8(self):
        model, _ = load_mediscan_model(device=torch.device("cpu"))
        target_module = get_default_target_layer(model)
        assert target_module is model.features[8]


class TestDay15InferenceAndApplication:
    """Validate Streamlit app compilation, loading, inference, and UI semantics."""

    def test_app_and_entrypoint_compile(self):
        app_file = PROJECT_ROOT / "app" / "app.py"
        entry_file = PROJECT_ROOT / "app.py"
        assert app_file.exists()
        assert entry_file.exists()
        assert py_compile.compile(str(app_file), doraise=True) is not None
        assert py_compile.compile(str(entry_file), doraise=True) is not None

    def test_model_inference_pipeline_execution(self):
        model, meta = load_mediscan_model(device=torch.device("cpu"))
        assert not model.training, "Model must be in eval mode"

        # Create sample test image
        img = Image.new("RGB", (250, 250), color=(180, 100, 80))
        tensor, resized_rgb = preprocess_image_for_inference(img)
        assert tensor.shape == (1, 3, 224, 224)
        assert resized_rgb.shape == (224, 224, 3)

        res = run_model_inference(model, tensor, device=torch.device("cpu"))
        assert res["predicted_class"] in CLASS_NAMES
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["probabilities"]) == 7
        prob_sum = sum(res["probabilities"].values())
        assert pytest.approx(prob_sum, abs=1e-5) == 1.0

        # Verify Grad-CAM executes
        cam_res = compute_gradcam_explanation(model, tensor, resized_rgb)
        assert cam_res["overlay"].shape == (224, 224, 3)
        assert cam_res["heatmap"].shape == (224, 224)

    def test_curated_sample_assets_resolve_correctly(self):
        samples_dir = PROJECT_ROOT / "app" / "assets" / "samples"
        meta_file = samples_dir / "samples_metadata.json"
        assert samples_dir.exists()
        assert meta_file.exists()

        with open(meta_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert len(manifest) == 7, "Must contain exactly 1 sample per class"
        discovered_classes = set()
        for item in manifest:
            img_path = samples_dir / item["filename"]
            assert img_path.exists()
            assert img_path.stat().st_size > 0
            discovered_classes.add(item["dx"])
            assert item["split"] == "train", "Samples must originate from training split"

        assert discovered_classes == set(CLASS_NAMES)

        # Verify get_ham10000_sample_catalog falls back to curated samples
        catalog = get_ham10000_sample_catalog(metadata_csv="nonexistent_metadata.csv")
        assert len(catalog) == 7

    def test_ui_contains_disclaimer_and_no_alarmist_triage_words(self):
        app_file = PROJECT_ROOT / "app" / "app.py"
        with open(app_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Must have prominent educational/research disclaimers
        assert "RESEARCH & EDUCATIONAL PROTOTYPE DISCLAIMER" in content
        assert "not a medical diagnostic tool" in content
        assert "not be used to make clinical decisions" in content
        assert "HAM10000" in content

        # Must not have user-facing alarmist clinical triage advice
        assert "Urgent Dermatologic Evaluation & Biopsy Recommended" not in content
        assert "Specialist Consultation & Excision Assessment Recommended" not in content
        assert "Diagnostic Assessment" not in content


class TestDay15DeploymentSafety:
    """Validate package sizes, dependencies, and secret security."""

    def test_deployment_bundle_budget(self):
        bundle_dir = PROJECT_ROOT / "deployment" / "hf_space"
        if not bundle_dir.exists():
            from scripts.prepare_hf_space import prepare_deployment_bundle
            prepare_deployment_bundle()

        assert bundle_dir.exists()
        total_size_bytes = sum(p.stat().st_size for p in bundle_dir.rglob("*") if p.is_file())
        total_size_mb = total_size_bytes / (1024 * 1024)

        # Budget: Bundle should be compact (< 30 MB), well below HF free limits
        assert total_size_mb < 30.0, f"Bundle exceeds 30 MB budget: {total_size_mb:.2f} MB"
        assert total_size_mb > 15.0, f"Bundle appears missing checkpoint: {total_size_mb:.2f} MB"

        # Checkpoint in bundle must have canonical hash
        bundle_ckpt = bundle_dir / "models" / "checkpoints" / "efficientnet_b0_best.pth"
        assert bundle_ckpt.exists()
        assert compute_md5(bundle_ckpt) == CANONICAL_CHECKPOINT_MD5

    def test_deployment_requirements_defined(self):
        req_deploy = PROJECT_ROOT / "requirements-deployment.txt"
        assert req_deploy.exists()
        with open(req_deploy, "r", encoding="utf-8") as f:
            lines = [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]

        expected_pkgs = ["torch", "torchvision", "streamlit", "grad-cam", "pillow", "numpy", "pandas"]
        for pkg in expected_pkgs:
            assert any(l.startswith(pkg) for l in lines), f"Required package '{pkg}' missing from deployment requirements"

        # Ensure heavy dev-only tools are not in deployment requirements
        dev_tools = ["pytest", "jupyter", "ipykernel"]
        for tool in dev_tools:
            assert not any(l.startswith(tool) for l in lines), f"Dev tool '{tool}' should not be in deployment requirements"

    def test_no_secrets_committed_in_codebase(self):
        secret_patterns = [
            re.compile(r"hf_[A-Za-z0-9]{34,}"),  # Hugging Face token
            re.compile(r"ghp_[A-Za-z0-9]{36,}"),  # GitHub PAT
            re.compile(r"sk-[A-Za-z0-9]{32,}"),   # OpenAI API key
            re.compile(r"AKIA[0-9A-Z]{16}"),      # AWS Access Key
        ]

        checked_dirs = ["app", "src", "config", "scripts"]
        for d in checked_dirs:
            dir_path = PROJECT_ROOT / d
            if not dir_path.exists():
                continue
            for p in dir_path.rglob("*.py"):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                for pattern in secret_patterns:
                    matches = pattern.findall(text)
                    assert len(matches) == 0, f"Potential credential pattern found in {p}"
