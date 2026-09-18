"""
MediScan — Hugging Face Space Deployment Bundle Preparation Script

Assembles a self-contained, lightweight deployment bundle for Hugging Face Spaces:
- Root entrypoint (app.py)
- Web application (app/app.py, app/utils.py, app/assets/samples/)
- Source inference modules (src/models, src/explainability, src/data, src/training)
- Canonical configuration (config/config.yaml)
- Canonical frozen model weights (models/checkpoints/efficientnet_b0_best.pth)
- Deployment requirements (requirements.txt)
- Space metadata card (README.md with Hugging Face YAML frontmatter)
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = PROJECT_ROOT / "deployment" / "hf_space"

CANONICAL_CHECKPOINT_MD5 = "7c1c6fcbe02e93f0ff8b4a20f62e29e3"


def compute_md5(file_path: Path) -> str:
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def prepare_deployment_bundle() -> Path:
    print(f"Preparing Hugging Face Spaces bundle at: {BUNDLE_DIR}")
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy root app.py entrypoint
    shutil.copy2(PROJECT_ROOT / "app.py", BUNDLE_DIR / "app.py")

    # 2. Copy app/ directory (including app/assets/samples)
    shutil.copytree(PROJECT_ROOT / "app", BUNDLE_DIR / "app")

    # 3. Copy config directory
    shutil.copytree(PROJECT_ROOT / "config", BUNDLE_DIR / "config")

    # 4. Copy src/ modules
    shutil.copytree(PROJECT_ROOT / "src", BUNDLE_DIR / "src")

    # 5. Copy canonical checkpoint
    ckpt_dest = BUNDLE_DIR / "models" / "checkpoints"
    ckpt_dest.mkdir(parents=True, exist_ok=True)
    src_ckpt = PROJECT_ROOT / "models" / "checkpoints" / "efficientnet_b0_best.pth"
    if not src_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {src_ckpt}")
    shutil.copy2(src_ckpt, ckpt_dest / "efficientnet_b0_best.pth")

    # Verify copied checkpoint hash
    copied_md5 = compute_md5(ckpt_dest / "efficientnet_b0_best.pth")
    if copied_md5 != CANONICAL_CHECKPOINT_MD5:
        raise ValueError(f"MD5 mismatch on bundle checkpoint: expected {CANONICAL_CHECKPOINT_MD5}, got {copied_md5}")
    print(f"Checkpoint MD5 verified: {copied_md5}")

    # 6. Copy deployment requirements as requirements.txt in bundle
    shutil.copy2(PROJECT_ROOT / "requirements-deployment.txt", BUNDLE_DIR / "requirements.txt")

    # 7. Create README.md with HF Spaces YAML frontmatter
    readme_content = """---
title: MediScan — Dermatoscopic Skin Lesion Classifier
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.54.0
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
short_description: Educational AI skin lesion classifier with Grad-CAM interpretability
---

# MediScan — Skin Lesion Classification Prototype

> ⚠️ **DISCLAIMER:** This application is an **educational and research prototype** for automated dermatoscopic image classification. It is **not a medical diagnostic tool** and must **never be used for clinical decision-making or patient triage**. Predictions may be incorrect and performance varies across populations, devices, and illumination settings.

## Model Overview
- **Backbone Architecture:** EfficientNet-B0 (Pretrained on ImageNet-1K, frozen feature extractor)
- **Dataset:** HAM10000 (10,015 dermatoscopic images across 7 classes)
- **Input Resolution:** 224 × 224 RGB (ImageNet normalized)
- **Explainability:** Grad-CAM visual activation targeting `model.features[8]`
- **Locked Test Accuracy:** 75.79% Top-1, 89.55% Top-2
- **Macro ROC-AUC:** 0.9211

## 7 Diagnostic Categories
1. **AKIEC:** Actinic keratoses / Intraepithelial carcinoma
2. **BCC:** Basal cell carcinoma
3. **BKL:** Benign keratosis-like lesions
4. **DF:** Dermatofibroma
5. **MEL:** Melanoma
6. **NV:** Melanocytic nevi
7. **VASC:** Vascular lesions

## Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
"""
    with open(BUNDLE_DIR / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)

    # 8. Compute total bundle size and file manifest
    total_size = sum(p.stat().st_size for p in BUNDLE_DIR.rglob("*") if p.is_file())
    file_count = sum(1 for p in BUNDLE_DIR.rglob("*") if p.is_file())
    print(f"Deployment bundle ready:")
    print(f"  Total files: {file_count}")
    print(f"  Total size: {total_size / (1024*1024):.2f} MB")
    print(f"  Location: {BUNDLE_DIR}")
    return BUNDLE_DIR


if __name__ == "__main__":
    prepare_deployment_bundle()
