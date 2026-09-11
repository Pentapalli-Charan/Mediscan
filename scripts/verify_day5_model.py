"""
MediScan — Day 5 Model Verification Script

Verifies the base EfficientNet-B0 transfer learning model implementation:
1. Model initialization and ImageNet pretrained weight loading
2. Classifier head replacement to 7 classes (raw logits, no Softmax)
3. Parameter freezing (backbone frozen, classifier trainable)
4. Device placement (CPU, CUDA if available)
5. Forward pass on real DataLoader batch [32, 3, 224, 224] -> [32, 7]
6. No NaN/Inf in logits
7. CrossEntropyLoss compatibility on real batch
8. Untrained initialization checkpoint creation
9. Model summary generation in JSON and Markdown formats
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataloader import create_dataloaders
from src.models import (
    CLASS_NAMES,
    NUM_CLASSES,
    build_model,
    count_parameters,
    get_model_summary,
    save_initialization_checkpoint,
)
from src.utils.device import get_device, get_device_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_verification():
    logger.info("=" * 60)
    logger.info("MediScan — Day 5 Base Model Verification")
    logger.info("=" * 60)

    # 1. Device detection
    device_str = get_device()
    device_info = get_device_info()
    logger.info(f"Compute Device detected: {device_str}")
    logger.info(f"Device Info: {device_info}")

    # 2. Build model from config
    logger.info("Instantiating EfficientNet-B0 from config...")
    model = build_model()
    logger.info(f"Model architecture: {model.architecture}")
    logger.info(f"Weights used: {model.weights_used}")

    # 3. Parameter counts
    param_counts = count_parameters(model)
    logger.info(f"Total parameters:     {param_counts['total']:,}")
    logger.info(f"Trainable parameters: {param_counts['trainable']:,}")
    logger.info(f"Frozen parameters:    {param_counts['frozen']:,}")
    logger.info(f"Trainable percentage: {param_counts['trainable_pct']:.4f}%")

    # Verify parameter counts match expectation
    assert param_counts["total"] == 4016515, f"Unexpected total params: {param_counts['total']}"
    assert param_counts["trainable"] == 8967, f"Unexpected trainable params: {param_counts['trainable']}"
    assert param_counts["frozen"] == 4007548, f"Unexpected frozen params: {param_counts['frozen']}"

    # 4. Programmatic verification of requires_grad
    feature_frozen = all(not p.requires_grad for p in model.features.parameters())
    classifier_trainable = all(p.requires_grad for p in model.classifier.parameters())
    logger.info(f"Backbone fully frozen: {feature_frozen}")
    logger.info(f"Classifier head fully trainable: {classifier_trainable}")
    assert feature_frozen, "Error: Backbone features are not frozen!"
    assert classifier_trainable, "Error: Classifier parameters are not trainable!"

    # 5. Move to device and set eval mode
    device = torch.device(device_str)
    model.to(device)
    model.eval()

    # 6. Real DataLoader forward pass
    logger.info("Loading one batch from train DataLoader...")
    loaders = create_dataloaders()
    train_loader = loaders["train"]
    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)
    batch_size = images.shape[0]

    logger.info(f"Input batch shape: {list(images.shape)}, dtype: {images.dtype}")
    logger.info(f"Input labels shape: {list(labels.shape)}, dtype: {labels.dtype}")

    # Forward pass
    start_time = time.time()
    with torch.no_grad():
        logits = model(images)
    forward_time_ms = (time.time() - start_time) * 1000

    logger.info(f"Forward pass completed in {forward_time_ms:.2f} ms")
    logger.info(f"Output logits shape: {list(logits.shape)}")

    # Assertions on output
    assert logits.shape == (batch_size, NUM_CLASSES), f"Unexpected output shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN found in logits!"
    assert not torch.isinf(logits).any(), "Inf found in logits!"

    # 7. CrossEntropyLoss compatibility
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    loss_val = loss.item()
    logger.info(f"CrossEntropyLoss calculated: {loss_val:.4f}")
    assert not torch.isnan(loss).any(), "NaN in CrossEntropyLoss!"
    assert not torch.isinf(loss).any(), "Inf in CrossEntropyLoss!"
    assert loss_val > 0.0, "Loss value must be positive"

    # 8. Checkpoint structure (untrained initialization checkpoint)
    checkpoint_dir = PROJECT_ROOT / "models" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "efficientnet_b0_init.pth"
    save_initialization_checkpoint(
        model,
        checkpoint_path,
        metadata={
            "device": device_str,
            "verification_loss": loss_val,
            "forward_time_ms": forward_time_ms,
        },
    )
    logger.info(f"Saved initialization checkpoint to: {checkpoint_path}")
    assert checkpoint_path.exists(), "Checkpoint file was not created!"

    # 9. Structured Summary
    summary = get_model_summary(model)
    summary["device_tested"] = device_str
    summary["device_info"] = device_info
    summary["forward_pass_batch_size"] = batch_size
    summary["forward_pass_time_ms"] = round(forward_time_ms, 2)
    summary["sample_loss"] = round(loss_val, 4)
    summary["weights_trained"] = False
    summary["checkpoint_saved"] = str(checkpoint_path)

    # Save summary JSON
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "day5_model_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved model summary JSON to: {json_path}")

    # Save summary Markdown
    md_path = reports_dir / "DAY5_MODEL_SUMMARY.md"
    md_content = f"""# MediScan — Day 5 Base Model Summary: EfficientNet-B0

## 1. Model Overview
- **Architecture:** `{summary['architecture']}`
- **Pretrained Weights:** `{summary['pretrained_weights']}`
- **Input Shape:** `{summary['input_shape']}` (Batch x Channels x Height x Width)
- **Output Shape:** `{summary['output_shape']}` (Batch x Classes)
- **Number of Diagnostic Classes:** `{summary['num_classes']}`
- **Target Classes:** `{', '.join(CLASS_NAMES)}`
- **Output Activation:** Raw logits (No Softmax in forward pass)

## 2. Transfer Learning Parameter Distribution
| Category | Parameter Count | Percentage | Status |
|---|---|---|---|
| **Total Parameters** | {summary['total_parameters']:,} | 100.00% | Full Model |
| **Trainable Parameters** | {summary['trainable_parameters']:,} | {summary['trainable_pct']:.2f}% | Classifier Head (`requires_grad=True`) |
| **Frozen Parameters** | {summary['frozen_parameters']:,} | {100.0 - summary['trainable_pct']:.2f}% | Backbone Features (`requires_grad=False`) |

## 3. Classifier Head Structure
```
{summary['classifier_structure']}
```
- **In-features:** 1280
- **Dropout Rate:** 0.2
- **Out-features:** 7 (akiec=0, bcc=1, bkl=2, df=3, mel=4, nv=5, vasc=6)

## 4. Forward Pass & Loss Verification
- **Device Tested:** `{device_str}` ({device_info.get('torch_version', 'torch')})
- **Real DataLoader Batch Shape:** `[{batch_size}, 3, 224, 224]`
- **Real DataLoader Labels Shape:** `[{batch_size}]`
- **Output Logits Shape:** `[{batch_size}, 7]`
- **Logit Range Check:** Finite, 0 NaN, 0 Inf
- **Forward Pass Latency:** {forward_time_ms:.2f} ms
- **CrossEntropyLoss on Real Batch:** {loss_val:.4f} (compatible, finite, non-NaN)

## 5. Checkpoint & Training Status
- **Weights Trained:** NO (0 updates performed, strictly transfer learning base initialization)
- **Initialization Checkpoint:** `{checkpoint_path.relative_to(PROJECT_ROOT)}`
- **Status:** **DAY 5 BASE MODEL VERIFIED & READY FOR DAY 6**
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved model summary Markdown to: {md_path}")

    logger.info("=" * 60)
    logger.info("DAY 5 VERIFICATION PASSED SUCCESSFULLY")
    logger.info("=" * 60)
    return summary


if __name__ == "__main__":
    run_verification()
