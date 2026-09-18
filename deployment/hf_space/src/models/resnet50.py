"""
MediScan — ResNet-50 Comparison Architecture (Day 11)

Implements the ResNet-50 transfer learning model for comparative benchmarking
against the primary EfficientNet-B0 baseline:
- Backbone: torchvision.models.resnet50
- Pretrained Weights: ResNet50_Weights.IMAGENET1K_V1
- Feature dimension: 2048
- Classifier Head: Sequential(Dropout(p=0.2), Linear(2048, 7))
- Output: Raw logits [B, 7]
- Base Strategy: Feature extraction (frozen backbone, trainable classifier head)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7


def count_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """Count total, trainable, and frozen parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_pct = round((trainable / total * 100), 4) if total > 0 else 0.0

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable_pct,
    }


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all feature extractor layers in ResNet-50.
    Only the classification head (model.fc) will have requires_grad=True.
    """
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters in the model for fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def get_resnet50(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone_weights: bool = True,
    dropout: float = 0.2,
    weights_name: str = "IMAGENET1K_V1",
) -> nn.Module:
    """
    Instantiate and configure torchvision's pretrained ResNet-50 for MediScan.

    Args:
        num_classes: Number of target classes (default: 7).
        pretrained: Whether to load ImageNet pretrained weights (default: True).
        freeze_backbone_weights: Whether to freeze feature extractor parameters (default: True).
        dropout: Dropout rate for classifier head (default: 0.2).
        weights_name: Name of torchvision weights enum (default: 'IMAGENET1K_V1').

    Returns:
        nn.Module: Configured ResNet-50 model returning raw logits [B, num_classes].
    """
    if pretrained:
        if weights_name in ("IMAGENET1K_V1", "DEFAULT", "default"):
            weights = ResNet50_Weights.IMAGENET1K_V1
        elif weights_name == "IMAGENET1K_V2":
            weights = ResNet50_Weights.IMAGENET1K_V2
        else:
            try:
                weights = getattr(ResNet50_Weights, weights_name)
            except AttributeError:
                logger.warning(
                    f"Unknown weights enum '{weights_name}', defaulting to IMAGENET1K_V1."
                )
                weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
        exact_weights_str = str(weights)
    else:
        model = models.resnet50(weights=None)
        exact_weights_str = "None (random initialization)"

    in_features = model.fc.in_features  # 2048 for ResNet-50

    # Replace fc with custom classification head for 7 classes
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout, inplace=False),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True),
    )

    # Apply transfer learning freezing strategy if requested
    if freeze_backbone_weights:
        freeze_backbone(model)

    # Attach metadata attributes to model instance
    model.architecture = "resnet50"
    model.num_classes = num_classes
    model.pretrained = pretrained
    model.weights_used = exact_weights_str
    model.in_features = in_features
    model.backbone_frozen = freeze_backbone_weights
    model.dropout_p = dropout

    return model


def get_resnet50_summary(model: nn.Module) -> Dict[str, Any]:
    """Generate structured summary dictionary of the ResNet-50 model."""
    param_counts = count_parameters(model)
    classifier_repr = str(getattr(model, "fc", None))

    return {
        "architecture": getattr(model, "architecture", "resnet50"),
        "pretrained_weights": getattr(model, "weights_used", "Unknown"),
        "input_shape": [1, 3, 224, 224],
        "output_shape": [1, getattr(model, "num_classes", NUM_CLASSES)],
        "num_classes": getattr(model, "num_classes", NUM_CLASSES),
        "total_parameters": param_counts["total"],
        "trainable_parameters": param_counts["trainable"],
        "frozen_parameters": param_counts["frozen"],
        "trainable_pct": param_counts["trainable_pct"],
        "classifier_structure": classifier_repr,
        "backbone_frozen": getattr(model, "backbone_frozen", True),
    }
