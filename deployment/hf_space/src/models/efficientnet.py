"""
MediScan — EfficientNet-B0 Base Model Implementation

This module establishes the base EfficientNet-B0 transfer learning model for
the MediScan skin lesion classification task.

Key Architectural Details:
- Backbone: torchvision.models.efficientnet_b0
- Pretrained Weights: EfficientNet_B0_Weights.IMAGENET1K_V1 (DEFAULT)
- Feature extractor dimension: 1280
- Classifier Head: Sequential(Dropout(p=0.2), Linear(1280, 7))
- Output: Raw logits [B, 7] (no internal Softmax)
- Base Strategy: Feature extraction (frozen backbone, trainable classifier head)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

logger = logging.getLogger(__name__)

# Canonical class labels for HAM10000 7-class diagnostic task
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = 7


def count_parameters(model: nn.Module) -> Dict[str, Union[int, float]]:
    """
    Count total, trainable, and frozen parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        dict: Parameter counts:
            - 'total': total parameter count
            - 'trainable': trainable parameter count (requires_grad=True)
            - 'frozen': frozen parameter count (requires_grad=False)
            - 'trainable_pct': percentage of parameters that are trainable
    """
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
    Freeze the feature extractor backbone of an EfficientNet model.
    Only the classifier head will have requires_grad=True.

    Args:
        model: EfficientNet model instance.
    """
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False

    # Ensure classifier parameters remain trainable
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all parameters in the model for fine-tuning.

    Args:
        model: EfficientNet model instance.
    """
    for param in model.parameters():
        param.requires_grad = True


def get_efficientnet_b0(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone_weights: bool = True,
    dropout: float = 0.2,
    weights_name: str = "IMAGENET1K_V1",
) -> nn.Module:
    """
    Instantiate and configure torchvision's pretrained EfficientNet-B0 for MediScan.

    Args:
        num_classes: Number of target classes (default: 7).
        pretrained: Whether to load ImageNet pretrained weights (default: True).
        freeze_backbone_weights: Whether to freeze feature extractor parameters (default: True).
        dropout: Dropout rate for classifier head (default: 0.2).
        weights_name: Name of torchvision weights enum (default: 'IMAGENET1K_V1').

    Returns:
        nn.Module: Configured EfficientNet-B0 model returning raw logits [B, num_classes].
    """
    if pretrained:
        if weights_name in ("IMAGENET1K_V1", "DEFAULT", "default"):
            weights = EfficientNet_B0_Weights.DEFAULT
        else:
            try:
                weights = getattr(EfficientNet_B0_Weights, weights_name)
            except AttributeError:
                logger.warning(
                    f"Unknown weights enum '{weights_name}', defaulting to IMAGENET1K_V1."
                )
                weights = EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        exact_weights_str = str(weights)
    else:
        model = models.efficientnet_b0(weights=None)
        exact_weights_str = "None (random initialization)"

    # Inspect and replace the classifier head
    # Original: Sequential(Dropout(p=0.2, inplace=True), Linear(in_features=1280, out_features=1000))
    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0

    # Replace classifier with custom head for 7 classes
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes, bias=True),
    )

    # Apply transfer learning freezing strategy if requested
    if freeze_backbone_weights:
        freeze_backbone(model)

    # Attach metadata attributes to model instance for easy inspection
    model.architecture = "efficientnet_b0"
    model.num_classes = num_classes
    model.pretrained = pretrained
    model.weights_used = exact_weights_str
    model.in_features = in_features
    model.backbone_frozen = freeze_backbone_weights
    model.dropout_p = dropout

    return model


def build_model(
    config: Optional[Union[Dict[str, Any], str, Path]] = None,
    **kwargs: Any,
) -> nn.Module:
    """
    Factory function to build a model based on project configuration.

    Args:
        config: Optional configuration dictionary, or path to YAML config file.
        **kwargs: Overrides for configuration parameters.

    Returns:
        nn.Module: Configured PyTorch model.
    """
    model_cfg: Dict[str, Any] = {}

    if isinstance(config, (str, Path)):
        import yaml

        with open(config, "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f)
            model_cfg = full_cfg.get("model", {})
    elif isinstance(config, dict):
        if "model" in config:
            model_cfg = config["model"]
        else:
            model_cfg = config
    else:
        # Default config lookup
        default_config_path = (
            Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"
        )
        if default_config_path.exists():
            import yaml

            with open(default_config_path, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f)
                model_cfg = full_cfg.get("model", {})

    architecture = kwargs.get(
        "architecture", model_cfg.get("architecture", model_cfg.get("primary", "efficientnet_b0"))
    )
    pretrained = kwargs.get("pretrained", model_cfg.get("pretrained", True))
    num_classes = kwargs.get("num_classes", model_cfg.get("num_classes", NUM_CLASSES))
    freeze_bb = kwargs.get(
        "freeze_backbone", model_cfg.get("freeze_backbone", True)
    )
    dropout = kwargs.get("dropout", model_cfg.get("dropout", 0.2))
    weights_name = kwargs.get("weights", model_cfg.get("weights", "IMAGENET1K_V1"))

    if architecture == "efficientnet_b0":
        return get_efficientnet_b0(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone_weights=freeze_bb,
            dropout=dropout,
            weights_name=weights_name,
        )
    elif architecture == "resnet50":
        from src.models.resnet50 import get_resnet50
        return get_resnet50(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone_weights=freeze_bb,
            dropout=dropout,
            weights_name=weights_name,
        )
    else:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Currently supported: 'efficientnet_b0', 'resnet50'."
        )


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Generate a comprehensive structured summary dictionary of the model.

    Args:
        model: Model instance.

    Returns:
        dict: Summary metadata.
    """
    param_counts = count_parameters(model)
    classifier_repr = str(getattr(model, "classifier", None))

    return {
        "architecture": getattr(model, "architecture", "efficientnet_b0"),
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


def save_initialization_checkpoint(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save an untrained initialization checkpoint clearly labeled as an initial/pretrained checkpoint.

    Args:
        model: Model instance.
        checkpoint_path: Destination filepath.
        metadata: Optional additional metadata dictionary.

    Returns:
        Path: Path to the saved checkpoint file.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = get_model_summary(model)
    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "checkpoint_type": "initialization_pretrained",
        "description": "Untrained base EfficientNet-B0 transfer learning model initialization",
        "summary": summary,
        "metadata": metadata or {},
    }

    torch.save(checkpoint_dict, str(path))
    logger.info(f"Saved initialization checkpoint to {path}")
    return path
