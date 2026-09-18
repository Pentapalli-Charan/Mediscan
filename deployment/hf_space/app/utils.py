"""
MediScan — Streamlit Web Application Utilities (Day 12)

Provides helper functions for:
- EfficientNet-B0 model loading, checkpoint verification & safe resource caching
- Exact deterministic image preprocessing (224x224, ImageNet normalized)
- 7-class probability prediction & confidence computation
- Grad-CAM heatmap & overlay generation using existing explainability modules (model.features[8])
- Safe HAM10000 dataset sample discovery
- Robust input image validation and error handling
"""

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import yaml

from src.data.preprocessing_day3 import get_augmentation_transform, load_config
from src.explainability.gradcam import GradCAM, snapshot_model_weights, verify_model_weights_unchanged
from src.explainability.visualization import create_gradcam_overlay
from src.models.efficientnet import build_model, CLASS_NAMES, NUM_CLASSES
from src.training.trainer import load_training_checkpoint

logger = logging.getLogger(__name__)

# Canonical project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical human-readable descriptions for HAM10000 7 classes
CLASS_DESCRIPTIONS: Dict[str, str] = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

# Supported image file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_IMAGE_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB max limit


def resolve_project_path(path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to the current working directory or the project root.

    Args:
        path: Path string or Path object.

    Returns:
        Path: Resolved existing path, or normalized absolute path.
    """
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    # Check relative to cwd
    if p.exists():
        return p.resolve()
    # Check relative to PROJECT_ROOT
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate.resolve()
    # Fallback to normalized path
    return candidate


def get_inference_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mediscan_model(
    checkpoint_path: Union[str, Path] = "models/checkpoints/efficientnet_b0_best.pth",
    config_path: Union[str, Path] = "config/config.yaml",
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build EfficientNet-B0 and load the validated checkpoint.

    Ensures:
    - Exactly EfficientNet-B0 architecture is built.
    - Offline instantiation (pretrained=False for architecture creation since
      full state_dict is restored from checkpoint).
    - Checkpoint contains compatible model weights and matches EfficientNet-B0.
    - Model is put into eval mode with requires_grad=False.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        config_path: Path to config.yaml.
        device: Device to map tensors onto (CPU/CUDA).

    Returns:
        tuple: (model in eval mode on device, checkpoint metadata dict)

    Raises:
        FileNotFoundError: If checkpoint or config file cannot be found.
        ValueError: If checkpoint is invalid or belongs to another architecture.
    """
    if device is None:
        device = get_inference_device()

    ckpt_file = resolve_project_path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {ckpt_file}")

    cfg_file = resolve_project_path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file}")

    # Inspect checkpoint before loading to verify architecture
    try:
        raw_checkpoint = torch.load(str(ckpt_file), map_location="cpu")
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint file '{ckpt_file}': {e}") from e

    if not isinstance(raw_checkpoint, dict) or "model_state_dict" not in raw_checkpoint:
        raise ValueError(f"Checkpoint at '{ckpt_file}' is invalid: missing 'model_state_dict'.")

    ckpt_arch = raw_checkpoint.get("architecture")
    if ckpt_arch is not None and ckpt_arch != "efficientnet_b0":
        raise ValueError(
            f"Checkpoint architecture mismatch: expected 'efficientnet_b0', got '{ckpt_arch}'. "
            f"ResNet-50 or other architectures cannot be loaded into the primary MediScan model."
        )

    ckpt_classes = raw_checkpoint.get("num_classes")
    if ckpt_classes is not None and ckpt_classes != NUM_CLASSES:
        raise ValueError(
            f"Checkpoint num_classes mismatch: expected {NUM_CLASSES}, got {ckpt_classes}."
        )

    # Instantiate model offline (pretrained=False avoids redundant CDN checks; weights are restored below)
    model = build_model(str(cfg_file), architecture="efficientnet_b0", pretrained=False)

    # Load weights
    checkpoint_meta = load_training_checkpoint(str(ckpt_file), model, device=str(device))

    model.to(device)
    model.eval()

    # Ensure no gradients on model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, checkpoint_meta


# Streamlit-safe caching wrapper
try:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading MediScan Diagnostic Model...")
    def _cached_load_mediscan_model(checkpoint_path_str: str, config_path_str: str, device_str: str):
        dev = torch.device(device_str)
        return load_mediscan_model(
            checkpoint_path=checkpoint_path_str,
            config_path=config_path_str,
            device=dev,
        )

    def load_cached_mediscan_model(
        checkpoint_path: Union[str, Path] = "models/checkpoints/efficientnet_b0_best.pth",
        config_path: Union[str, Path] = "config/config.yaml",
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Safely cached model loader for Streamlit UI.
        Uses Streamlit's resource cache when the Streamlit runtime is active.
        Falls back to direct loading outside Streamlit runtime (e.g., CLI, unit tests).
        """
        if device is None:
            device = get_inference_device()

        # Check if running within an active Streamlit runtime
        if hasattr(st, "runtime") and st.runtime.exists():
            resolved_ckpt = str(resolve_project_path(checkpoint_path))
            resolved_cfg = str(resolve_project_path(config_path))
            return _cached_load_mediscan_model(resolved_ckpt, resolved_cfg, str(device))

        return load_mediscan_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)

except ImportError:
    def load_cached_mediscan_model(
        checkpoint_path: Union[str, Path] = "models/checkpoints/efficientnet_b0_best.pth",
        config_path: Union[str, Path] = "config/config.yaml",
        device: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        return load_mediscan_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)


def validate_and_load_image(
    image_input: Any,
) -> Image.Image:
    """
    Safely load and validate an input image.

    Validates:
    - Supported formats (.jpg, .jpeg, .png)
    - File size limit (<= 25 MB)
    - Non-empty buffer / file
    - Decodable by PIL into 3-channel RGB

    Args:
        image_input: Raw image bytes, buffer, file path, UploadedFile, or PIL Image.

    Returns:
        PIL.Image in RGB format.

    Raises:
        ValueError: If image is empty, exceeds size limit, has unsupported extension, or cannot be decoded.
        FileNotFoundError: If image file path does not exist.
        TypeError: If image_input is of an unsupported type.
    """
    if isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
        return img

    # Check for Streamlit UploadedFile or file-like buffer
    if hasattr(image_input, "getvalue") and hasattr(image_input, "name"):
        name = getattr(image_input, "name", "")
        suffix = Path(name).suffix.lower()
        if suffix and suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
        data = image_input.getvalue()
        if len(data) == 0:
            raise ValueError("Uploaded image file is empty (0 bytes).")
        if len(data) > MAX_IMAGE_FILE_SIZE_BYTES:
            raise ValueError(f"Image exceeds size limit of {MAX_IMAGE_FILE_SIZE_BYTES // (1024 * 1024)} MB.")
        try:
            with Image.open(io.BytesIO(data)) as img_raw:
                return img_raw.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode uploaded image data: {e}") from e

    if isinstance(image_input, (str, Path)):
        path = resolve_project_path(image_input)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension '{suffix}'. Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError("Image file is empty (0 bytes).")
        if file_size > MAX_IMAGE_FILE_SIZE_BYTES:
            raise ValueError(f"Image file size exceeds limit ({MAX_IMAGE_FILE_SIZE_BYTES // (1024 * 1024)} MB)")

        try:
            with Image.open(path) as img_raw:
                return img_raw.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image file: {e}") from e

    if isinstance(image_input, (bytes, io.BytesIO)):
        if isinstance(image_input, bytes):
            data = image_input
            buffer = io.BytesIO(data)
        else:
            buffer = image_input
            buffer.seek(0)
            data = buffer.read()
            buffer.seek(0)

        if len(data) == 0:
            raise ValueError("Image byte data is empty (0 bytes).")
        if len(data) > MAX_IMAGE_FILE_SIZE_BYTES:
            raise ValueError(f"Image data exceeds limit ({MAX_IMAGE_FILE_SIZE_BYTES // (1024 * 1024)} MB)")

        try:
            with Image.open(buffer) as img_raw:
                return img_raw.convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to decode image data: {e}") from e

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def preprocess_image_for_inference(
    pil_image: Image.Image,
    config_path: Union[str, Path] = "config/config.yaml",
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Apply exact Day 2-4 deterministic inference preprocessing:
    - RGB conversion
    - Resize to 224 x 224
    - ImageNet normalization
    - Tensor shape [1, 3, 224, 224]

    Args:
        pil_image: PIL Image in RGB format.
        config_path: Path to config.yaml.

    Returns:
        tuple: (
            tensor: torch.Tensor of shape [1, 3, 224, 224],
            resized_rgb: np.ndarray of shape [224, 224, 3] uint8
        )

    Raises:
        ValueError: If tensor has invalid shape or non-finite values.
    """
    if not isinstance(pil_image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(pil_image)}")

    rgb_array = np.array(pil_image.convert("RGB"))
    cfg_file = resolve_project_path(config_path)
    cfg = load_config(str(cfg_file))
    transform = get_augmentation_transform(cfg, mode="test")

    transformed = transform(image=rgb_array)
    tensor = transformed["image"]  # shape: [3, 224, 224]

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

    # Verify tensor shape
    if tensor.shape != (1, 3, 224, 224):
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}, expected (1, 3, 224, 224)")

    # Verify finite values
    if not torch.isfinite(tensor).all():
        raise ValueError("Inference tensor contains non-finite values (NaN or Inf)")

    # Create resized RGB numpy array for Grad-CAM overlay
    resized_rgb = cv2.resize(rgb_array, (224, 224), interpolation=cv2.INTER_LINEAR)

    return tensor, resized_rgb


def run_model_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run forward inference with gradients disabled.

    Validates tensor dimension, computes softmax across all 7 classes,
    and returns rich diagnostic metadata.

    Args:
        model: EfficientNet-B0 model in eval mode.
        input_tensor: Shape [1, 3, 224, 224] (or [B, 3, 224, 224]).
        device: Device to execute on.

    Returns:
        dict:
            - 'predicted_class': str,
            - 'predicted_class_idx': int,
            - 'predicted_description': str,
            - 'confidence': float (0.0 - 1.0),
            - 'confidence_pct': float (0.0 - 100.0),
            - 'probabilities': Dict[str, float] for all 7 classes,
            - 'sorted_probabilities': List[Tuple[str, float]],
            - 'logits': np.ndarray of shape [7]

    Raises:
        TypeError: If input_tensor is not a torch.Tensor.
        ValueError: If tensor has invalid shape, non-finite values, or produces non-finite logits.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"input_tensor must be a torch.Tensor, got {type(input_tensor)}")

    if input_tensor.dim() != 4 or input_tensor.shape[1] != 3:
        raise ValueError(
            f"input_tensor must have shape [B, 3, H, W], got {input_tensor.shape}"
        )

    if not torch.isfinite(input_tensor).all():
        raise ValueError("input_tensor contains non-finite values (NaN or Inf)")

    if device is None:
        device = next(model.parameters()).device

    tensor_dev = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_dev)

        if not torch.isfinite(logits).all():
            raise ValueError("Model produced non-finite logits (NaN or Inf)")

        probs = torch.softmax(logits, dim=1).cpu().squeeze(0).numpy()

    if not np.isfinite(probs).all():
        raise ValueError("Model output contains non-finite probabilities")

    probs_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    sorted_probs = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "predicted_class": pred_class,
        "predicted_class_idx": pred_idx,
        "predicted_description": CLASS_DESCRIPTIONS.get(pred_class, pred_class),
        "confidence": confidence,
        "confidence_pct": round(confidence * 100.0, 2),
        "probabilities": probs_dict,
        "sorted_probabilities": sorted_probs,
        "logits": logits.cpu().squeeze(0).numpy(),
    }


def compute_gradcam_explanation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    resized_rgb: np.ndarray,
    target_class_idx: Optional[int] = None,
    alpha: float = 0.45,
) -> Dict[str, Any]:
    """
    Generate Grad-CAM activation map and blended overlay.

    Strictly targets model.features[8] (final conv block of EfficientNet-B0).
    Verifies model weights remain bitwise unchanged after execution.

    Args:
        model: EfficientNet-B0 model.
        input_tensor: Shape [1, 3, 224, 224].
        resized_rgb: Original RGB array resized to 224x224 [224, 224, 3] uint8.
        target_class_idx: Target class index. If None, uses top predicted class.
        alpha: Overlay blending weight.

    Returns:
        dict:
            - 'heatmap': np.ndarray [224, 224] float in [0, 1],
            - 'heatmap_rgb': np.ndarray [224, 224, 3] uint8 (Jet colormap),
            - 'overlay': np.ndarray [224, 224, 3] uint8,
            - 'target_class_idx': int,
            - 'target_class_name': str

    Raises:
        ValueError: If input tensor is invalid or target layer cannot be found.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"input_tensor must be a torch.Tensor, got {type(input_tensor)}")

    if input_tensor.dim() != 4 or input_tensor.shape[1] != 3:
        raise ValueError(f"input_tensor must have shape [1, 3, H, W], got {input_tensor.shape}")

    # Snapshot weights to ensure integrity
    weights_before = snapshot_model_weights(model)

    device = next(model.parameters()).device
    tensor_dev = input_tensor.to(device)

    # Strictly target model.features[8] for EfficientNet-B0
    if not hasattr(model, "features") or len(model.features) <= 8:
        raise ValueError(
            "Model does not contain target layer 'features[8]'. "
            "Ensure the model is EfficientNet-B0."
        )
    target_layer = model.features[8]

    cam_engine = GradCAM(model, target_layer=target_layer)
    try:
        cam_norm, pred_class, pred_conf, all_probs = cam_engine.generate_cam(
            tensor_dev, target_class=target_class_idx
        )
    finally:
        cam_engine.remove_hooks()
        # Verify model weights remained untouched
        verify_model_weights_unchanged(weights_before, model)

    target_idx = pred_class if target_class_idx is None else target_class_idx
    target_name = CLASS_NAMES[target_idx]

    # Convert 2D CAM to colored heatmap
    cam_uint8 = np.uint8(255 * np.clip(cam_norm, 0.0, 1.0))
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Create overlay
    overlay_rgb = create_gradcam_overlay(resized_rgb, cam_norm, alpha=alpha)

    return {
        "heatmap": cam_norm,
        "heatmap_rgb": heatmap_rgb,
        "overlay": overlay_rgb,
        "target_class_idx": target_idx,
        "target_class_name": target_name,
    }


def get_ham10000_sample_catalog(
    metadata_csv: Union[str, Path] = "data/raw/HAM10000_metadata.csv",
    image_dirs: Optional[List[Union[str, Path]]] = None,
    samples_per_class: int = 3,
    curated_dir: Union[str, Path] = "app/assets/samples",
) -> List[Dict[str, Any]]:
    """
    Discover a catalog of sample images across all 7 classes from HAM10000 dataset
    without loading all image files into memory.

    Supports automatic fallback to curated deployment samples in `curated_dir`
    when the complete raw multi-GB dataset is not present in deployment.

    Args:
        metadata_csv: Path to metadata CSV file.
        image_dirs: List of directories containing images.
        samples_per_class: Number of sample images per diagnostic class.
        curated_dir: Directory containing deployment-safe curated samples and metadata.

    Returns:
        List of dicts:
            [{
                'image_id': str,
                'dx': str,
                'description': str,
                'path': str,
                'label': str
            }, ...]
    """
    curated_path = resolve_project_path(curated_dir)
    curated_meta_file = curated_path / "samples_metadata.json"

    def _load_curated_samples() -> List[Dict[str, Any]]:
        if not curated_meta_file.exists():
            return []
        try:
            with open(curated_meta_file, "r", encoding="utf-8") as f:
                meta_list = json.load(f)
            curated_catalog = []
            for item in meta_list:
                img_path = curated_path / item.get("filename", f"{item['image_id']}.jpg")
                if img_path.exists():
                    cls = item["dx"]
                    curated_catalog.append({
                        "image_id": item["image_id"],
                        "dx": cls,
                        "description": item.get("description", CLASS_DESCRIPTIONS.get(cls, cls)),
                        "path": str(img_path),
                        "label": f"{cls.upper()} - {item['image_id']} ({CLASS_DESCRIPTIONS.get(cls, cls)})",
                    })
            return curated_catalog
        except Exception as e:
            logger.warning(f"Error loading curated sample catalog: {e}")
            return []

    csv_path = resolve_project_path(metadata_csv)
    if not csv_path.exists():
        curated_catalog = _load_curated_samples()
        if curated_catalog:
            return curated_catalog
        logger.warning(f"Metadata CSV not found at {csv_path} and no curated samples available.")
        return []

    if image_dirs is None:
        image_dirs = [
            resolve_project_path("data/raw/HAM10000_images_part_1"),
            resolve_project_path("data/raw/HAM10000_images_part_2"),
        ]
    else:
        image_dirs = [resolve_project_path(d) for d in image_dirs]

    existing_dirs = [d for d in image_dirs if d.is_dir()]
    if not existing_dirs:
        curated_catalog = _load_curated_samples()
        if curated_catalog:
            return curated_catalog
        logger.warning("No HAM10000 image directories found.")
        return []

    df = pd.read_csv(csv_path)
    catalog: List[Dict[str, Any]] = []

    for cls in CLASS_NAMES:
        cls_df = df[df["dx"] == cls]
        count = 0
        for _, row in cls_df.iterrows():
            img_id = str(row["image_id"])
            # Resolve image file
            found_path = None
            for d in existing_dirs:
                for ext in (".jpg", ".jpeg", ".png"):
                    candidate = d / f"{img_id}{ext}"
                    if candidate.exists():
                        found_path = candidate
                        break
                if found_path is not None:
                    break

            if found_path is not None:
                catalog.append({
                    "image_id": img_id,
                    "dx": cls,
                    "description": CLASS_DESCRIPTIONS.get(cls, cls),
                    "path": str(found_path),
                    "label": f"{cls.upper()} - {img_id} ({CLASS_DESCRIPTIONS.get(cls, cls)})",
                })
                count += 1
                if count >= samples_per_class:
                    break

    return catalog
