"""
MediScan — Grad-CAM Heatmap Visualization & Overlays (Day 10)

Provides high-resolution diagnostic plotting for Grad-CAM explainability:
- Colormap conversion and alpha blending with dermatoscopic RGB images.
- Triplet figures: [A] Original, [B] Heatmap, [C] Overlay.
- Consolidated multi-case visual comparison grids.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_gradcam_overlay(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Generate an RGB overlay of a Grad-CAM heatmap on an original RGB image.

    Args:
        image_rgb: Original image array [H, W, 3] with values in [0, 255] (uint8)
            or [0, 1] (float).
        cam: 2D activation map [H, W] normalized in [0.0, 1.0].
        alpha: Heatmap blending weight (default: 0.45).
        colormap: OpenCV colormap enum (default: cv2.COLORMAP_JET).

    Returns:
        np.ndarray: Blended RGB image of shape [H, W, 3] as uint8 in [0, 255].
    """
    # Ensure image_rgb is uint8 [0, 255]
    if image_rgb.dtype != np.uint8:
        if image_rgb.max() <= 1.0:
            img_uint8 = np.uint8(255 * np.clip(image_rgb, 0, 1))
        else:
            img_uint8 = np.uint8(np.clip(image_rgb, 0, 255))
    else:
        img_uint8 = image_rgb.copy()

    h, w = img_uint8.shape[:2]

    # Resize CAM to match image dimensions if needed
    if cam.shape != (h, w):
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        cam_resized = cam

    # Ensure CAM is in [0, 1]
    cam_clipped = np.clip(cam_resized, 0.0, 1.0)
    cam_uint8 = np.uint8(255 * cam_clipped)

    # OpenCV applyColorMap produces BGR
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Alpha blending
    overlay = cv2.addWeighted(img_uint8, 1.0 - alpha, heatmap_rgb, alpha, 0)
    return overlay


def plot_gradcam_triplet(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """
    Save a presentation-quality 3-panel figure: [A] Original, [B] Heatmap, [C] Overlay.

    Args:
        image_rgb: Original image [H, W, 3] uint8.
        cam: 2D heatmap [H, W] float in [0, 1].
        overlay: Blended overlay [H, W, 3] uint8.
        metadata: Dictionary containing diagnostic details:
            - 'image_id': str
            - 'true_class': str
            - 'predicted_class': str
            - 'confidence': float
            - 'target_class': str
            - 'description': Optional[str]
        output_path: Filepath where the plot will be saved.

    Returns:
        Path: Path to saved figure.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), dpi=300)

    # 1. Original
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Dermatoscopic Image", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    # 2. Heatmap
    im1 = axes[1].imshow(cam, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")
    cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Activation Weight", fontsize=9)

    # 3. Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap Overlay (α = 0.45)", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    # Header annotation
    img_id = metadata.get("image_id", "Unknown")
    true_cls = metadata.get("true_class", "Unknown").upper()
    pred_cls = metadata.get("predicted_class", "Unknown").upper()
    conf = metadata.get("confidence", 0.0)
    target_cls = metadata.get("target_class", pred_cls).upper()

    is_correct = (true_cls == pred_cls)
    status_color = "#1b5e20" if is_correct else "#b71c1c"
    status_text = "CORRECT" if is_correct else "MISCLASSIFIED"

    suptip = (
        f"Image: {img_id} | True: {true_cls} | Predicted: {pred_cls} ({conf:.1%}) [{status_text}]\n"
        f"Grad-CAM Target Class: {target_cls} | Layer: EfficientNet-B0 (model.features[8])"
    )

    plt.suptitle(suptip, fontsize=12, fontweight="bold", color=status_color, y=0.99)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_gradcam_grid(
    samples_list: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "MediScan Baseline — Grad-CAM Explainability Grid",
) -> Path:
    """
    Save a consolidated multi-case comparison grid.
    Each row presents: [Original Image | Grad-CAM Heatmap | Overlay].

    Args:
        samples_list: List of dicts, each containing:
            - 'image_rgb': np.ndarray
            - 'cam': np.ndarray
            - 'overlay': np.ndarray
            - 'metadata': dict (image_id, true_class, predicted_class, confidence, target_class)
        output_path: Destination filepath.
        title: Overall title for the grid.

    Returns:
        Path: Path to saved grid figure.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_samples = len(samples_list)
    if n_samples == 0:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.text(0.5, 0.5, "No Grad-CAM samples provided", ha="center", va="center")
        plt.savefig(out_path)
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3.4 * n_samples), dpi=250)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, item in enumerate(samples_list):
        meta = item["metadata"]
        img_rgb = item["image_rgb"]
        cam = item["cam"]
        overlay = item["overlay"]

        true_cls = meta.get("true_class", "unknown").upper()
        pred_cls = meta.get("predicted_class", "unknown").upper()
        target_cls = meta.get("target_class", pred_cls).upper()
        conf = meta.get("confidence", 0.0)
        img_id = meta.get("image_id", "")
        category_label = meta.get("category_label", "")

        is_correct = (true_cls == pred_cls)
        row_color = "#1b5e20" if is_correct else "#b71c1c"

        # 1. Original
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].axis("off")
        label_str = f"[{category_label}]\nID: {img_id}" if category_label else f"ID: {img_id}"
        axes[i, 0].set_title(label_str, fontsize=9.5, fontweight="bold", pad=4)

        # 2. Heatmap
        axes[i, 1].imshow(cam, cmap="jet", vmin=0.0, vmax=1.0)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Target: {target_cls} CAM", fontsize=9.5, pad=4)

        # 3. Overlay
        axes[i, 2].imshow(overlay)
        axes[i, 2].axis("off")
        axes[i, 2].set_title(
            f"True: {true_cls} → Pred: {pred_cls} ({conf:.1%})",
            fontsize=9.5,
            fontweight="bold",
            color=row_color,
            pad=4,
        )

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return out_path
