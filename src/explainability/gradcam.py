"""
MediScan — Grad-CAM (Gradient-weighted Class Activation Mapping) (Day 10)

Implements hook-based Grad-CAM for convolutional feature maps in PyTorch:
1. Registers forward & backward hooks on the final convolutional layer.
2. Captures activations and class-specific gradient flows.
3. Computes spatial global average pooling of gradients.
4. Generates rectified linear combination (ReLU).
5. Normalizes and upsamples heatmap to input resolution (224x224).
6. Ensures complete hook lifecycle management and gradient safety.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_default_target_layer(model: nn.Module) -> nn.Module:
    """
    Identify and return the final convolutional block of an EfficientNet-B0 model.
    In torchvision EfficientNet-B0, this is model.features[8].

    Args:
        model: EfficientNet model instance.

    Returns:
        nn.Module: The target convolutional block.
    """
    if hasattr(model, "features") and len(model.features) > 8:
        return model.features[8]
    elif hasattr(model, "features") and len(model.features) > 0:
        return model.features[-1]
    raise ValueError("Could not automatically locate the target convolutional layer in model.")


def snapshot_model_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Take a cloned snapshot of all model parameters for bitwise verification.

    Args:
        model: PyTorch model.

    Returns:
        dict: Mapping of parameter name to cloned detached tensor.
    """
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def verify_model_weights_unchanged(
    weights_before: Dict[str, torch.Tensor],
    model: nn.Module,
) -> bool:
    """
    Assert that all model parameters remain bitwise identical to the snapshot.

    Args:
        weights_before: Pre-computation snapshot from snapshot_model_weights.
        model: Model after Grad-CAM execution.

    Returns:
        bool: True if all parameters match bitwise.
    """
    for name, param in model.named_parameters():
        if name not in weights_before:
            raise KeyError(f"Parameter '{name}' not present in baseline snapshot.")
        if not torch.equal(param.detach(), weights_before[name]):
            raise AssertionError(f"Model parameter '{name}' was modified during execution!")
    return True


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) handler.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ):
        """
        Initialize Grad-CAM on a PyTorch model and target layer.

        Args:
            model: PyTorch model in eval mode.
            target_layer: The convolutional layer to inspect. Defaults to
                model.features[8] for EfficientNet-B0.
        """
        self.model = model
        self.target_layer = target_layer if target_layer is not None else get_default_target_layer(model)

        self.activations: List[torch.Tensor] = []
        self.gradients: List[torch.Tensor] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on the target layer."""
        self._remove_hooks_internal()

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor):
            self.activations.append(output)

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]):
            self.gradients.append(grad_output[0])

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [h_fwd, h_bwd]

    def _remove_hooks_internal(self) -> None:
        """Remove existing hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = []
        self.gradients = []

    def remove_hooks(self) -> None:
        """Public method to clean up all hooks when finished."""
        self._remove_hooks_internal()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float, np.ndarray]:
        """
        Generate a normalized 2D Grad-CAM heatmap for the specified input tensor.

        Args:
            input_tensor: Image tensor of shape [1, 3, H, W] or [3, H, W].
            target_class: Class index for which to compute the activation map.
                If None, defaults to the model's predicted class (argmax).

        Returns:
            tuple: (
                cam: np.ndarray [H, W] normalized in [0.0, 1.0],
                pred_class: int predicted class index,
                pred_conf: float confidence of the predicted class,
                all_probs: np.ndarray [num_classes] full softmax probability distribution
            )
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        if input_tensor.shape[0] != 1:
            raise ValueError(f"GradCAM expects batch size of 1, got {input_tensor.shape[0]}")

        # Ensure input tensor requires grad so autograd traces back through the frozen backbone
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # Ensure model is in eval mode
        self.model.eval()
        self.model.zero_grad()

        # Clear prior hooks cache
        self.activations.clear()
        self.gradients.clear()

        # Forward pass (requires grad on input or internal layers to backprop score)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = int(torch.argmax(probs, dim=1).item())
        pred_conf = float(probs[0, pred_class].item())

        selected_target = pred_class if target_class is None else target_class

        # Check activations were captured
        if not self.activations:
            raise RuntimeError("Forward hook did not capture any activations from the target layer.")

        # Target score for backpropagation
        score = logits[0, selected_target]
        score.backward(retain_graph=False)

        # Check gradients were captured
        if not self.gradients:
            raise RuntimeError("Backward hook did not capture any gradients from the target layer.")

        act = self.activations[0]  # [1, C, H_feat, W_feat]
        grad = self.gradients[0]   # [1, C, H_feat, W_feat]

        # 1. Global Average Pooling over spatial dimensions (H, W)
        alpha = torch.mean(grad, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # 2. Weighted combination of feature maps
        cam = torch.sum(alpha * act, dim=1, keepdim=True)   # [1, 1, H_feat, W_feat]

        # 3. Rectified Linear Unit (ReLU) to isolate positive contributions
        cam = torch.relu(cam)

        # 4. Upsample to input tensor resolution
        input_h, input_w = input_tensor.shape[2], input_tensor.shape[3]
        cam = torch.nn.functional.interpolate(
            cam,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=False,
        )

        # Convert to numpy [H, W]
        cam_np = cam.squeeze().detach().cpu().numpy()

        # 5. Min-Max Normalization to [0.0, 1.0]
        cam_min = float(cam_np.min())
        cam_max = float(cam_np.max())

        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Zero gradients to maintain clean model state
        self.model.zero_grad()
        self.activations.clear()
        self.gradients.clear()

        all_probs = probs.squeeze().detach().cpu().numpy()
        return cam_norm, pred_class, pred_conf, all_probs
