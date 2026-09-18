"""
MediScan — Reproducibility Seed Control

Sets random seeds across Python, NumPy, PyTorch, and CUDA to ensure
reproducible experiments.

NOTE: Setting torch.backends.cudnn.deterministic = True may reduce
training performance by ~10-20%. This is an acceptable tradeoff for
reproducibility in this educational project.
"""

import random
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value. Default is 42.

    Controls:
        - Python's built-in random module
        - NumPy's random number generator
        - PyTorch CPU and CUDA random number generators
        - cuDNN deterministic behavior (if CUDA available)
    """
    # Python built-in
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    if TORCH_AVAILABLE:
        # PyTorch CPU
        torch.manual_seed(seed)

        # PyTorch CUDA (all GPUs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # cuDNN deterministic behavior
        # This may reduce performance but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
