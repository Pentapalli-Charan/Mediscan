"""
MediScan — Device Detection

Safely detects the best available compute device for PyTorch operations.
Supports CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU fallback.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_device() -> str:
    """
    Detect and return the best available compute device.

    Returns:
        str: Device string — 'cuda', 'mps', or 'cpu'.

    Priority:
        1. CUDA (NVIDIA GPU) — preferred for training
        2. MPS (Apple Silicon GPU) — macOS acceleration
        3. CPU — universal fallback
    """
    if not TORCH_AVAILABLE:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_device_info() -> dict:
    """
    Return detailed information about the detected device.

    Returns:
        dict: Device information including name, type, and memory (if GPU).
    """
    device = get_device()
    info = {
        "device": device,
        "torch_available": TORCH_AVAILABLE,
    }

    if TORCH_AVAILABLE:
        info["torch_version"] = torch.__version__

        if device == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
            total_mem = torch.cuda.get_device_properties(0).total_mem
            info["gpu_memory_gb"] = round(total_mem / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda
        elif device == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"

    return info
