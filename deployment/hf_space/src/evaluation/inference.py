"""
MediScan — Model Inference & Latency Benchmarking (Day 9)

Executes evaluation inference across held-out datasets in eval mode with torch.no_grad().
Measures batch throughput and single-image latency.
Extracts per-sample predicted classes, confidences, and full probability distributions.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_test_inference(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Execute full inference across the evaluation DataLoader.

    Args:
        model: PyTorch model loaded with weights.
        dataloader: DataLoader delivering (images, labels, metadata).
        class_names: List of class names in index order.
        device: Compute device.

    Returns:
        tuple: (
            predictions_df: pd.DataFrame with per-sample predictions,
            all_logits: np.ndarray [N, num_classes],
            all_labels: np.ndarray [N],
            timing_stats: dict containing durations, throughput, and hardware info
        )
    """
    model.eval()
    records: List[Dict[str, Any]] = []
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    total_samples = 0
    batch_count = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, metadata = batch
            elif len(batch) == 2:
                images, labels = batch
                metadata = None
            else:
                raise ValueError(f"Unexpected batch structure of length {len(batch)}")

            batch_size = images.size(0)
            total_samples += batch_size
            batch_count += 1

            images = images.to(device)
            logits = model(images)

            # Check logits are finite
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits (NaN or Inf) during inference.")

            probs = torch.softmax(logits, dim=1)
            confidences, preds = torch.max(probs, dim=1)

            logits_np = logits.cpu().numpy()
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            confidences_np = confidences.cpu().numpy()

            logits_list.append(logits_np)
            labels_list.append(labels_np)

            for i in range(batch_size):
                rec: Dict[str, Any] = {}
                if metadata is not None:
                    rec["image_id"] = str(metadata["image_id"][i])
                    rec["lesion_id"] = str(metadata["lesion_id"][i])
                    rec["true_class"] = str(metadata["dx"][i])
                else:
                    rec["image_id"] = f"sample_{total_samples - batch_size + i}"
                    rec["lesion_id"] = "unknown"
                    rec["true_class"] = class_names[labels_np[i]]

                rec["predicted_class"] = class_names[preds_np[i]]
                rec["predicted_confidence"] = float(confidences_np[i])

                # Record per-class probabilities
                for c_idx, c_name in enumerate(class_names):
                    prob_val = float(probs_np[i, c_idx])
                    rec[f"prob_{c_name}"] = prob_val

                records.append(rec)

    total_duration = time.perf_counter() - start_time
    avg_time_per_image = total_duration / total_samples if total_samples > 0 else 0.0
    throughput = total_samples / total_duration if total_duration > 0 else 0.0

    predictions_df = pd.DataFrame(records)
    all_logits = np.concatenate(logits_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    # Validate probability distributions
    prob_cols = [f"prob_{c}" for c in class_names]
    prob_matrix = predictions_df[prob_cols].to_numpy()

    if not np.all(np.isfinite(prob_matrix)):
        raise ValueError("Non-finite probability values detected in predictions DataFrame.")

    if not np.all((prob_matrix >= 0.0) & (prob_matrix <= 1.0)):
        raise ValueError("Probabilities outside [0.0, 1.0] range detected.")

    prob_sums = prob_matrix.sum(axis=1)
    if not np.allclose(prob_sums, 1.0, atol=1e-4):
        raise ValueError("Predicted probabilities do not sum to 1.0 within tolerance.")

    timing_stats = {
        "total_inference_duration_sec": round(total_duration, 4),
        "total_samples": total_samples,
        "batch_count": batch_count,
        "batch_size": dataloader.batch_size,
        "avg_time_per_image_sec": round(avg_time_per_image, 6),
        "throughput_images_per_sec": round(throughput, 2),
        "device": str(device),
    }

    return predictions_df, all_logits, all_labels, timing_stats


def measure_single_image_latency(
    model: nn.Module,
    sample_tensor: torch.Tensor,
    device: torch.device,
    num_warmup: int = 5,
    num_runs: int = 30,
) -> Dict[str, float]:
    """
    Measure single-image forward pass latency with warmup.

    Args:
        model: PyTorch model.
        sample_tensor: Tensor of shape [1, 3, 224, 224].
        device: Compute device.
        num_warmup: Number of untimed warmup runs.
        num_runs: Number of timed runs.

    Returns:
        dict: Mean and median single-image latencies in seconds and milliseconds.
    """
    model.eval()
    if sample_tensor.dim() == 3:
        sample_tensor = sample_tensor.unsqueeze(0)
    sample_tensor = sample_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_tensor)

    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(sample_tensor)
            latencies.append(time.perf_counter() - t0)

    latencies_arr = np.array(latencies)
    mean_sec = float(np.mean(latencies_arr))
    median_sec = float(np.median(latencies_arr))

    return {
        "single_image_mean_latency_sec": round(mean_sec, 6),
        "single_image_median_latency_sec": round(median_sec, 6),
        "single_image_mean_latency_ms": round(mean_sec * 1000, 2),
        "single_image_median_latency_ms": round(median_sec * 1000, 2),
        "benchmark_runs": num_runs,
    }
