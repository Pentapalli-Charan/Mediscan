"""
MediScan — Day 4: PyTorch DataLoader Pipeline

This module provides factory functions for constructing train, validation,
and test DataLoaders from the Day 3 HAM10000Dataset and augmentation
transforms.

All DataLoader settings (batch_size, num_workers, pin_memory, drop_last,
persistent_workers) are read from config/config.yaml -> dataloader section.

Functions:
    create_dataloaders       — Build train/val/test DataLoaders from config
    verify_image_paths       — Check every split image can be located on disk
    verify_batch             — Validate shape, dtype, label range of a batch
    verify_dataset_accounting — Check dataset lengths match expected counts
    verify_class_labels      — Confirm class mapping across all datasets
    benchmark_dataloader     — Measure loading throughput (samples/sec)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.preprocessing_day3 import (
    HAM10000Dataset,
    get_augmentation_transform,
    get_class_label_mapping,
    load_config,
)


# ═════════════════════════════════════════════════════════════════════
# DATALOADER FACTORY
# ═════════════════════════════════════════════════════════════════════


def create_dataloaders(
    config_path: str = "config/config.yaml",
    project_root: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders from configuration.

    All DataLoader hyper-parameters are read from the ``dataloader`` section
    of the YAML config.  Transforms come from Day 3's
    ``get_augmentation_transform``.

    Args:
        config_path: Path to config YAML (relative to project_root or CWD).
        project_root: Absolute path to the project root.  Defaults to this
            file's grandparent-grandparent directory.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to their
        respective ``torch.utils.data.DataLoader`` instances.
    """
    if project_root is None:
        project_root = str(_PROJECT_ROOT)
    project_root = Path(project_root)

    # Resolve config path
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path

    config = load_config(str(cfg_path))

    # DataLoader settings (with safe defaults)
    dl_cfg = config.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 32)
    num_workers = dl_cfg.get("num_workers", 2)
    pin_memory = dl_cfg.get("pin_memory", True)
    drop_last = dl_cfg.get("drop_last", False)
    persistent_workers = dl_cfg.get("persistent_workers", False)

    # On Windows, persistent_workers with num_workers=0 is invalid
    if num_workers == 0:
        persistent_workers = False

    # Paths
    splits_dir = project_root / "reports" / "data"
    image_base_dir = str(project_root / config.get("paths", {}).get("data_raw", "data/raw"))

    # Build datasets with correct transforms
    datasets = {}
    for split_name, mode in [("train", "train"), ("val", "val"), ("test", "test")]:
        csv_path = str(splits_dir / f"split_{split_name}.csv")
        transform = get_augmentation_transform(config, mode=mode)
        datasets[split_name] = HAM10000Dataset(
            split_csv_path=csv_path,
            image_base_dir=image_base_dir,
            transform=transform,
            include_metadata=False,
        )

    # Build DataLoaders
    loaders = {}
    loaders["train"] = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    for split_name in ("val", "test"):
        loaders[split_name] = DataLoader(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,  # Never drop for evaluation sets
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    return loaders


# ═════════════════════════════════════════════════════════════════════
# VERIFICATION UTILITIES
# ═════════════════════════════════════════════════════════════════════


def verify_image_paths(
    project_root: Optional[str] = None,
) -> Dict:
    """
    Verify that every image_id in all three split CSVs can be found on disk.

    Returns:
        dict with ``success`` bool, ``missing`` list, and per-split counts.
    """
    if project_root is None:
        project_root = str(_PROJECT_ROOT)
    root = Path(project_root)

    splits_dir = root / "reports" / "data"
    image_base_dir = root / "data" / "raw"

    # Locate image directories
    image_dirs = sorted(
        p for p in image_base_dir.iterdir()
        if p.is_dir() and "HAM10000_images" in p.name
    )
    if not image_dirs:
        return {"success": False, "error": f"No HAM10000_images dirs in {image_base_dir}", "missing": []}

    # Build a set of available stems for fast lookup
    available_stems = set()
    for d in image_dirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                available_stems.add(f.stem)

    missing = []
    counts = {}
    for split_name in ("train", "val", "test"):
        csv_path = splits_dir / f"split_{split_name}.csv"
        df = pd.read_csv(csv_path)
        counts[split_name] = len(df)
        for image_id in df["image_id"]:
            if str(image_id) not in available_stems:
                missing.append({"split": split_name, "image_id": str(image_id)})

    return {
        "success": len(missing) == 0,
        "missing": missing,
        "counts": counts,
        "total_available_on_disk": len(available_stems),
    }


def verify_batch(
    dataloader: DataLoader,
    split_name: str = "unknown",
    expected_channels: int = 3,
    expected_size: int = 224,
    num_classes: int = 7,
) -> Dict:
    """
    Pull one batch from a DataLoader and verify shapes/types/ranges.

    Args:
        dataloader: A DataLoader to pull a batch from.
        split_name: Name used in log messages (e.g. ``"train"``).
        expected_channels: Expected number of image channels.
        expected_size: Expected spatial dimension (height == width).
        num_classes: Number of classes (labels must be in ``[0, num_classes)``).

    Returns:
        dict with ``success``, ``errors``, ``batch_size``, ``image_shape``,
        ``label_dtype``, ``label_min``, ``label_max``.
    """
    errors = []
    info: Dict = {}

    try:
        batch = next(iter(dataloader))
        images, labels = batch[0], batch[1]

        info["batch_size"] = images.shape[0]
        info["image_shape"] = tuple(images.shape)
        info["label_dtype"] = str(labels.dtype)
        info["label_min"] = int(labels.min())
        info["label_max"] = int(labels.max())

        # Shape check: [B, C, H, W]
        if images.dim() != 4:
            errors.append(f"[{split_name}] Expected 4-D tensor, got {images.dim()}-D")
        if images.shape[1] != expected_channels:
            errors.append(f"[{split_name}] Expected {expected_channels} channels, got {images.shape[1]}")
        if images.shape[2] != expected_size or images.shape[3] != expected_size:
            errors.append(
                f"[{split_name}] Expected spatial {expected_size}x{expected_size}, "
                f"got {images.shape[2]}x{images.shape[3]}"
            )

        # Label range
        if labels.min() < 0 or labels.max() >= num_classes:
            errors.append(
                f"[{split_name}] Labels out of range [0, {num_classes}): "
                f"min={labels.min()}, max={labels.max()}"
            )

        # NaN / Inf
        if torch.isnan(images).any():
            errors.append(f"[{split_name}] Batch contains NaN values")
        if torch.isinf(images).any():
            errors.append(f"[{split_name}] Batch contains Inf values")

    except Exception as e:
        errors.append(f"[{split_name}] Failed to load batch: {e}")

    info["success"] = len(errors) == 0
    info["errors"] = errors
    return info


def verify_dataset_accounting(
    loaders: Dict[str, DataLoader],
    expected_train: int = 6982,
    expected_val: int = 1521,
    expected_test: int = 1512,
) -> Dict:
    """
    Verify that dataset lengths match expected split sizes.

    Returns:
        dict with ``success``, ``errors``, and per-split actual/expected counts.
    """
    errors = []
    actual = {
        "train": len(loaders["train"].dataset),
        "val": len(loaders["val"].dataset),
        "test": len(loaders["test"].dataset),
    }
    expected = {"train": expected_train, "val": expected_val, "test": expected_test}

    for name in ("train", "val", "test"):
        if actual[name] != expected[name]:
            errors.append(
                f"{name}: expected {expected[name]}, got {actual[name]}"
            )

    total_actual = sum(actual.values())
    total_expected = sum(expected.values())
    if total_actual != total_expected:
        errors.append(f"Total: expected {total_expected}, got {total_actual}")

    return {
        "success": len(errors) == 0,
        "errors": errors,
        "actual": actual,
        "expected": expected,
        "total_actual": total_actual,
        "total_expected": total_expected,
    }


def verify_class_labels(
    loaders: Dict[str, DataLoader],
) -> Dict:
    """
    Confirm that every dataset uses the same class-label mapping and that
    no unexpected labels appear in the split CSVs.

    Returns:
        dict with ``success``, ``errors``, and the canonical mapping.
    """
    errors = []
    canonical = get_class_label_mapping()

    for name, loader in loaders.items():
        ds = loader.dataset
        if hasattr(ds, "class_label_mapping"):
            if ds.class_label_mapping != canonical:
                errors.append(
                    f"[{name}] Class mapping differs from canonical: "
                    f"{ds.class_label_mapping}"
                )
        else:
            errors.append(f"[{name}] Dataset has no class_label_mapping attribute")

        # Check that all dx values in the CSV are in the mapping
        unexpected = set(ds.df["dx"].unique()) - set(canonical.keys())
        if unexpected:
            errors.append(f"[{name}] Unexpected class labels: {unexpected}")

    return {
        "success": len(errors) == 0,
        "errors": errors,
        "canonical_mapping": canonical,
    }


# ═════════════════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARK
# ═════════════════════════════════════════════════════════════════════


def benchmark_dataloader(
    dataloader: DataLoader,
    max_batches: int = 20,
    split_name: str = "train",
) -> Dict:
    """
    Time DataLoader iteration to estimate loading throughput.

    Args:
        dataloader: DataLoader to benchmark.
        max_batches: Maximum number of batches to iterate.
        split_name: Human-readable label for the report.

    Returns:
        dict with samples_processed, batches_processed, elapsed_seconds,
        and samples_per_second.
    """
    samples = 0
    batches = 0

    start = time.perf_counter()
    for batch in dataloader:
        images = batch[0]
        samples += images.shape[0]
        batches += 1
        if batches >= max_batches:
            break
    elapsed = time.perf_counter() - start

    return {
        "split": split_name,
        "samples_processed": samples,
        "batches_processed": batches,
        "elapsed_seconds": round(elapsed, 3),
        "samples_per_second": round(samples / elapsed, 1) if elapsed > 0 else 0,
    }


# ═════════════════════════════════════════════════════════════════════
# MAIN — Run all Day 4 verifications when executed directly
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 70)
    print("MediScan — Day 4: DataLoader Pipeline Verification")
    print("=" * 70)

    # 1. Create DataLoaders
    print("\n[1/7] Creating DataLoaders ...")
    loaders = create_dataloaders()
    for name, loader in loaders.items():
        ds = loader.dataset
        print(f"  {name:>5}: {len(ds):>6} samples | batch_size={loader.batch_size} | "
              f"shuffle={'Yes' if isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler) else 'No'}")

    # 2. Dataset accounting
    print("\n[2/7] Dataset accounting ...")
    acct = verify_dataset_accounting(loaders)
    print(f"  Success: {acct['success']}")
    for k, v in acct["actual"].items():
        print(f"    {k}: {v} (expected {acct['expected'][k]})")
    print(f"    Total: {acct['total_actual']} (expected {acct['total_expected']})")
    if acct["errors"]:
        for e in acct["errors"]:
            print(f"    ERROR: {e}")

    # 3. Class label verification
    print("\n[3/7] Class label verification ...")
    cls_result = verify_class_labels(loaders)
    print(f"  Success: {cls_result['success']}")
    print(f"  Mapping: {cls_result['canonical_mapping']}")
    if cls_result["errors"]:
        for e in cls_result["errors"]:
            print(f"    ERROR: {e}")

    # 4. Image path verification
    print("\n[4/7] Image path verification ...")
    img_result = verify_image_paths()
    print(f"  Success: {img_result['success']}")
    print(f"  Images on disk: {img_result.get('total_available_on_disk', 'N/A')}")
    if img_result.get("missing"):
        for m in img_result["missing"][:10]:
            print(f"    MISSING: {m}")
    if img_result.get("counts"):
        for k, v in img_result["counts"].items():
            print(f"    {k}: {v} images in CSV")

    # 5. Batch verification
    print("\n[5/7] Batch verification ...")
    for name in ("train", "val", "test"):
        bv = verify_batch(loaders[name], split_name=name)
        status = "PASS" if bv["success"] else "FAIL"
        print(f"  {name:>5} [{status}]: shape={bv.get('image_shape')} "
              f"labels=[{bv.get('label_min')}, {bv.get('label_max')}] "
              f"batch_size={bv.get('batch_size')}")
        if bv["errors"]:
            for e in bv["errors"]:
                print(f"    ERROR: {e}")

    # 6. Performance benchmark
    print("\n[6/7] DataLoader performance benchmark ...")
    for name in ("train", "val", "test"):
        bm = benchmark_dataloader(loaders[name], max_batches=10, split_name=name)
        print(f"  {name:>5}: {bm['samples_processed']} samples in "
              f"{bm['elapsed_seconds']}s = {bm['samples_per_second']} samples/sec "
              f"({bm['batches_processed']} batches)")

    # 7. Shuffle verification
    print("\n[7/7] Shuffle behaviour ...")
    for name, loader in loaders.items():
        is_random = isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler)
        expected = (name == "train")
        status = "PASS" if is_random == expected else "FAIL"
        print(f"  {name:>5}: shuffle={'Yes' if is_random else 'No'} "
              f"(expected {'Yes' if expected else 'No'}) [{status}]")

    print("\n" + "=" * 70)
    print("Day 4 verification complete.")
    print("=" * 70)
