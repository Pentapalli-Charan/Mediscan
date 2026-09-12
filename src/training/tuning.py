"""
MediScan — Day 8: Controlled Hyperparameter Tuning & Evaluation Module

Provides modular, reproducible infrastructure for running controlled experiments
against the baseline EfficientNet-B0 model.

Key Guarantees:
- Every experiment starts from the identical ImageNet pretrained weights.
- Strictly protects the test set (test split is NEVER loaded or evaluated).
- Independent checkpoint, history CSV, and curves for every experiment.
- Deterministic random seeding (seed=42).
- Pure unweighted CrossEntropyLoss (no class weights, no focal loss).
"""

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataloader import create_dataloaders
from src.data.preprocessing_day3 import HAM10000Dataset, get_augmentation_transform, load_config
from src.models import build_model, count_parameters
from src.training.trainer import EarlyStopping, save_training_checkpoint
from src.utils.device import get_device, get_device_info
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def build_tuning_model(config: Dict[str, Any], dropout_rate: Optional[float] = None) -> nn.Module:
    """
    Build a fresh EfficientNet-B0 with ImageNet pretrained weights and optional custom dropout.
    Backbone is frozen; only classifier is trainable.
    """
    cfg_copy = copy.deepcopy(config)
    if dropout_rate is not None:
        cfg_copy.setdefault("model", {})["dropout"] = dropout_rate

    model = build_model(cfg_copy)
    return model


def build_tuning_dataloaders(
    config: Dict[str, Any],
    project_root: Path,
    custom_aug_config: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders with optional custom training augmentation.
    STRICT TEST SET PROTECTION: The test DataLoader is NEVER created or returned.
    """
    cfg_copy = copy.deepcopy(config)
    if custom_aug_config:
        cfg_copy["augmentation"] = {**cfg_copy.get("augmentation", {}), **custom_aug_config}

    splits_dir = project_root / "reports" / "data"
    image_base_dir = str(project_root / cfg_copy.get("paths", {}).get("data_raw", "data/raw"))
    dl_cfg = cfg_copy.get("dataloader", {})
    batch_size = dl_cfg.get("batch_size", 32)
    num_workers = dl_cfg.get("num_workers", 2)
    pin_memory = dl_cfg.get("pin_memory", True)

    train_csv = str(splits_dir / "split_train.csv")
    val_csv = str(splits_dir / "split_val.csv")

    train_transform = get_augmentation_transform(cfg_copy, mode="train")
    val_transform = get_augmentation_transform(cfg_copy, mode="val")

    train_dataset = HAM10000Dataset(train_csv, image_base_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_csv, image_base_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


def run_experiment(
    experiment_id: int,
    experiment_name: str,
    config_path: str = "config/config.yaml",
    project_root: Optional[Path] = None,
    lr_override: Optional[float] = None,
    dropout_override: Optional[float] = None,
    aug_override: Optional[Dict[str, Any]] = None,
    max_epochs_override: Optional[int] = None,
    patience_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a single controlled hyperparameter tuning experiment.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
    cfg = load_config(str(root / config_path))

    # Reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device_str = get_device()
    device = torch.device(device_str)
    if os.cpu_count() and not torch.cuda.is_available():
        torch.set_num_threads(min(os.cpu_count(), 12))

    logger.info("=" * 65)
    logger.info(f"STARTING EXPERIMENT {experiment_id}: {experiment_name.upper()}")
    logger.info("=" * 65)

    # 1. Hyperparameters
    lr = lr_override if lr_override is not None else cfg.get("training", {}).get("learning_rate", 1e-3)
    dropout = dropout_override if dropout_override is not None else cfg.get("model", {}).get("dropout", 0.2)
    max_epochs = max_epochs_override if max_epochs_override is not None else cfg.get("training", {}).get("epochs", 15)
    patience = patience_override if patience_override is not None else cfg.get("training", {}).get("early_stopping", {}).get("patience", 3)

    logger.info(f"Configuration: LR={lr:.1e} | Dropout={dropout} | Max Epochs={max_epochs} | ES Patience={patience}")

    # 2. Build Model
    model = build_tuning_model(cfg, dropout_rate=dropout).to(device)

    # 3. DataLoaders (Train & Val ONLY)
    train_loader, val_loader = build_tuning_dataloaders(cfg, root, custom_aug_config=aug_override)
    logger.info(f"Loaded train ({len(train_loader)} batches) & val ({len(val_loader)} batches). Test set is PROTECTED.")

    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg.get("training", {}).get("weight_decay", 0.0),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    early_stopping = EarlyStopping(
        patience=patience,
        min_delta=0.001,
        mode="min",
    )

    # 5. Output file paths
    ckpt_dir = root / "models" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / f"day8_exp{experiment_id}_{experiment_name}_best.pth"

    data_dir = root / "reports" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    history_csv_path = data_dir / f"day8_history_exp{experiment_id}.csv"

    figures_dir = root / "reports" / "figures" / "day8"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 6. Training Loop
    history = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
        "epoch_time_seconds": [],
    }

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_loss_epoch = 0
    highest_val_acc = 0.0
    highest_acc_epoch = 0
    early_stopped = False
    stopped_at_epoch = max_epochs

    total_start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(epoch_val_loss)

        # Record history
        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_accuracy"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_accuracy"].append(round(epoch_val_acc, 4))
        history["learning_rate"].append(current_lr)
        history["epoch_time_seconds"].append(round(epoch_time, 2))

        logger.info(
            f"Exp {experiment_id} | Epoch {epoch:02d}/{max_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
            f"LR: {current_lr:.1e} | Time: {epoch_time:.1f}s"
        )

        # Track highest accuracy
        if epoch_val_acc > highest_val_acc:
            highest_val_acc = epoch_val_acc
            highest_acc_epoch = epoch

        # Save checkpoint if val_loss improved
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_loss_epoch = epoch
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=epoch_val_loss,
                val_acc=epoch_val_acc,
                checkpoint_path=best_ckpt_path,
                config_dict=cfg,
            )
            logger.info(f"Checkpoint saved: val_loss improved to {best_val_loss:.4f}")

        # Early stopping
        if early_stopping.step(epoch_val_loss, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch} (patience={early_stopping.patience})")
            early_stopped = True
            stopped_at_epoch = epoch
            break

    total_duration = time.time() - total_start_time

    # Save History CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Saved experiment history to: {history_csv_path}")

    # Plot Individual Experiment Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_arr = history_df["epoch"].values

    ax1.plot(epochs_arr, history_df["train_loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs_arr, history_df["val_loss"], "s-", label="Val Loss", color="#ff7f0e")
    ax1.axvline(x=best_loss_epoch, color="green", linestyle="--", alpha=0.6, label=f"Best Loss (Ep {best_loss_epoch})")
    ax1.set_title(f"Exp {experiment_id} ({experiment_name}): Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(epochs_arr, history_df["train_accuracy"] * 100, "o-", label="Train Acc (%)", color="#2ca02c")
    ax2.plot(epochs_arr, history_df["val_accuracy"] * 100, "s-", label="Val Acc (%)", color="#d62728")
    ax2.axvline(x=highest_acc_epoch, color="purple", linestyle="--", alpha=0.6, label=f"Peak Acc (Ep {highest_acc_epoch})")
    ax2.set_title(f"Exp {experiment_id} ({experiment_name}): Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    fig.tight_layout()
    curve_path = figures_dir / f"exp{experiment_id}_{experiment_name}_curves.png"
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)

    result_dict = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "learning_rate": lr,
        "dropout": dropout,
        "augmentation_configuration": "custom" if aug_override else "baseline",
        "best_epoch": best_loss_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "val_accuracy_at_best_loss": round(best_val_acc, 4),
        "highest_val_accuracy": round(highest_val_acc, 4),
        "epoch_of_highest_val_accuracy": highest_acc_epoch,
        "total_epochs": stopped_at_epoch,
        "early_stopped": early_stopped,
        "training_duration": round(total_duration, 2),
        "average_epoch_duration": round(total_duration / len(history_df), 2),
        "device": device_str,
        "checkpoint_path": str(best_ckpt_path),
        "history_csv_path": str(history_csv_path),
        "curve_path": str(curve_path),
    }

    logger.info(f"FINISHED EXPERIMENT {experiment_id}: Best Val Loss = {best_val_loss:.4f} (Epoch {best_loss_epoch})")
    return result_dict
