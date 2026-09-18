"""
MediScan — Training Pipeline (Day 6)

Implements the complete PyTorch training and validation pipeline:
- CrossEntropyLoss (unweighted baseline)
- Adam optimizer (optimizing classifier head parameters only)
- ReduceLROnPlateau learning rate scheduler (monitored on validation loss)
- Early stopping based on validation loss
- Checkpoint saving on validation loss improvement
- Full tracking of training & validation loss and accuracy
- Plotting loss and accuracy curves
- Machine-readable epoch history logging (CSV)
- Strict test set isolation (test data is NEVER accessed or used)
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data.dataloader import create_dataloaders
from src.data.preprocessing_day3 import load_config
from src.models import build_model, count_parameters
from src.utils.device import get_device, get_device_info
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
# 1. EARLY STOPPING
# ═════════════════════════════════════════════════════════════════════


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    """

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        mode: str = "min",
    ):
        """
        Args:
            patience: Number of epochs to wait with no improvement before stopping.
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            mode: 'min' for loss, 'max' for accuracy.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_epoch = 0

    def step(self, current_metric: float, epoch: int) -> bool:
        """
        Evaluate early stopping condition.

        Args:
            current_metric: Validation metric value (e.g. val_loss).
            epoch: Current epoch index.

        Returns:
            bool: True if early stopping should trigger, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_metric
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = (self.best_score - current_metric) > self.min_delta
        else:
            improved = (current_metric - self.best_score) > self.min_delta

        if improved:
            self.best_score = current_metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience} "
                f"(best: {self.best_score:.4f} at epoch {self.best_epoch})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


# ═════════════════════════════════════════════════════════════════════
# 2. CHECKPOINTING UTILITIES
# ═════════════════════════════════════════════════════════════════════


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    val_loss: float,
    val_acc: float,
    checkpoint_path: Union[str, Path],
    config_dict: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save training checkpoint containing model, optimizer, scheduler, and epoch metadata.

    Args:
        model: Trained PyTorch model.
        optimizer: PyTorch optimizer.
        scheduler: Optional learning rate scheduler.
        epoch: Current epoch index.
        val_loss: Validation loss for this epoch.
        val_acc: Validation accuracy for this epoch.
        checkpoint_path: Destination filepath.
        config_dict: Optional configuration settings.

    Returns:
        Path: Path to saved checkpoint file.
    """
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "checkpoint_type": "trained_best_model",
        "architecture": getattr(model, "architecture", "efficientnet_b0"),
        "num_classes": getattr(model, "num_classes", 7),
        "config": config_dict or {},
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    torch.save(checkpoint, str(path))
    logger.info(f"Checkpoint saved to {path} (Epoch {epoch}, Val Loss: {val_loss:.4f})")
    return path


def load_training_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a saved training checkpoint into a model and optionally restore optimizer/scheduler.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model instance to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map tensors onto.

    Returns:
        dict: Checkpoint metadata dictionary.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(str(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    logger.info(
        f"Loaded checkpoint from {path} (Epoch {checkpoint.get('epoch')}, "
        f"Val Loss: {checkpoint.get('val_loss'):.4f})"
    )
    return checkpoint


# ═════════════════════════════════════════════════════════════════════
# 3. EPOCH TRAINING & VALIDATION LOOPS
# ═════════════════════════════════════════════════════════════════════


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute one full training epoch: forward pass, loss, backward pass, optimizer step.

    Args:
        model: PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target compute device.

    Returns:
        tuple: (train_loss, train_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = running_correct / total_samples if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Execute one full validation pass: forward pass, loss calculation.
    STRICTLY no gradients, NO optimizer steps.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Target compute device.

    Returns:
        tuple: (val_loss, val_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    val_acc = running_correct / total_samples if total_samples > 0 else 0.0
    return val_loss, val_acc


# ═════════════════════════════════════════════════════════════════════
# 4. PRE-TRAINING SANITY CHECK & FREEZING VERIFICATION
# ═════════════════════════════════════════════════════════════════════


def verify_freezing_sanity(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Perform pre-training sanity check:
    1. Forward pass produces finite logits
    2. Loss is finite
    3. Backward pass succeeds
    4. Gradients exist for classifier head parameters
    5. Frozen backbone parameters do NOT receive gradients (grad is None)
    6. Optimizer step succeeds
    7. Classifier parameters changed
    8. Frozen backbone parameters did NOT change

    Args:
        model: Model to verify.
        batch: (images, labels) tuple.
        criterion: Loss criterion.
        optimizer: Optimizer.
        device: Target device.

    Returns:
        dict: Sanity check results.
    """
    model.train()
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)

    # 1. Snapshot parameter tensors before step
    classifier_params_before = [p.clone().detach() for p in model.classifier.parameters()]
    backbone_params_before = [p.clone().detach() for p in model.features.parameters()]

    # 2. Forward pass
    optimizer.zero_grad()
    logits = model(images)
    assert not torch.isnan(logits).any(), "Sanity check failed: NaN in logits"
    assert not torch.isinf(logits).any(), "Sanity check failed: Inf in logits"

    # 3. Loss calculation
    loss = criterion(logits, labels)
    assert not torch.isnan(loss), "Sanity check failed: NaN in loss"
    assert not torch.isinf(loss), "Sanity check failed: Inf in loss"

    # 4. Backward pass
    loss.backward()

    # 5. Verify gradients
    # Backbone features must have grad == None
    backbone_has_no_grad = all(p.grad is None for p in model.features.parameters())
    assert backbone_has_no_grad, "Sanity check failed: Backbone received gradients!"

    # Classifier head must have grad != None and non-zero
    classifier_grads_exist = all(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.classifier.parameters()
    )
    assert classifier_grads_exist, "Sanity check failed: Classifier did not receive gradients!"

    # 6. Optimizer step
    optimizer.step()

    # 7. Verify parameter updates
    classifier_params_after = list(model.classifier.parameters())
    backbone_params_after = list(model.features.parameters())

    classifier_changed = any(
        not torch.equal(before, after)
        for before, after in zip(classifier_params_before, classifier_params_after)
    )
    assert classifier_changed, "Sanity check failed: Classifier parameters did not change after step!"

    backbone_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(backbone_params_before, backbone_params_after)
    )
    assert backbone_unchanged, "Sanity check failed: Backbone parameters changed after step!"

    logger.info("Pre-training sanity check passed: Backbone frozen & Classifier trainable verified.")
    return {
        "success": True,
        "loss": loss.item(),
        "backbone_has_no_grad": backbone_has_no_grad,
        "classifier_grads_exist": classifier_grads_exist,
        "classifier_changed": classifier_changed,
        "backbone_unchanged": backbone_unchanged,
    }


# ═════════════════════════════════════════════════════════════════════
# 5. TRAINING CURVES & LOGGING
# ═════════════════════════════════════════════════════════════════════


def plot_training_curves(
    history: Union[pd.DataFrame, Dict[str, List[Any]]],
    output_dir: Union[str, Path] = "reports/figures",
) -> Tuple[Path, Path]:
    """
    Generate and save:
    1. Training vs Validation Loss Curve
    2. Training vs Validation Accuracy Curve

    Args:
        history: DataFrame or dictionary containing epoch metrics.
        output_dir: Output directory for figures.

    Returns:
        tuple: (loss_curve_path, accuracy_curve_path)
    """
    if isinstance(history, dict):
        df = pd.DataFrame(history)
    else:
        df = history.copy()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    epochs = df["epoch"].values

    # 1. Loss Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, df["train_loss"].values, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax.plot(epochs, df["val_loss"].values, "s-", label="Validation Loss", color="#ff7f0e", linewidth=2)
    ax.set_title("MediScan — Training vs Validation Loss (Day 6 Baseline)", fontsize=13, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("CrossEntropyLoss", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    loss_curve_path = out_path / "day6_loss_curve.png"
    fig.savefig(loss_curve_path, dpi=150)
    plt.close(fig)

    # 2. Accuracy Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, df["train_accuracy"].values * 100, "o-", label="Train Accuracy (%)", color="#2ca02c", linewidth=2)
    ax.plot(epochs, df["val_accuracy"].values * 100, "s-", label="Validation Accuracy (%)", color="#d62728", linewidth=2)
    ax.set_title("MediScan — Training vs Validation Accuracy (Day 6 Baseline)", fontsize=13, pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()
    acc_curve_path = out_path / "day6_accuracy_curve.png"
    fig.savefig(acc_curve_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved training curves: {loss_curve_path}, {acc_curve_path}")
    return loss_curve_path, acc_curve_path


def save_training_history(
    history: Union[pd.DataFrame, Dict[str, List[Any]]],
    output_path: Union[str, Path] = "reports/data/day6_training_history.csv",
) -> Path:
    """
    Save training history to CSV.

    Args:
        history: DataFrame or dictionary of metrics.
        output_path: CSV output filepath.

    Returns:
        Path: Path to saved CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(history, dict):
        df = pd.DataFrame(history)
    else:
        df = history

    df.to_csv(str(path), index=False)
    logger.info(f"Saved training history CSV to: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════
# 6. MAIN TRAINING PIPELINE FUNCTION
# ═════════════════════════════════════════════════════════════════════


def train_model(
    config_path: Union[str, Path] = "config/config.yaml",
    project_root: Optional[Union[str, Path]] = None,
    override_epochs: Optional[int] = None,
    override_lr: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute the Day 6 baseline training run.

    Args:
        config_path: Path to configuration YAML.
        project_root: Optional root directory path.
        override_epochs: Optional override for number of epochs.
        override_lr: Optional override for learning rate.

    Returns:
        dict: Complete results dictionary containing metrics, checkpoint paths, and summaries.
    """
    # Optimize CPU threads if on CPU
    if os.cpu_count() and not torch.cuda.is_available():
        torch.set_num_threads(min(os.cpu_count(), 12))

    # Resolve paths
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent.parent
    cfg_file = Path(config_path) if Path(config_path).is_absolute() else root / config_path
    cfg = load_config(str(cfg_file))

    # 1. Reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # 2. Device
    device_str = get_device()
    device = torch.device(device_str)
    device_info = get_device_info()
    logger.info(f"Training device: {device_str} ({device_info.get('torch_version', '')})")

    # 3. Model construction
    model = build_model(cfg).to(device)

    # Verify only classifier is trainable
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    classifier_params = list(model.classifier.parameters())
    assert len(trainable_params) == len(classifier_params), (
        "Optimizer setup error: Trainable params do not match classifier params!"
    )

    # 4. DataLoaders
    # STRICT TEST SET ISOLATION: Only request train and val loaders
    loaders = create_dataloaders(config_path=str(cfg_file), project_root=str(root))
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    # Explicitly verify test loader is NOT used for training
    logger.info(f"Loaded Train DataLoader: {len(train_loader)} batches")
    logger.info(f"Loaded Validation DataLoader: {len(val_loader)} batches")
    logger.info("Test DataLoader is protected and will NOT be accessed during Day 6 training.")

    # 5. Training configuration
    train_cfg = cfg.get("training", {})
    epochs = override_epochs if override_epochs is not None else train_cfg.get("epochs", 15)
    learning_rate = override_lr if override_lr is not None else train_cfg.get("learning_rate", 1e-3)
    weight_decay = train_cfg.get("weight_decay", 0.0)

    # Loss function (unweighted baseline)
    criterion = nn.CrossEntropyLoss()

    # Optimizer (Adam on trainable parameters only)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Scheduler (ReduceLROnPlateau monitoring val_loss)
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 2),
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    # Early stopping
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 3),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="min",
    )

    # Checkpoint settings
    ckpt_cfg = train_cfg.get("checkpoint", {})
    ckpt_dir = root / ckpt_cfg.get("dir", "models/checkpoints")
    ckpt_filename = ckpt_cfg.get("filename", "efficientnet_b0_best.pth")
    best_checkpoint_path = ckpt_dir / ckpt_filename

    # 6. Sanity check (run on 1 batch from train_loader, then reset state)
    logger.info("Running pre-training sanity check...")
    sample_batch = next(iter(train_loader))
    sanity_results = verify_freezing_sanity(model, sample_batch, criterion, optimizer, device)

    # Reset model & optimizer for clean start of actual epoch 1
    logger.info("Rebuilding clean model & optimizer for actual training run...")
    set_seed(seed)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 2),
        min_lr=sched_cfg.get("min_lr", 1e-6),
    )

    # 7. Main Training Loop
    logger.info("=" * 65)
    logger.info(f"STARTING BASELINE TRAINING RUN ({epochs} EPOCHS CONFIGURED)")
    logger.info(f"Model: EfficientNet-B0 | Device: {device_str} | LR: {learning_rate}")
    logger.info("=" * 65)

    history: Dict[str, List[Any]] = {
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
    best_epoch = 0
    total_training_start = time.time()
    early_stopped = False
    stopped_at_epoch = epochs

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Validate
        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        epoch_duration = time.time() - epoch_start

        # Step learning rate scheduler with validation loss
        scheduler.step(val_loss)

        # Record metrics
        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["train_accuracy"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["learning_rate"].append(current_lr)
        history["epoch_time_seconds"].append(round(epoch_duration, 2))

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"LR: {current_lr:.1e} | Time: {epoch_duration:.1f}s"
        )

        # Checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            save_training_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                checkpoint_path=best_checkpoint_path,
                config_dict=cfg,
            )

        # Early stopping check
        if early_stopping.step(val_loss, epoch):
            logger.info(f"Early stopping triggered at epoch {epoch} (patience={early_stopping.patience})")
            early_stopped = True
            stopped_at_epoch = epoch
            break

    total_training_duration = time.time() - total_training_start

    logger.info("=" * 65)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Total Training Duration: {total_training_duration:.1f}s ({total_training_duration/60:.2f} min)")
    logger.info(f"Best Validation Epoch:   {best_epoch}")
    logger.info(f"Best Validation Loss:    {best_val_loss:.4f}")
    logger.info(f"Best Validation Acc:     {best_val_acc*100:.2f}%")
    logger.info(f"Early Stopped:           {early_stopped} (Epoch {stopped_at_epoch})")
    logger.info("=" * 65)

    # 8. Save training history CSV
    history_csv_path = root / "reports" / "data" / "day6_training_history.csv"
    save_training_history(history, history_csv_path)

    # 9. Plot training curves
    figures_dir = root / "reports" / "figures"
    loss_curve_path, acc_curve_path = plot_training_curves(history, figures_dir)

    # 10. Verify Checkpoint Reloadability
    logger.info(f"Verifying checkpoint reload from: {best_checkpoint_path}")
    fresh_model = build_model(cfg).to(device)
    fresh_model.eval()
    ckpt_meta = load_training_checkpoint(best_checkpoint_path, fresh_model, device=device_str)

    # Verify forward pass with reloaded model
    with torch.no_grad():
        test_fwd = fresh_model(torch.randn(2, 3, 224, 224, device=device))
    assert test_fwd.shape == (2, 7), f"Reloaded model output shape mismatch: {test_fwd.shape}"
    logger.info("Checkpoint reload and forward pass verified successfully.")

    return {
        "success": True,
        "epochs_configured": epochs,
        "epochs_completed": stopped_at_epoch,
        "early_stopped": early_stopped,
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_train_acc": history["train_accuracy"][-1] if history["train_accuracy"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_acc": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "total_duration_seconds": round(total_training_duration, 2),
        "device": device_str,
        "checkpoint_path": str(best_checkpoint_path),
        "history_csv_path": str(history_csv_path),
        "loss_curve_path": str(loss_curve_path),
        "accuracy_curve_path": str(acc_curve_path),
        "sanity_check": sanity_results,
    }
