#!/usr/bin/env python
"""
Training Script for Hybrid Fall Detection Transformer

Implements:
- Custom PyTorch Dataset with online augmentation
- Full training loop with early stopping
- Model checkpointing and logging
"""

from __future__ import annotations

import gc
import json
import logging
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.config import (
    DATA_DIR,
    LOG_DIR,
    MODEL_SAVE_DIR,
    TrainingConfig,
)
from src.hybrid_transformer import HybridFallTransformer


# =============================================================================
# EPSILON FOR NUMERICAL STABILITY
# =============================================================================

EPSILON: float = 1e-8
"""Small constant added to denominators to prevent division by zero."""


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: str) -> logging.Logger:
    """
    Configure timestamped logging to file and console.

    Args:
        log_dir: Directory to save log files.

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =============================================================================
# CUSTOM DATASET
# =============================================================================

class FallDataset(Dataset):
    """
    PyTorch Dataset for Fall Detection with online augmentation.

    Args:
        X: Feature arrays of shape (N, num_frames, features).
        y: Binary labels (0 = non-fall, 1 = fall).
        augment: Whether to apply data augmentation.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample with optional augmentation."""
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply online augmentation: Gaussian Noise + Temporal Masking."""
        noise = torch.randn_like(x) * TrainingConfig.noise_std
        x = x + noise

        num_frames = x.shape[0]
        num_mask = int(num_frames * TrainingConfig.mask_ratio)

        if num_mask > 0:
            mask_indices = torch.randperm(num_frames)[:num_mask]
            x[mask_indices] = 0

        return x


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load all preprocessed .npy files from directory.

    Args:
        data_dir: Directory containing X_*.npy and y_*.npy files.

    Returns:
        tuple: (X, y) arrays.

    Raises:
        FileNotFoundError: If no data files found.
    """
    try:
        X_files = sorted([f for f in os.listdir(data_dir) if f.startswith("X_")])
        y_files = sorted([f for f in os.listdir(data_dir) if f.startswith("y_")])

        if len(X_files) == 0:
            raise FileNotFoundError(f"No X_*.npy files found in {data_dir}")

        X = np.array([np.load(os.path.join(data_dir, f)) for f in X_files])
        y = np.array([np.load(os.path.join(data_dir, f)).item() for f in y_files])

        return X, y

    except Exception as e:
        logging.error("Failed to load data from %s: %s", data_dir, e)
        raise


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders with proper splitting.

    Args:
        X: Feature arrays.
        y: Binary labels.
        config: Training configuration.

    Returns:
        tuple: (train_loader, val_loader, test_loader).
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=config.val_size,
        random_state=42, stratify=y_train_val
    )

    train_dataset = FallDataset(X_train, y_train, augment=True)
    val_dataset = FallDataset(X_val, y_val, augment=False)
    test_dataset = FallDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute classification metrics with division-by-zero protection.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Prediction probabilities (optional).

    Returns:
        dict: Dictionary of computed metrics.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 1:
            tn = int(cm[0, 0]) if cm.shape == (1, 1) else 0
            fp, fn, tp = 0, 0, 0
        else:
            tn, fp, fn, tp = cm.ravel()
    except Exception:
        tn, fp, fn, tp = 0, 0, 0, 0

    precision_denom = tp + fp + EPSILON
    recall_denom = tp + fn + EPSILON

    metrics: dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": tp / precision_denom,
        "recall": tp / recall_denom,
        "f1": 2 * tp / (2 * tp + fp + fn + EPSILON),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }

    if y_prob is not None and len(set(y_true)) > 1:
        try:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc"] = 0.0
    else:
        metrics["auc"] = 0.0

    return metrics


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> dict[str, Any]:
    """
    Train for one epoch and return metrics.

    Args:
        model: The neural network model.
        loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Compute device.

    Returns:
        dict: Training metrics for the epoch.
    """
    model.train()

    total_loss: float = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    num_batches: int = 0

    try:
        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()
            num_batches += 1

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y_batch.detach().cpu().numpy().tolist())

            del X_batch, y_batch, outputs, loss, probs, preds

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:
        logging.error("Error during training batch: %s", e)
        raise

    if num_batches == 0:
        return {
            "loss": 0.0, "accuracy": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]], "auc": 0.0
        }

    avg_loss = total_loss / num_batches
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = avg_loss

    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """
    Evaluate model on validation/test set and return metrics.

    Args:
        model: The neural network model.
        loader: Evaluation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        dict: Evaluation metrics.
    """
    model.eval()

    total_loss: float = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[float] = []
    num_batches: int = 0

    try:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)

            total_loss += loss.detach().cpu().item()
            num_batches += 1

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(y_batch.detach().cpu().numpy().tolist())
            all_probs.extend(probs.detach().cpu().numpy().tolist())

            del X_batch, y_batch, outputs, loss, probs, preds

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as e:
        logging.error("Error during evaluation: %s", e)
        raise

    if num_batches == 0:
        return {
            "loss": 0.0, "accuracy": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]], "auc": 0.0
        }

    avg_loss = total_loss / num_batches
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = avg_loss

    return metrics


def train_model(config: TrainingConfig, logger: logging.Logger) -> dict[str, Any]:
    """
    Main training function with memory management.

    Args:
        config: Training configuration.
        logger: Logger instance.

    Returns:
        dict: Final test metrics.

    Raises:
        RuntimeError: If training fails.
    """
    try:
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        os.makedirs(config.resolved_model_dir, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        logger.info("Loading data from: %s", config.resolved_data_dir)
        X, y = load_data(str(config.resolved_data_dir))
        logger.info("Loaded %d samples. Class distribution: %s", len(X), np.bincount(y))

        train_loader, val_loader, test_loader = create_data_loaders(X, y, config)
        logger.info(
            "Data splits - Train: %d, Val: %d, Test: %d",
            len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)
        )

        logger.info(
            "Initializing HybridFallTransformer: d_model=%d, nhead=%d, num_layers=%d",
            config.d_model, config.nhead, config.num_layers
        )

        model = HybridFallTransformer(
            input_dim=config.input_dim,
            num_frames=config.num_frames,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dropout=config.dropout
        ).to(device)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %d", num_params)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", patience=10, factor=0.5, verbose=True
        )

        best_val_f1: float = 0.0
        patience_counter: int = 0
        history: dict[str, list[float]] = {
            "train_loss": [], "train_acc": [], "train_f1": [],
            "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []
        }

        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)

        for epoch in range(config.epochs):
            try:
                train_metrics = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_metrics = evaluate(model, val_loader, criterion, device)

                scheduler.step(val_metrics["f1"])

                history["train_loss"].append(train_metrics["loss"])
                history["train_acc"].append(float(train_metrics["accuracy"]))
                history["train_f1"].append(float(train_metrics["f1"]))
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(float(val_metrics["accuracy"]))
                history["val_f1"].append(float(val_metrics["f1"]))
                history["val_auc"].append(float(val_metrics["auc"]))

                logger.info(
                    "Epoch %3d/%3d | Train Loss: %.4f, Acc: %.4f, F1: %.4f | "
                    "Val Loss: %.4f, Acc: %.4f, F1: %.4f, AUC: %.4f",
                    epoch + 1, config.epochs,
                    train_metrics['loss'], train_metrics['accuracy'], train_metrics['f1'],
                    val_metrics['loss'], val_metrics['accuracy'], val_metrics['f1'],
                    val_metrics['auc']
                )

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = float(val_metrics["f1"])
                    patience_counter = 0

                    checkpoint_path = os.path.join(
                        config.resolved_model_dir, "best_model.pth"
                    )
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info("  >> Saved best model (Val F1: %.4f)", best_val_f1)
                else:
                    patience_counter += 1

                if patience_counter >= config.early_stopping_patience:
                    logger.info("Early stopping triggered at epoch %d", epoch + 1)
                    break

            except RuntimeError as e:
                logger.error("Error in epoch %d: %s", epoch + 1, e)
                raise

            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        logger.info("=" * 60)
        logger.info("Training complete. Loading best model for test evaluation...")

        model.load_state_dict(
            torch.load(
                os.path.join(config.resolved_model_dir, "best_model.pth"),
                weights_only=True
            )
        )

        test_metrics = evaluate(model, test_loader, criterion, device)

        logger.info("=" * 60)
        logger.info("FINAL TEST RESULTS")
        logger.info("=" * 60)
        logger.info("Accuracy:  %.4f", test_metrics['accuracy'])
        logger.info("Precision: %.4f", test_metrics['precision'])
        logger.info("Recall:    %.4f", test_metrics['recall'])
        logger.info("F1 Score:  %.4f", test_metrics['f1'])
        logger.info("AUC-ROC:   %.4f", test_metrics['auc'])
        logger.info("Confusion Matrix: %s", test_metrics['confusion_matrix'])

        results: dict[str, Any] = {
            "test_metrics": {
                k: v for k, v in test_metrics.items()
            },
            "best_val_f1": best_val_f1,
            "config": {
                "d_model": config.d_model,
                "nhead": config.nhead,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "epochs_trained": epoch + 1,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
            }
        }

        with open(os.path.join(config.resolved_log_dir, "metrics.json"), "w") as f:
            json.dump(results, f, indent=2)

        with open(os.path.join(config.resolved_log_dir, "history.json"), "w") as f:
            json.dump(history, f)

        logger.info("Metrics saved to %s/metrics.json", config.resolved_log_dir)
        logger.info("History saved to %s/history.json", config.resolved_log_dir)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return test_metrics

    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        raise
    except ValueError as e:
        logger.error("Invalid configuration: %s", e)
        raise
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_training() -> dict[str, Any]:
    """
    Entry point for training - compatible with main.py.

    Returns:
        dict: Final test metrics.
    """
    _data_dir = str(DATA_DIR)
    _model_dir = str(MODEL_SAVE_DIR)
    _log_dir = str(LOG_DIR)

    config = TrainingConfig(
        data_dir=_data_dir,
        model_dir=_model_dir,
        log_dir=_log_dir
    )
    logger = setup_logging(config.resolved_log_dir)

    logger.info("=" * 60)
    logger.info("HYBRID FALL DETECTION TRANSFORMER - TRAINING")
    logger.info("=" * 60)

    try:
        test_metrics = train_model(config, logger)
        logger.info("Training completed successfully!")
        return test_metrics

    except Exception as e:
        logger.error("Training failed with error: %s", e)
        raise

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    """Main entry point for standalone execution."""
    run_training()


if __name__ == "__main__":
    main()
