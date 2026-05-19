"""
Trainer module for Fall Detection.
Contains dataset, data loading, training and evaluation utilities.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple


# =============================================================================
# DATASET
# =============================================================================

class FallDataset(Dataset):
    """Dataset for fall detection from preprocessed numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.02
        x = x + noise
        x = torch.clamp(x, -3, 3)
        return x


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed data from directory.

    Args:
        data_dir: Directory containing processed numpy files

    Returns:
        Tuple of (X, y) arrays
    """
    X = np.load(os.path.join(data_dir, "X.npy"))
    y = np.load(os.path.join(data_dir, "y.npy"))
    return X, y


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_probs: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


# =============================================================================
# TRAINING & EVALUATION
# =============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader,
                   criterion: nn.Module, optimizer: torch.optim.Optimizer,
                   device: torch.device) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device,
             return_probs: bool = False) -> Dict:
    """
    Evaluate model on validation/test set.

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation
        criterion: Loss function
        device: torch device
        return_probs: If True, return y_probs for threshold calculation

    Returns:
        Dictionary of metrics (and optionally y_probs if return_probs=True)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.unsqueeze(1).to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(y_batch.cpu().numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = total_loss / len(loader)

    if return_probs:
        metrics["y_true"] = all_labels
        metrics["y_probs"] = all_probs

    return metrics


def setup_logging(log_dir: str) -> None:
    """Setup logging to file."""
    import logging
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler()
        ]
    )
