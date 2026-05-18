"""
Utility functions for Fall Detection Project.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_gpu() -> dict:
    """Check GPU availability and return system info."""
    result = {
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": None,
        "gpu_memory_gb": 0.0,
    }
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            result["gpu_name"] = props.name
            result["gpu_memory_gb"] = props.total_memory / (1024 ** 3)
        except Exception:
            pass
    return result


def load_npy_files(directory: str | os.PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all preprocessed .npy feature files from a directory.

    Looks for pairs of X_*.npy (features) and y_*.npy (labels).

    Args:
        directory: Path to the directory containing .npy files.

    Returns:
        tuple[np.ndarray, np.ndarray]: Stacked feature array X and label array y.
    """
    directory = os.fspath(directory)

    X_files = sorted(
        f for f in os.listdir(directory)
        if f.startswith("X_") and f.endswith(".npy")
    )
    y_files = sorted(
        f for f in os.listdir(directory)
        if f.startswith("y_") and f.endswith(".npy")
    )

    X = np.array([np.load(os.path.join(directory, f)) for f in X_files])
    y = np.array([np.load(os.path.join(directory, f)).item() for f in y_files])

    return X, y


def safe_arccos(value: float) -> float:
    """Safe arccos with clipping to prevent NaN."""
    clipped = np.clip(value, -1.0, 1.0)
    if np.isnan(clipped) or np.isinf(clipped):
        return 0.0
    return np.arccos(clipped)


def normalize_keypoints(
    keypoints: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Normalize keypoints by frame dimensions."""
    normalized = np.zeros_like(keypoints)
    normalized[:, 0] = keypoints[:, 0] / width
    normalized[:, 1] = keypoints[:, 1] / height
    normalized[:, 2] = keypoints[:, 2]
    return normalized


def standardize_temporal_dim(
    video_features: np.ndarray,
    target_frames: int = 60,
    max_frames: int = 120,
) -> np.ndarray:
    """
    Standardize video features to exact shape (target_frames, 60).

    Pipeline:
        1. Truncate to first max_frames if longer.
        2. Subsample every 2nd frame → max 60 frames.
        3. Pad with last frame if shorter than 60.

    Args:
        video_features: Feature array of shape (N, 60).
        target_frames: Desired number of frames (default: 60).
        max_frames: Maximum frames to consider (default: 120).

    Returns:
        Standardized array of shape (target_frames, 60).
    """
    if video_features is None or len(video_features) == 0:
        return np.zeros((target_frames, 60), dtype=np.float32)

    video_features = np.array(video_features, dtype=np.float32)

    if len(video_features) > max_frames:
        video_features = video_features[:max_frames]

    video_features = video_features[::2]

    if len(video_features) < target_frames:
        last_frame = video_features[-1]
        padding = np.tile(last_frame, (target_frames - len(video_features), 1))
        video_features = np.vstack([video_features, padding])

    return video_features


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
