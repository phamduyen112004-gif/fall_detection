"""
Video Processor Module for Fall Detection.

Refactored from scripts/preprocess.py to avoid code duplication.
Provides unified video processing utilities for keypoint extraction
and PIFR feature computation.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import (
    CONF_THRESHOLD,
    TARGET_FRAMES,
    MAX_FRAMES,
)
from src.pifr_features import COCO_IDX, compute_9_pifr_features


# ============================================================
# PIFR Feature Extraction
# ============================================================

def extract_keypoints(
    frame: np.ndarray,
    model: YOLO,
    frame_width: int,
    frame_height: int,
) -> np.ndarray | None:
    """
    Run YOLOv11-Pose on a single frame and extract normalized 17 keypoints.

    Args:
        frame: BGR frame from OpenCV.
        model: Loaded YOLO pose model.
        frame_width: Original frame width for normalization.
        frame_height: Original frame height for normalization.

    Returns:
        np.ndarray | None: Normalized keypoints of shape (17, 3)
        [x/width, y/height, confidence] or None if detection fails.
    """
    results = model(frame, verbose=False, conf=CONF_THRESHOLD)

    if results[0].keypoints is None or len(results[0].keypoints) == 0:
        return None

    keypoints = results[0].keypoints.data[0].cpu().numpy()

    if len(keypoints) < 17:
        return None

    normalized = np.zeros((17, 3), dtype=np.float32)
    for i in range(17):
        x, y, conf = keypoints[i]
        normalized[i, 0] = x / frame_width
        normalized[i, 1] = y / frame_height
        normalized[i, 2] = conf

    return normalized


def extract_pifr_60d(keypoints: np.ndarray) -> np.ndarray:
    """
    Extract 60D PIFR feature vector from keypoints.

    51 keypoint values (17 x 3) + 9 geometric angles = 60D

    Args:
        keypoints: Normalized keypoints of shape (17, 3).

    Returns:
        np.ndarray: 60D feature vector.
    """
    return np.concatenate([keypoints.flatten(), compute_9_pifr_features(keypoints)])


# ============================================================
# Temporal Standardization
# ============================================================

def standardize_to_60x60(video_features: list[np.ndarray] | None) -> np.ndarray:
    """
    Standardize video features to exact shape (60, 60).

    Pipeline:
        1. Truncate to first 120 frames if longer
        2. Subsample every 2nd frame -> max 60 frames
        3. Pad with last frame if shorter than 60

    Args:
        video_features: List of 60D feature vectors.

    Returns:
        np.ndarray: Standardized array of shape (60, 60).
    """
    if video_features is None or len(video_features) == 0:
        return np.zeros((TARGET_FRAMES, 60), dtype=np.float32)

    video_features = np.array(video_features, dtype=np.float32)

    if len(video_features) > MAX_FRAMES:
        video_features = video_features[:MAX_FRAMES]

    video_features = video_features[::2]

    if len(video_features) < TARGET_FRAMES:
        last_frame = video_features[-1]
        padding = np.tile(last_frame, (TARGET_FRAMES - len(video_features), 1))
        video_features = np.vstack([video_features, padding])

    assert video_features.shape == (TARGET_FRAMES, 60), (
        f"Expected shape ({TARGET_FRAMES}, 60), got {video_features.shape}"
    )
    return video_features


# ============================================================
# Video Processing
# ============================================================

def process_video_full(
    video_path: Path | str,
    model: YOLO,
    fallback: np.ndarray,
) -> list[np.ndarray] | None:
    """
    Process an entire video and extract PIFR features for all frames.

    Args:
        video_path: Path to video file.
        model: YOLO pose model.
        fallback: Zero vector for frames with no detection.

    Returns:
        list[np.ndarray] | None: List of 60D feature vectors or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    features: list[np.ndarray] = []
    prev: np.ndarray | None = fallback

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        kpts = extract_keypoints(frame, model, w, h)

        if kpts is None:
            features.append(prev.copy() if prev is not None else np.zeros(60, dtype=np.float32))
        else:
            vec = extract_pifr_60d(kpts)
            features.append(vec)
            prev = vec

    cap.release()
    del cap
    gc.collect()

    return features if features else None


def process_video_segment(
    video_path: Path | str,
    start: int,
    end: int,
    model: YOLO,
    fallback: np.ndarray,
) -> list[np.ndarray] | None:
    """
    Process a specific segment of a video [start, end] frames.

    Args:
        video_path: Path to video file.
        start: Start frame index (inclusive).
        end: End frame index (inclusive).
        model: YOLO pose model.
        fallback: Zero vector for frames with no detection.

    Returns:
        list[np.ndarray] | None: List of 60D feature vectors or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    features: list[np.ndarray] = []
    prev: np.ndarray | None = fallback
    cur = start

    while True:
        ret, frame = cap.read()
        if not ret or cur > end:
            break

        h, w = frame.shape[:2]
        kpts = extract_keypoints(frame, model, w, h)

        if kpts is None:
            features.append(prev.copy() if prev is not None else np.zeros(60, dtype=np.float32))
        else:
            vec = extract_pifr_60d(kpts)
            features.append(vec)
            prev = vec

        cur += 1

    cap.release()
    del cap
    gc.collect()

    return features if features else None


# ============================================================
# Utility Functions
# ============================================================

def is_fall(action_name: str) -> int:
    """Determine label from action directory name."""
    return 1 if "fall" in action_name.lower() else 0


def safe_name(s: str) -> str:
    """Sanitize string for use in filenames."""
    return s.replace(".", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    print("Video Processor Module")
    print("Use: from src.video_processor import process_video_full, process_video_segment")
