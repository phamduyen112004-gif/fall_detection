"""
Data Preprocessing Pipeline for Fall Detection.
============================================

Converts raw videos from CaucaFall and MCFD datasets into preprocessed
.npy feature matrices using YOLOv11n-Pose keypoint extraction.

Output Shape: (60, 60) per video segment
    - 60 frames (temporal dimension)
    - 60 features per frame (51 keypoint values + 9 geometric angles)

Datasets: CaucaFall, MCFD only. Legacy datasets excluded.

Usage:
    python main.py --mode preprocess
    # or
    from src.data_prep import run_preprocessing; run_preprocessing()
"""

from __future__ import annotations

import gc
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.config import (
    CAUCAFALL_DIR,
    MCFD_CSV,
    MCFD_DIR,
    MODEL_SAVE_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    TARGET_FRAMES,
    YOLO_MODEL,
    CONF_THRESHOLD,
    MAX_FRAMES,
)
from src.pifr_features import COCO_IDX, extract_keypoints, compute_pifr
from src.utils import standardize_temporal_dim


# =============================================================================
# MODULE-LEVEL LOGGING
# =============================================================================

def _setup_module_logger(name: str) -> logging.Logger:
    """
    Configure and return a module-level logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured logger instance with file and console handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


_logger: logging.Logger = _setup_module_logger(__name__)


# =============================================================================
# INITIALIZATION
# =============================================================================

for _dir in (OUTPUT_DIR, MODEL_SAVE_DIR, RESULTS_DIR):
    Path(_dir).mkdir(parents=True, exist_ok=True)


# =============================================================================
# TEMPORAL STANDARDIZATION (placeholder for future refactoring)
# =============================================================================

# Note: standardize_temporal_dim is imported from src.utils
# This section reserved for any module-level temporal processing utilities


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def process_video_full(
    video_path: Path | str,
    model: YOLO,
    fallback: np.ndarray,
) -> Optional[List[np.ndarray]]:
    """
    Process an entire video and extract PIFR features for all frames.

    Opens the video file, extracts keypoints for each frame using YOLO-Pose,
    computes the 60-dimensional PIFR feature vector, and returns a list
    of features for temporal standardization downstream.

    Args:
        video_path: Path to the input video file (supports .avi, .mp4, etc.).
        model: YOLO pose estimation model for keypoint extraction.
        fallback: Zero vector of shape (60,) used when no person is detected.

    Returns:
        Optional[List[np.ndarray]]: List of 60D feature vectors, one per frame,
        or None if the video cannot be opened.

    Raises:
        No explicit exceptions - failures are logged and return None.

    Example:
        >>> model = YOLO("yolo11n-pose.pt")
        >>> fallback = np.zeros(60, dtype=np.float32)
        >>> features = process_video_full("video.avi", model, fallback)
        >>> if features:
        ...     print(f"Extracted {len(features)} frames")
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _logger.warning("Could not open video: %s", video_path)
        return None

    features: List[np.ndarray] = []
    prev: Optional[np.ndarray] = fallback

    try:
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret:
                break

            h: int
            w: int
            h, w = frame.shape[:2]
            kpts: Optional[np.ndarray] = extract_keypoints(frame, model)

            if kpts is None:
                features.append(
                    prev.copy() if prev is not None
                    else np.zeros(60, dtype=np.float32)
                )
            else:
                vec: np.ndarray = compute_pifr(kpts, w, h)
                features.append(vec)
                prev = vec

    finally:
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
) -> Optional[List[np.ndarray]]:
    """
    Process a specific segment of a video [start, end] frames.

    Seeks to the specified start frame and processes frames until reaching
    the end frame, extracting PIFR features for each. Useful for datasets
    with per-segment annotations like MCFD.

    Args:
        video_path: Path to the input video file.
        start: Start frame index (inclusive, 0-based).
        end: End frame index (inclusive).
        model: YOLO pose estimation model.
        fallback: Zero vector (60D) for frames without detection.

    Returns:
        Optional[List[np.ndarray]]: List of 60D feature vectors for the
        segment, or None if the video cannot be opened.

    Raises:
        No explicit exceptions - failures are logged and return None.

    Example:
        >>> features = process_video_segment("video.avi", start=0, end=59, model=m, fallback=f)
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        _logger.warning("Could not open video: %s", video_path)
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    features: List[np.ndarray] = []
    prev: Optional[np.ndarray] = fallback
    cur: int = start

    try:
        while True:
            ret: bool
            frame: np.ndarray
            ret, frame = cap.read()
            if not ret or cur > end:
                break

            h: int
            w: int
            h, w = frame.shape[:2]
            kpts: Optional[np.ndarray] = extract_keypoints(frame, model)

            if kpts is None:
                features.append(
                    prev.copy() if prev is not None
                    else np.zeros(60, dtype=np.float32)
                )
            else:
                vec: np.ndarray = compute_pifr(kpts, w, h)
                features.append(vec)
                prev = vec

            cur += 1

    finally:
        cap.release()
        del cap
        gc.collect()

    return features if features else None


# =============================================================================
# DATASET: CAUCAFALL
# =============================================================================

def _is_fall(action_name: str) -> int:
    """
    Determine binary label from Caucasus Fall dataset action directory name.

    Args:
        action_name: Directory name containing the action description.

    Returns:
        int: 1 if action contains "fall" (case-insensitive), else 0.

    Example:
        >>> _is_fall("Standing")
        0
        >>> _is_fall("Fall_Forward")
        1
    """
    return 1 if "fall" in action_name.lower() else 0


def _safe_name(s: str) -> str:
    """
    Sanitize a string for safe use in filenames.

    Replaces problematic characters (dots, spaces, slashes) with underscores
    to ensure cross-platform filename compatibility.

    Args:
        s: Input string to sanitize.

    Returns:
        str: Sanitized string safe for use in filenames.

    Example:
        >>> _safe_name("Subject.1/Walking/")
        'Subject_1_Walking_'
    """
    return s.replace(".", "_").replace(" ", "_").replace("/", "_").replace("\\", "_")


def _process_video_item(
    item: Dict[str, Any],
    model: YOLO,
    zero_fallback: np.ndarray,
) -> Tuple[bool, bool]:
    """
    Process a single video item and save features to disk.

    Extracts features from the video, standardizes temporal dimension,
    and saves X (features) and y (label) as .npy files.

    Args:
        item: Dictionary with keys: path, subject, action, label.
        model: YOLO pose model.
        zero_fallback: Fallback vector for missing detections.

    Returns:
        Tuple[bool, bool]: (success, skipped) indicating whether processing
        succeeded or the file was skipped due to existing output.
    """
    x_name: str = f"X_cauca_{_safe_name(item['subject'])}_{_safe_name(item['action'])}.npy"
    y_name: str = f"y_cauca_{_safe_name(item['subject'])}_{_safe_name(item['action'])}.npy"
    x_path: Path = OUTPUT_DIR / x_name
    y_path: Path = OUTPUT_DIR / y_name

    # Skip if already processed
    if x_path.exists() and y_path.exists():
        return False, True

    raw_features: Optional[List[np.ndarray]] = None
    try:
        raw_features = process_video_full(item["path"], model, zero_fallback)
        if raw_features is None or len(raw_features) == 0:
            _logger.debug("No features extracted from: %s", item["path"])
            return False, False

        features: np.ndarray = standardize_temporal_dim(
            np.array(raw_features), TARGET_FRAMES, MAX_FRAMES
        )
        np.save(x_path, features)
        np.save(y_path, np.array([item["label"]], dtype=np.int32))
        return True, False

    except Exception as e:
        _logger.error("Error processing %s: %s", item["path"], e)
        _logger.debug("Traceback:\n%s", traceback.format_exc())
        return False, False

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def process_caucafall(model: YOLO) -> Tuple[int, int, int]:
    """
    Process all videos in the CaucaFall dataset.

    Scans the directory structure (Subject.*/Action/*.avi), extracts
    PIFR features for each video, and saves preprocessed .npy files.
    Labels are derived from the action directory name.

    Args:
        model: YOLO pose model for keypoint extraction.

    Returns:
        Tuple[int, int, int]: Counts of (processed, skipped, error) videos.

    Note:
        CaucaFall structure: Subject.*/Action/*.avi
        Label: "fall" in Action name → 1, otherwise 0
    """
    _logger.info("=" * 50)
    _logger.info("CaucaFall Dataset Processing")
    _logger.info("=" * 50)

    zero_fallback: np.ndarray = np.zeros(60, dtype=np.float32)
    video_list: List[Dict[str, Any]] = []

    if not CAUCAFALL_DIR.exists():
        _logger.error("CaucaFall directory not found: %s", CAUCAFALL_DIR)
        return 0, 0, 0

    # Build video list from directory structure
    for subj_dir in sorted(os.listdir(CAUCAFALL_DIR)):
        subj_path: Path = CAUCAFALL_DIR / subj_dir
        if not subj_path.is_dir() or not subj_dir.startswith("Subject."):
            continue

        for action_dir in sorted(os.listdir(subj_path)):
            action_path: Path = subj_path / action_dir
            if not action_path.is_dir():
                continue

            for video in os.listdir(action_path):
                if not video.endswith(".avi"):
                    continue

                video_list.append({
                    "path": action_path / video,
                    "subject": subj_dir,
                    "action": action_dir,
                    "label": _is_fall(action_dir),
                })

    _logger.info("Found %d videos to process", len(video_list))

    processed: int = 0
    skipped: int = 0
    errors: int = 0

    for item in tqdm(video_list, desc="CaucaFall", unit="video"):
        success, was_skipped = _process_video_item(item, model, zero_fallback)

        if success:
            processed += 1
        elif was_skipped:
            skipped += 1
        else:
            errors += 1

    _logger.info(
        "CaucaFall Summary: Processed=%d, Skipped=%d, Errors=%d",
        processed, skipped, errors
    )
    return processed, skipped, errors


# =============================================================================
# DATASET: MCFD
# =============================================================================

def _process_mcfd_item(
    row: pd.Series,
    idx: int,
    model: YOLO,
    zero_fallback: np.ndarray,
) -> Tuple[bool, bool]:
    """
    Process a single MCFD annotation row and save features to disk.

    Args:
        row: Pandas Series with columns: chute, cam, start, end, label.
        idx: DataFrame index for naming output files.
        model: YOLO pose model.
        zero_fallback: Fallback vector for missing detections.

    Returns:
        Tuple[bool, bool]: (success, skipped) flags.
    """
    chute: int = int(row["chute"])
    cam: int = int(row["cam"])
    start: int = int(row["start"])
    end: int = int(row["end"])
    label: int = int(row["label"])

    video_path: Path = MCFD_DIR / f"chute{chute:02d}" / f"cam{cam}.avi"

    if not video_path.exists():
        _logger.debug("Video not found: %s", video_path)
        return False, False

    x_name: str = f"X_mcfd_c{chute:02d}_cam{cam}_row{idx}.npy"
    y_name: str = f"y_mcfd_c{chute:02d}_cam{cam}_row{idx}.npy"
    x_path: Path = OUTPUT_DIR / x_name
    y_path: Path = OUTPUT_DIR / y_name

    if x_path.exists() and y_path.exists():
        return False, True

    try:
        raw_features: Optional[List[np.ndarray]] = process_video_segment(
            video_path, start, end, model, zero_fallback
        )
        if raw_features is None or len(raw_features) == 0:
            _logger.debug("No features extracted from: %s", video_path)
            return False, False

        features: np.ndarray = standardize_temporal_dim(
            np.array(raw_features), TARGET_FRAMES, MAX_FRAMES
        )
        np.save(x_path, features)
        np.save(y_path, np.array([label], dtype=np.int32))
        return True, False

    except Exception as e:
        _logger.error("Error processing %s: %s", video_path, e)
        _logger.debug("Traceback:\n%s", traceback.format_exc())
        return False, False

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def process_mcfd(model: YOLO) -> Tuple[int, int, int]:
    """
    Process all segments in the MCFD (Multiple Cameras Fall Dataset).

    Reads annotations from data_tuple3.csv, which specifies video paths,
    frame ranges, and labels for each fall/non-fall segment.

    Args:
        model: YOLO pose model for keypoint extraction.

    Returns:
        Tuple[int, int, int]: Counts of (processed, skipped, error) segments.

    Note:
        MCFD structure: chute{nn}/cam{n}.avi with [start, end] frame ranges
        CSV columns: chute, cam, start, end, label
    """
    _logger.info("=" * 50)
    _logger.info("MCFD Dataset Processing")
    _logger.info("=" * 50)

    zero_fallback: np.ndarray = np.zeros(60, dtype=np.float32)

    if not MCFD_CSV.exists():
        _logger.error("MCFD CSV not found: %s", MCFD_CSV)
        return 0, 0, 0

    df: pd.DataFrame = pd.read_csv(MCFD_CSV)
    _logger.info("Found %d annotations to process", len(df))

    processed: int = 0
    skipped: int = 0
    errors: int = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="MCFD", unit="segment"):
        success, was_skipped = _process_mcfd_item(row, idx, model, zero_fallback)

        if success:
            processed += 1
        elif was_skipped:
            skipped += 1
        else:
            errors += 1

    _logger.info(
        "MCFD Summary: Processed=%d, Skipped=%d, Errors=%d",
        processed, skipped, errors
    )
    return processed, skipped, errors


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_preprocessing() -> Optional[Dict[str, int]]:
    """
    Run the full preprocessing pipeline for CaucaFall and MCFD datasets.

    Loads the YOLO pose model once and processes both datasets,
    logging progress and a final summary.

    Returns:
        Optional[Dict[str, int]]: Dictionary with keys 'processed', 'skipped',
        and 'errors' totaling across both datasets, or None on critical failure.

    Example:
        >>> result = run_preprocessing()
        >>> if result:
        ...     print(f"Total processed: {result['processed']}")
    """
    _logger.info("=" * 60)
    _logger.info("PREPROCESSING: CaucaFall + MCFD")
    _logger.info("Output shape: (%d, 60) for ALL files", TARGET_FRAMES)
    _logger.info("Output directory: %s", OUTPUT_DIR)
    _logger.info("=" * 60)

    _logger.info("Loading YOLOv11n-Pose model...")
    model: YOLO = YOLO(YOLO_MODEL)

    p1: int
    s1: int
    e1: int
    p1, s1, e1 = process_caucafall(model)

    p2: int
    s2: int
    e2: int
    p2, s2, e2 = process_mcfd(model)

    _logger.info("=" * 60)
    _logger.info("PREPROCESSING COMPLETE")
    _logger.info("  Processed: %d", p1 + p2)
    _logger.info("  Skipped:   %d", s1 + s2)
    _logger.info("  Errors:    %d", e1 + e2)
    _logger.info("  Output:   %s", OUTPUT_DIR)
    _logger.info("=" * 60)

    return {
        "processed": p1 + p2,
        "skipped": s1 + s2,
        "errors": e1 + e2,
    }


if __name__ == "__main__":
    run_preprocessing()
