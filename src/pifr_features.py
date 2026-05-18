"""
PIFR (Pose-based Image Feature Representation) Features Module

Extracts geometric features from human pose keypoints for fall detection.
Implements the full PIFR feature pipeline: 51D COCO keypoints + 9D geometric
angles = 60D total feature vector per frame.
"""

from __future__ import annotations

import logging
from typing import Any, TypeAlias

import cv2
import numpy as np

# Type aliases for readability
Frame: TypeAlias = np.ndarray
Keypoints: TypeAlias = np.ndarray
YOLOModel: TypeAlias = Any
RGBColor: TypeAlias = tuple[int, int, int]

# ============================================================
# MAGIC NUMBER CONSTANTS — Academic Justification
# ============================================================

# Model confidence threshold for YOLO keypoint extraction.
# Set to 0.5 per COCO evaluation protocol — balances precision/recall
# for person keypoint detection in indoor fall scenarios.
YOLO_CONF_THRESHOLD: float = 0.5

# Minimum confidence per individual keypoint for skeleton rendering.
# Suppresses noisy keypoints from YOLO's low-confidence predictions.
KEYPOINT_VISUALIZATION_CONF: float = 0.3

# Epsilon value to prevent division-by-zero when computing confidence-
# weighted center-of-mass (F1, F2). A value of 1e-8 is chosen as it is
# below the float32 numerical precision floor (~1e-7) for single additions,
# ensuring safe addition without altering the original sum meaningfully.
EPSILON: float = 1e-8

# Number of COCO keypoints extracted by YOLOv11-Pose.
# Standard 17-keypoint body model: nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees, ankles.
NUM_COCO_KEYPOINTS: int = 17

# Total PIFR feature dimension: 51D (17 COCO keypoints × 3 channels [x,y,conf])
# + 9D geometric features = 60D per frame.
TOTAL_PIFR_DIM: int = 60

# Dimensionality of the geometric feature sub-vector (F1-F9).
GEOMETRIC_DIM: int = 9

# ============================================================
# COCO KEYPOINT INDEX MAPPING
# ============================================================

COCO_IDX: dict[str, int] = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# Human-readable aliases — used throughout angle computations.
NOSE: int = 0
LEFT_SHOULDER: int = 5
RIGHT_SHOULDER: int = 6
LEFT_HIP: int = 11
RIGHT_HIP: int = 12
LEFT_KNEE: int = 13
RIGHT_KNEE: int = 14
LEFT_ANKLE: int = 15
RIGHT_ANKLE: int = 16

# ============================================================
# LOGGER
# ============================================================

_logger: logging.Logger = logging.getLogger(__name__)


# ============================================================
# KEYPOINT EXTRACTION
# ============================================================

def extract_keypoints(
    frame: Frame,
    yolo_model: YOLOModel,
    fps: float = 30.0,
) -> Keypoints | None:
    """
    Extract 17 COCO keypoints from a single frame using YOLOv11-Pose.

    The YOLO model processes the frame in BGR format (OpenCV native).
    Keypoints are returned in normalized [0, 1] xy coordinates plus
    per-keypoint confidence scores.

    Args:
        frame: BGR image from OpenCV (shape: H × W × 3).
        yolo_model: Loaded YOLO model from ultralytics library.
        fps: Video FPS — unused in this implementation, kept for API
             compatibility with the broader pipeline.

    Returns:
        Keypoints array of shape (17, 3) with [x, y, confidence],
        all normalized to [0, 1] range. Returns None if no person is
        detected or if keypoint extraction fails.
    """
    try:
        # Run YOLO inference with verbose suppression for clean logs.
        # conf=0.5 filters out low-confidence person detections.
        results = yolo_model(frame, verbose=False, conf=YOLO_CONF_THRESHOLD)

        if not results or len(results) == 0:
            _logger.debug("YOLO returned no results for this frame.")
            return None

        result = results[0]

        if result.keypoints is None:
            _logger.debug("YOLO result contains no keypoint data.")
            return None

        # Access normalized xy coordinates (xyn) and confidence scores separately.
        # xyn[0] returns the first detected person's keypoints.
        if len(result.keypoints.xyn) == 0:
            _logger.debug("YOLO keypoints.xyn is empty.")
            return None

        keypoints_xy: np.ndarray = result.keypoints.xyn[0].cpu().numpy()
        keypoints_conf: np.ndarray = result.keypoints.conf[0].cpu().numpy()

        # Reject incomplete detections — must have all 17 COCO keypoints.
        if keypoints_xy.shape[0] < NUM_COCO_KEYPOINTS:
            _logger.debug(
                f"Incomplete keypoint detection: {keypoints_xy.shape[0]} "
                f"detected, expected {NUM_COCO_KEYPOINTS}."
            )
            return None

        # Concatenate xy + confidence into (17, 3) array.
        # np.float32 used throughout for PyTorch compatibility and memory efficiency.
        normalized: np.ndarray = np.concatenate(
            [keypoints_xy, keypoints_conf.reshape(-1, 1)], axis=1
        ).astype(np.float32)

        return normalized

    except Exception as e:
        # YOLO can raise on malformed frames (e.g., all-black images from
        # corrupt video streams). Gracefully return None rather than crashing.
        _logger.error(f"Keypoint extraction failed: {e}")
        return None


# ============================================================
# PIFR FEATURE COMPUTATION
# ============================================================

def compute_pifr(
    keypoints: Keypoints | None,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Compute the full 60D PIFR feature vector from normalized COCO keypoints.

    Feature composition:
      - Dimensions 0-50  (51D): Flattened 17 × 3 raw COCO keypoints [x, y, conf].
      - Dimensions 51-58  (9D):  Geometric features (F1-F9) encoding pose geometry.
      - Total: 60 dimensions.

    The 9D geometric sub-vector captures body configuration invariant to
    scale and absolute position, making it robust for fall detection:
      F1:  Center of mass X (confidence-weighted horizontal position).
      F2:  Center of mass Y (confidence-weighted vertical position).
      F3:  Shoulder-Nose angle (head tilt).
      F4:  Torso angle (nose-to-hip axis, radians from vertical).
      F5:  Hip angle (left-right hip axis).
      F6:  Shoulder angle (left-right shoulder axis).
      F7:  Left leg angle (hip-knee-ankle joint).
      F8:  Right leg angle (hip-knee-ankle joint).
      F9:  Nose-ankle angle (full-body orientation).

    Args:
        keypoints: Normalized keypoints of shape (17, 3) with [x, y, conf],
            all values in [0, 1]. Obtained from extract_keypoints().
        width:  Original frame width in pixels (unused here — keypoints are
                already normalized — kept for API compatibility).
        height: Original frame height in pixels (same as above).

    Returns:
        60D feature vector as np.ndarray of dtype float32.
        Returns zero-vector on invalid input to avoid pipeline crashes.
    """
    if keypoints is None or len(keypoints) < NUM_COCO_KEYPOINTS:
        _logger.debug("Invalid keypoints input — returning zero vector.")
        return np.zeros(TOTAL_PIFR_DIM, dtype=np.float32)

    try:
        features: list[float] = []

        # ------------------------------------------------------------
        # Stage 1: Flatten COCO keypoints (51D)
        # Each of the 17 keypoints contributes x, y, confidence.
        # This preserves raw spatial information for the Transformer.
        # ------------------------------------------------------------
        for i in range(NUM_COCO_KEYPOINTS):
            features.extend([
                float(keypoints[i, 0]),
                float(keypoints[i, 1]),
                float(keypoints[i, 2]),
            ])

        # ------------------------------------------------------------
        # Stage 2: Geometric features (9D)
        # All angles computed in radians using arccos on normalized vectors.
        # EPSILON added before division to prevent NaN from /0.
        # ------------------------------------------------------------

        # F1: Confidence-weighted center of mass — X coordinate.
        # Weighted by keypoint confidence to suppress noisy detections.
        conf_sum: float = float(np.sum(keypoints[:, 2])) + EPSILON
        center_x: float = float(np.sum(keypoints[:, 0] * keypoints[:, 2])) / conf_sum
        features.append(center_x)

        # F2: Confidence-weighted center of mass — Y coordinate.
        center_y: float = float(np.sum(keypoints[:, 1] * keypoints[:, 2])) / conf_sum
        features.append(center_y)

        # F3: Shoulder-Nose angle — measures head tilt relative to shoulders.
        # Two vectors from nose to each shoulder, angle between them.
        v1: np.ndarray = keypoints[LEFT_SHOULDER, :2] - keypoints[NOSE, :2]
        v2: np.ndarray = keypoints[RIGHT_SHOULDER, :2] - keypoints[NOSE, :2]
        n1: float = float(np.linalg.norm(v1))
        n2: float = float(np.linalg.norm(v2))
        angle: float
        if n1 > 0 and n2 > 0:
            # np.dot(v1, v2) / (n1 * n2) computes cos(angle).
            # np.clip ensures the argument is within [-1, 1] to prevent
            # domain errors in arccos due to floating-point rounding.
            angle = float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)))
        else:
            angle = 0.0
        features.append(angle)

        # F4: Torso angle — angle between nose-to-hip axis and vertical.
        # Computed as arccos(v[1] / ||v||) where v = mid_hip - nose.
        # This directly captures whether the person is upright (small angle)
        # or lying down (large angle, approaching π).
        mid_hip: np.ndarray = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2.0
        v: np.ndarray = mid_hip - keypoints[NOSE, :2]
        n: float = float(np.linalg.norm(v))
        torso_angle: float = float(np.arccos(np.clip(v[1] / n, -1.0, 1.0))) if n > 0 else 0.0
        features.append(torso_angle)

        # F5: Hip angle — angle of the hip axis relative to horizontal.
        v = keypoints[RIGHT_HIP, :2] - keypoints[LEFT_HIP, :2]
        n = float(np.linalg.norm(v))
        hip_angle: float = float(np.arccos(np.clip(v[0] / n, -1.0, 1.0))) if n > 0 else 0.0
        features.append(hip_angle)

        # F6: Shoulder angle — angle of the shoulder axis relative to horizontal.
        v = keypoints[RIGHT_SHOULDER, :2] - keypoints[LEFT_SHOULDER, :2]
        n = float(np.linalg.norm(v))
        shoulder_angle: float = float(np.arccos(np.clip(v[0] / n, -1.0, 1.0))) if n > 0 else 0.0
        features.append(shoulder_angle)

        # F7: Left leg angle — hip-knee-ankle joint angle.
        # A fall typically produces a sharp knee bend, detected here.
        v1 = keypoints[LEFT_KNEE, :2] - keypoints[LEFT_HIP, :2]
        v2 = keypoints[LEFT_ANKLE, :2] - keypoints[LEFT_KNEE, :2]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        left_leg_angle: float
        if n1 > 0 and n2 > 0:
            left_leg_angle = float(
                np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            )
        else:
            left_leg_angle = 0.0
        features.append(left_leg_angle)

        # F8: Right leg angle — symmetric to left leg angle.
        v1 = keypoints[RIGHT_KNEE, :2] - keypoints[RIGHT_HIP, :2]
        v2 = keypoints[RIGHT_ANKLE, :2] - keypoints[RIGHT_KNEE, :2]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        right_leg_angle: float
        if n1 > 0 and n2 > 0:
            right_leg_angle = float(
                np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            )
        else:
            right_leg_angle = 0.0
        features.append(right_leg_angle)

        # F9: Nose-ankle angle — full-body vertical alignment angle.
        # During a fall, the nose-ankle axis rotates from vertical (~0 rad)
        # toward horizontal (~π/2 rad), providing a strong fall indicator.
        mid_ankle: np.ndarray = (
            keypoints[LEFT_ANKLE, :2] + keypoints[RIGHT_ANKLE, :2]
        ) / 2.0
        v = mid_ankle - keypoints[NOSE, :2]
        n = float(np.linalg.norm(v))
        nose_ankle_angle: float = (
            float(np.arccos(np.clip(v[1] / n, -1.0, 1.0))) if n > 0 else 0.0
        )
        features.append(nose_ankle_angle)

        return np.array(features, dtype=np.float32)

    except Exception as e:
        # Numerical errors (e.g., NaN from arccos outside [-1,1]) are caught here.
        # Return zero vector so the sliding window is not corrupted by NaN.
        _logger.error(f"PIFR computation failed: {e}")
        return np.zeros(TOTAL_PIFR_DIM, dtype=np.float32)


def compute_9_pifr_features(keypoints: Keypoints | None) -> np.ndarray:
    """
    Compute only the 9D geometric PIFR sub-vector (F1-F9) from normalized keypoints.

    This standalone function is useful when the COCO keypoints (51D) are already
    available elsewhere and only the geometric features are needed.

    Args:
        keypoints: Normalized keypoints of shape (17, 3) with [x, y, conf],
            values in [0, 1] range.

    Returns:
        9D feature vector as np.ndarray of dtype float32.
    """
    if keypoints is None or len(keypoints) < NUM_COCO_KEYPOINTS:
        _logger.debug("Invalid keypoints for 9D PIFR — returning zero vector.")
        return np.zeros(GEOMETRIC_DIM, dtype=np.float32)

    try:
        features: np.ndarray = np.zeros(GEOMETRIC_DIM, dtype=np.float32)

        # F1: Center of mass X (confidence-weighted).
        # EPSILON prevents division by zero if all keypoint confidences are 0.
        conf_sum: float = float(np.sum(keypoints[:, 2])) + EPSILON
        features[0] = float(np.sum(keypoints[:, 0] * keypoints[:, 2])) / conf_sum

        # F2: Center of mass Y (confidence-weighted).
        features[1] = float(np.sum(keypoints[:, 1] * keypoints[:, 2])) / conf_sum

        # ----------------------------------------------------------------
        # Helper: _angle(v1, v2)
        # Computes the angle in radians between two 2D vectors using arccos
        # of their normalized dot product. Returns 0.0 if either vector
        # has zero magnitude (degenerate case).
        # ----------------------------------------------------------------
        def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
            n1: float = float(np.linalg.norm(v1))
            n2: float = float(np.linalg.norm(v2))
            if n1 > 0 and n2 > 0:
                return float(
                    np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                )
            return 0.0

        # F3: Shoulder-Nose angle.
        v1 = keypoints[LEFT_SHOULDER, :2] - keypoints[NOSE, :2]
        v2 = keypoints[RIGHT_SHOULDER, :2] - keypoints[NOSE, :2]
        features[2] = _angle(v1, v2)

        # F4: Torso angle (nose to mid-hip vs. vertical).
        mid_hip = (keypoints[LEFT_HIP, :2] + keypoints[RIGHT_HIP, :2]) / 2.0
        v = mid_hip - keypoints[NOSE, :2]
        n = float(np.linalg.norm(v))
        features[3] = float(np.arccos(np.clip(v[1] / n, -1.0, 1.0))) if n > 0 else 0.0

        # F5: Hip angle (left-right hip axis vs. horizontal).
        v = keypoints[RIGHT_HIP, :2] - keypoints[LEFT_HIP, :2]
        n = float(np.linalg.norm(v))
        features[4] = float(np.arccos(np.clip(v[0] / n, -1.0, 1.0))) if n > 0 else 0.0

        # F6: Shoulder angle (left-right shoulder axis vs. horizontal).
        v = keypoints[RIGHT_SHOULDER, :2] - keypoints[LEFT_SHOULDER, :2]
        n = float(np.linalg.norm(v))
        features[5] = float(np.arccos(np.clip(v[0] / n, -1.0, 1.0))) if n > 0 else 0.0

        # F7: Left leg angle (hip-knee-ankle).
        v1 = keypoints[LEFT_KNEE, :2] - keypoints[LEFT_HIP, :2]
        v2 = keypoints[LEFT_ANKLE, :2] - keypoints[LEFT_KNEE, :2]
        features[6] = _angle(v1, v2)

        # F8: Right leg angle (hip-knee-ankle).
        v1 = keypoints[RIGHT_KNEE, :2] - keypoints[RIGHT_HIP, :2]
        v2 = keypoints[RIGHT_ANKLE, :2] - keypoints[RIGHT_KNEE, :2]
        features[7] = _angle(v1, v2)

        # F9: Nose-ankle angle (full-body vertical alignment).
        mid_ankle = (keypoints[LEFT_ANKLE, :2] + keypoints[RIGHT_ANKLE, :2]) / 2.0
        v = mid_ankle - keypoints[NOSE, :2]
        n = float(np.linalg.norm(v))
        features[8] = float(np.arccos(np.clip(v[1] / n, -1.0, 1.0))) if n > 0 else 0.0

        return features

    except Exception as e:
        _logger.error(f"9D PIFR feature computation failed: {e}")
        return np.zeros(GEOMETRIC_DIM, dtype=np.float32)


# ============================================================
# VISUALIZATION
# ============================================================

def draw_skeleton(
    frame: Frame,
    keypoints: Keypoints | None,
    color: RGBColor = (0, 255, 0),
) -> Frame:
    """
    Render the COCO skeleton on a BGR frame for real-time visualization.

    Draws 12 line segments connecting anatomically adjacent keypoints
    (COCO topology) and individual keypoint circles with confidence
    threshold filtering.

    Args:
        frame: BGR image from OpenCV (H × W × 3).
        keypoints: Array of shape (17, 3) with [x, y, conf] in [0, 1].
        color: BGR color tuple for skeleton rendering (default: green).

    Returns:
        Deep-copy of the input frame with skeleton overlaid.
        Returns original frame unchanged if keypoints are invalid.
    """
    if keypoints is None or len(keypoints) < NUM_COCO_KEYPOINTS:
        _logger.debug("Cannot draw skeleton — invalid keypoints.")
        return frame

    try:
        h: int
        w: int
        h, w = frame.shape[:2]
        annotated: Frame = frame.copy()

        # COCO body topology: list of (start_index, end_index) pairs.
        # Connections follow anatomical structure: upper limbs, torso, lower limbs.
        connections: list[tuple[int, int]] = [
            (5, 6),    # shoulders
            (5, 7),    # left upper arm (shoulder → elbow)
            (7, 9),    # left forearm (elbow → wrist)
            (6, 8),    # right upper arm (shoulder → elbow)
            (8, 10),   # right forearm (elbow → wrist)
            (5, 11),   # left torso (shoulder → hip)
            (6, 12),   # right torso (shoulder → hip)
            (11, 12),  # hip connector
            (11, 13),  # left thigh (hip → knee)
            (13, 15),  # left shin (knee → ankle)
            (12, 14),  # right thigh (hip → knee)
            (14, 16),  # right shin (knee → ankle)
        ]

        # Draw each bone segment as a line if both endpoints are confident.
        for joint1, joint2 in connections:
            if (
                keypoints[joint1, 2] > KEYPOINT_VISUALIZATION_CONF
                and keypoints[joint2, 2] > KEYPOINT_VISUALIZATION_CONF
            ):
                pt1: tuple[int, int] = (
                    int(keypoints[joint1, 0] * w),
                    int(keypoints[joint1, 1] * h),
                )
                pt2: tuple[int, int] = (
                    int(keypoints[joint2, 0] * w),
                    int(keypoints[joint2, 1] * h),
                )
                cv2.line(annotated, pt1, pt2, color, 2)

        # Draw keypoint circles — outer white ring + inner colored dot.
        # The ring provides visual contrast against varied backgrounds.
        for kp in keypoints:
            if kp[2] > KEYPOINT_VISUALIZATION_CONF:
                x: int = int(kp[0] * w)
                y: int = int(kp[1] * h)
                cv2.circle(annotated, (x, y), 4, color, -1)   # filled center
                cv2.circle(annotated, (x, y), 6, (255, 255, 255), 1)  # white ring

        return annotated

    except Exception as e:
        _logger.error(f"Skeleton drawing failed: {e}")
        return frame
