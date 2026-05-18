"""
Visualization Utilities for Fall Detection
=========================================
Contains COCO skeleton connections and drawing helpers.
"""

import numpy as np
import cv2


# COCO keypoint connections for skeleton visualization
# Format: (start_idx, end_idx)
COCO_EDGES = [
    # Face
    (0, 1),   # nose-left_eye
    (0, 2),   # nose-right_eye
    (1, 3),   # left_eye-left_ear
    (2, 4),   # right_eye-right_ear
    
    # Torso
    (5, 6),   # left_shoulder-right_shoulder
    (5, 11),  # left_shoulder-left_hip
    (6, 12),  # right_shoulder-right_hip
    (11, 12), # left_hip-right_hip
    
    # Left arm
    (5, 7),   # left_shoulder-left_elbow
    (7, 9),   # left_elbow-left_wrist
    
    # Right arm
    (6, 8),   # right_shoulder-right_elbow
    (8, 10),  # right_elbow-right_wrist
    
    # Left leg
    (11, 13), # left_hip-left_knee
    (13, 15), # left_knee-left_ankle
    
    # Right leg
    (12, 14), # right_hip-right_knee
    (14, 16), # right_knee-right_ankle
]


# Keypoint names for reference
COCO_KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]


# Color scheme
KEYPOINT_COLOR = (0, 128, 255)      # Orange for keypoints
EDGE_COLOR = (0, 255, 180)          # Cyan for skeleton edges
CONFIDENCE_LOW_COLOR = (128, 128, 128)
CONFIDENCE_MEDIUM_COLOR = (0, 165, 255)
CONFIDENCE_HIGH_COLOR = (0, 255, 0)
FALL_COLOR = (0, 0, 255)            # Red for fall detection
SAFE_COLOR = (0, 255, 0)            # Green for safe


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.3,
    line_thickness: int = 2,
    point_radius: int = 4
) -> np.ndarray:
    """
    Draw skeleton on frame.
    
    Args:
        frame: BGR image
        keypoints: (17, 3) or (17, 2) - x, y, [conf]
        conf_threshold: Minimum confidence to draw
        line_thickness: Thickness of skeleton lines
        point_radius: Radius of keypoint circles
    
    Returns:
        Frame with skeleton drawn
    """
    if keypoints is None or len(keypoints) < 17:
        return frame
    
    h, w = frame.shape[:2]
    out = frame.copy()
    
    # Extract coordinates and confidence
    if keypoints.shape[1] == 3:
        kpts_xy = keypoints[:, :2]
        kpts_conf = keypoints[:, 2]
    else:
        kpts_xy = keypoints
        kpts_conf = np.ones(17)
    
    # Draw edges
    for start_idx, end_idx in COCO_EDGES:
        if (kpts_conf[start_idx] < conf_threshold or 
            kpts_conf[end_idx] < conf_threshold):
            continue
        
        pt1 = (int(np.clip(kpts_xy[start_idx, 0], 0, w-1)),
               int(np.clip(kpts_xy[start_idx, 1], 0, h-1)))
        pt2 = (int(np.clip(kpts_xy[end_idx, 0], 0, w-1)),
               int(np.clip(kpts_xy[end_idx, 1], 0, h-1)))
        
        cv2.line(out, pt1, pt2, EDGE_COLOR, line_thickness, cv2.LINE_AA)
    
    # Draw keypoints
    for i in range(17):
        if kpts_conf[i] < conf_threshold:
            continue
        
        pt = (int(np.clip(kpts_xy[i, 0], 0, w-1)),
              int(np.clip(kpts_xy[i, 1], 0, h-1)))
        
        # Color based on confidence
        if kpts_conf[i] > 0.7:
            color = CONFIDENCE_HIGH_COLOR
        elif kpts_conf[i] > 0.4:
            color = CONFIDENCE_MEDIUM_COLOR
        else:
            color = CONFIDENCE_LOW_COLOR
        
        cv2.circle(out, pt, point_radius, color, -1, cv2.LINE_AA)
    
    return out


def draw_bounding_box(
    frame: np.ndarray,
    box: np.ndarray,
    color: tuple = SAFE_COLOR,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on frame.
    
    Args:
        frame: BGR image
        box: (4,) - x1, y1, x2, y2
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Frame with bounding box drawn
    """
    if box is None or len(box.flatten()) < 4:
        return frame
    
    x1, y1, x2, y2 = [int(x) for x in box.flatten()[:4]]
    h, w = frame.shape[:2]
    
    # Clip to frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    
    return frame


def draw_fall_alert(
    frame: np.ndarray,
    probability: float,
    grade: str = 'medium'
) -> np.ndarray:
    """
    Draw fall alert overlay on frame.
    
    Args:
        frame: BGR image
        probability: Fall probability
        grade: 'low', 'medium', 'high'
    
    Returns:
        Frame with alert overlay
    """
    out = frame.copy()
    h, w = out.shape[:2]
    
    # Color based on grade
    if grade == 'high':
        color = FALL_COLOR
    elif grade == 'medium':
        color = CONFIDENCE_MEDIUM_COLOR
    else:
        color = SAFE_COLOR
    
    # Background rectangle for text
    cv2.rectangle(out, (5, 5), (int(w*0.4), 45), (0, 0, 0), -1)
    cv2.rectangle(out, (5, 5), (int(w*0.4), 45), color, 2)
    
    # Alert text
    text = f"FALL: {probability:.2f} [{grade.upper()}]"
    cv2.putText(out, text, (10, 32),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    return out
