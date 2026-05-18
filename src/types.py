"""
Type Definitions for Fall Detection
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class FrameDiag:
    """Diagnostic information for a single frame."""
    frame_idx: int
    keypoints: Optional[np.ndarray] = None
    box: Optional[np.ndarray] = None
    confidence: float = 0.0
    valid: bool = False
    vec60: Optional[np.ndarray] = None
    
    # Posture features
    torso_angle_deg: float = 0.0
    bbox_aspect_ratio: float = 0.0
    bottom_ratio: float = 0.0
    
    # Tracking
    center_x: float = 0.0
    center_y: float = 0.0


@dataclass  
class InferenceResult:
    """Result of fall detection inference."""
    frame_idx: int
    prob_fall: float
    is_fall: bool
    confidence_grade: str  # 'low', 'medium', 'high'
    posture_features: Optional[dict] = None


@dataclass
class AlertInfo:
    """Alert information for fall detection."""
    timestamp: float
    probability: float
    frame_snapshot: Optional[np.ndarray] = None
    alert_sent: bool = False
    telegram_message_id: Optional[str] = None
