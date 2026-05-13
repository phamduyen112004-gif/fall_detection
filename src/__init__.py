"""
Fall Detection Package.

A Hybrid YOLOv11-Pose + PIFR Geometric Features + Transformer system
for real-time human fall detection.

Modules:
    config          - Pipeline configuration
    pifr_features  - 60-D geometric feature extraction
    hybrid_fall_transformer - Transformer model
    pipeline       - End-to-end pipeline orchestrator
    stage1_preprocess - Frame preprocessing
    stage2_pose    - YOLOv11 pose extraction
    stage3_kinematics - Angle computation & posture classification
    stage4_alert   - Telegram alerting
    viz            - Visualization utilities
    groups         - Subject grouping for train/val split

Example:
    >>> from src import GeometricFeatureExtractor, HybridFallTransformer
    >>> extractor = GeometricFeatureExtractor()
    >>> model = HybridFallTransformer()

Usage:
    # Run pytest suite
    $ PYTHONPATH=. pytest tests/ -v
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Fall Detection Team"

# =============================================================================
# Core Feature Extraction
# =============================================================================
from .pifr_features import (
    GeometricFeatureExtractor,
    extract_pifr_features,
    get_default_extractor,
    EPS,
    IMGSZ,
    MIN_MEAN_CONF,
    FEATURE_DIM,
    SEQ_LEN,
    frame_to_vector_60,
    resample_to_length,
)

# =============================================================================
# Model Architecture
# =============================================================================
from .hybrid_fall_transformer import (
    HybridFallTransformer,
    SinusoidalPositionalEncoding,
)

# =============================================================================
# Configuration
# =============================================================================
from .config import (
    PipelineConfig,
    DEFAULT_CONFIG,
)

# =============================================================================
# Pipeline Components
# =============================================================================
from .stage2_pose import (
    PoseExtractor,
    PoseFrame,
)
from .stage3_kinematics import (
    KinematicsAnalyzer,
    Posture,
    FallTemporalFilter,
    compute_pose_angles,
    classify_posture,
)
from .stage4_alert import (
    TelegramAlerter,
    encode_jpeg_bgr,
)

# =============================================================================
# Visualization
# =============================================================================
from .viz import draw_pose_overlay, COCO_EDGES

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "__version__",
    # Feature extraction
    "GeometricFeatureExtractor",
    "extract_pifr_features",
    "get_default_extractor",
    "EPS",
    "IMGSZ",
    "MIN_MEAN_CONF",
    "FEATURE_DIM",
    "SEQ_LEN",
    "frame_to_vector_60",
    "resample_to_length",
    # Model
    "HybridFallTransformer",
    "SinusoidalPositionalEncoding",
    # Configuration
    "PipelineConfig",
    "DEFAULT_CONFIG",
    # Pipeline components
    "PoseExtractor",
    "PoseFrame",
    "KinematicsAnalyzer",
    "Posture",
    "FallTemporalFilter",
    "compute_pose_angles",
    "classify_posture",
    "TelegramAlerter",
    "encode_jpeg_bgr",
    # Visualization
    "draw_pose_overlay",
    "COCO_EDGES",
]
