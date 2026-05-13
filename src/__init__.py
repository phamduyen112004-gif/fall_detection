"""
Fall Detection Package.

Một hệ thống lai ghép YOLOv11-Pose + PIFR Geometric Features + Transformer
để phát hiện ngã người trong thời gian thực.

Modules:
    config          - Cấu hình pipeline
    pifr_features   - Trích xuất đặc trưng hình học 60-D
    hybrid_fall_transformer - Kiến trúc Transformer model
    pipeline        - Bộ điều phối pipeline end-to-end
    stage1_preprocess - Tiền xử lý frame
    stage2_pose     - Trích xuất pose từ YOLOv11
    stage3_kinematics - Tính toán góc & phân loại tư thế
    stage4_alert    - Cảnh báo Telegram
    viz             - Tiện ích visualization
    groups          - Nhóm subject cho train/val split
    types           - Dataclass dùng chung

Ví dụ sử dụng:
    >>> from src import GeometricFeatureExtractor, HybridFallTransformer
    >>> extractor = GeometricFeatureExtractor()
    >>> model = HybridFallTransformer()

Cách chạy:
    # Chạy bộ test
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
# Shared Types
# =============================================================================
from .types import FrameDiag

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
    # Shared types
    "FrameDiag",
]
