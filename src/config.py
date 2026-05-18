"""
Central Configuration for Fall Detection Pipeline.
================================================
Environment-aware paths + PipelineConfig dataclass.

All path constants and hyperparameters are defined here.
Import from this module in all scripts.

Datasets: CaucaFall, MCFD only.

Constants Reference:
    - YOLO: YOLOv11n-Pose model for keypoint detection
    - PIFR: Person-in-Frame Representation for feature extraction
    - Hybrid Transformer: Temporal modeling for fall classification

Environment Variables (optional):
    TELEGRAM_BOT_TOKEN: Bot token for Telegram alerts
    TELEGRAM_CHAT_ID: Chat ID for Telegram alerts
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

# =============================================================================
# Environment Detection
# =============================================================================

IS_KAGGLE: bool = os.path.exists("/kaggle")
_ROOT: Path = Path(__file__).parent.parent.resolve()

# =============================================================================
# Dataset Paths
# =============================================================================

if IS_KAGGLE:
    CAUCAFALL_DIR: Path = Path("/kaggle/input/caucafall/Dataset CAUCAFall/CAUCAFall")
    MCFD_DIR: Path = Path("/kaggle/input/multiple-cameras-fall-dataset/dataset/dataset")
    MCFD_CSV: Path = Path("/kaggle/input/multiple-cameras-fall-dataset/data_tuple3.csv")
else:
    CAUCAFALL_DIR = _ROOT / "data" / "raw" / "caucafall"
    MCFD_DIR = _ROOT / "data" / "raw" / "mcfd"
    MCFD_CSV = MCFD_DIR / "data_tuple3.csv"

# =============================================================================
# Output & Save Paths
# =============================================================================

if IS_KAGGLE:
    OUTPUT_DIR: Path = Path("/kaggle/working/processed")
    MODEL_SAVE_DIR: Path = Path("/kaggle/working/models")
    LOG_DIR: Path = Path("/kaggle/working/logs")
    RESULTS_DIR: Path = Path("/kaggle/working/results")
else:
    OUTPUT_DIR = _ROOT / "data" / "processed"
    MODEL_SAVE_DIR = _ROOT / "models"
    LOG_DIR = _ROOT / "logs"
    RESULTS_DIR = _ROOT / "results"

DATA_DIR: Path = OUTPUT_DIR

# =============================================================================
# Temporal Standardization
# =============================================================================

MAX_FRAMES: int = 120
"""Maximum number of frames per video sequence."""

TARGET_FRAMES: int = 60
"""Target number of frames after temporal standardization."""

# =============================================================================
# PIFR Feature Dimension
# =============================================================================

KEYPOINT_DIM: int = 17
"""Number of keypoints detected by YOLO pose model."""

KEYPOINT_FEATURES: int = KEYPOINT_DIM * 3
"""Total keypoint features: (x, y, confidence) * num_keypoints."""

GEOMETRIC_FEATURES: int = 9
"""Number of geometric features extracted per frame."""

TOTAL_FEATURES: int = KEYPOINT_FEATURES + GEOMETRIC_FEATURES
"""Total input feature dimension for the model."""

# =============================================================================
# Model Configuration
# =============================================================================

YOLO_MODEL: str = "yolo11n-pose.pt"
"""Path or name of the YOLO pose model."""

CONF_THRESHOLD: float = 0.5
"""Minimum confidence threshold for YOLO detections."""

# =============================================================================
# Training Configuration
# =============================================================================

RANDOM_SEED: int = 42
"""Random seed for reproducibility across all operations."""

# --- Architecture Hyperparameters ---
D_MODEL: int = 256
"""Dimension of model embeddings in the hybrid transformer."""

NHEAD: int = 4
"""Number of attention heads in the transformer encoder."""

NUM_LAYERS: int = 3
"""Number of transformer encoder layers."""

DROPOUT: float = 0.1
"""Dropout probability for regularization."""

# --- Training Hyperparameters ---
DEFAULT_EPOCHS: int = 100
"""Default number of training epochs."""

DEFAULT_BATCH_SIZE: int = 64
"""Default batch size for training."""

DEFAULT_LEARNING_RATE: float = 5e-4
"""Default learning rate (SOTA-preserved value)."""

WEIGHT_DECAY: float = 1e-5
"""L2 regularization strength."""

EARLY_STOPPING_PATIENCE: int = 25
"""Epochs to wait for improvement before stopping."""

# --- Augmentation Hyperparameters ---
NOISE_STD: float = 0.01
"""Standard deviation for Gaussian noise augmentation."""

MASK_RATIO: float = 0.05
"""Ratio of input features to mask during augmentation."""

# --- Dataset Split Ratios ---
TEST_SIZE: float = 0.2
"""Proportion of data reserved for testing."""

VAL_SIZE: float = 0.1
"""Proportion of training data reserved for validation."""


@dataclass
class TrainingConfig:
    """
    Training hyperparameters for the fall detection model.

    This dataclass is mutable to allow runtime configuration adjustments.
    SOTA values are preserved as defaults (lr=0.0005, d_model=256).

    Attributes:
        d_model: Dimension of model embeddings in the hybrid transformer.
        nhead: Number of attention heads in the transformer encoder.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout probability for regularization.
        input_dim: Input feature dimension (derived from PIFR extraction).
        num_frames: Number of frames per input sequence.
        epochs: Maximum number of training epochs.
        batch_size: Number of samples per training batch.
        learning_rate: Initial learning rate for the optimizer.
        weight_decay: L2 regularization strength.
        early_stopping_patience: Epochs without improvement before early stop.
        noise_std: Standard deviation for Gaussian noise augmentation.
        mask_ratio: Ratio of features to mask during augmentation.
        test_size: Proportion of data for testing.
        val_size: Proportion of training data for validation.
        data_dir: Override path for data directory (None uses default).
        model_dir: Override path for model saves (None uses default).
        log_dir: Override path for logs (None uses default).

    Example:
        >>> config = TrainingConfig(epochs=50, batch_size=32)
        >>> config.learning_rate = 1e-3  # Adjust learning rate
        >>> config.data_dir = "/custom/data/path"  # Use custom data path
    """
    # Model architecture
    d_model: int = D_MODEL
    nhead: int = NHEAD
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT

    # Input dimensions (derived from feature extraction)
    input_dim: int = TOTAL_FEATURES
    num_frames: int = TARGET_FRAMES

    # Training settings
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE

    # Augmentation
    noise_std: float = NOISE_STD
    mask_ratio: float = MASK_RATIO

    # Split ratios
    test_size: float = TEST_SIZE
    val_size: float = VAL_SIZE

    # Paths (None = use config.py defaults)
    data_dir: str | None = None
    model_dir: str | None = None
    log_dir: str | None = None

    # --- Property accessors for default paths ---
    @property
    def resolved_data_dir(self) -> Path:
        """Get the effective data directory, falling back to default."""
        return Path(self.data_dir) if self.data_dir else DATA_DIR

    @property
    def resolved_model_dir(self) -> Path:
        """Get the effective model directory, falling back to default."""
        return Path(self.model_dir) if self.model_dir else MODEL_SAVE_DIR

    @property
    def resolved_log_dir(self) -> Path:
        """Get the effective log directory, falling back to default."""
        return Path(self.log_dir) if self.log_dir else LOG_DIR


# Global instance - single source of truth
TRAINING_CONFIG: TrainingConfig = TrainingConfig()


# =============================================================================
# Detection / Inference Constants
# =============================================================================

# --- Person Detection Filters ---
MAX_MISSING_FRAMES: int = 15
"""Maximum consecutive frames a person can be missing before track removal."""

INFER_STRIDE: int = 15
"""Frame stride for inference on video streams."""

MIN_VALID_FRAMES_FOR_INFER: int = 8
"""Minimum valid frames required before performing inference."""

MIN_PERSON_AREA_RATIO: float = 0.02
"""Minimum bounding box area relative to frame size."""

MIN_PERSON_HEIGHT_RATIO: float = 0.18
"""Minimum person height relative to frame height."""

MIN_KEYPOINTS_CONFIDENT: int = 7
"""Minimum number of high-confidence keypoints required."""

MAX_TRACK_CENTER_JUMP_RATIO: float = 0.20
"""Maximum allowed center jump ratio between frames (track stability)."""

# --- Posture Angle Thresholds (degrees from vertical) ---
TILT_LOW_THRESHOLD: float = 15.0
"""Lower tilt threshold for posture classification (degrees)."""

TILT_HIGH_THRESHOLD: float = 55.0
"""Upper tilt threshold for fall detection (degrees)."""

FALL_MIN_BOTTOM_RATIO: float = 0.65
"""Minimum bottom position ratio to classify as lying down."""

SOFA_SIT_BOTTOM_RATIO_MAX: float = 0.83
"""Maximum bottom ratio for sitting posture detection."""

# --- Fall Detection Sensitivity ---
DROP_LOOKBACK_FRAMES: int = 4
"""Number of frames to look back for drop detection."""

MIN_DROP_DELTA_CENTER_Y: float = 0.06
"""Minimum center Y delta to detect a drop."""

LOW_CONF_THRESHOLD: float = 0.10
"""Low confidence threshold for uncertain predictions."""

HIGH_CONF_THRESHOLD: float = 0.25
"""High confidence threshold for certain predictions."""

ALERT_MIN_PROB: float = 0.18
"""Minimum probability to trigger a fall alert."""

ALERT_COOLDOWN_SEC: float = 10.0
"""Cooldown period between alerts (seconds)."""

# --- Stream Handling ---
STREAM_RECONNECT_DELAY_SEC: float = 2.5
"""Initial delay between reconnection attempts (seconds)."""

STREAM_RECONNECT_BACKOFF_MAX: float = 8.0
"""Maximum delay between reconnection attempts (seconds)."""


# =============================================================================
# Pipeline Configuration
# =============================================================================

# --- YOLO Input Processing ---
DEFAULT_INPUT_SIZE: tuple[int, int] = (640, 640)
"""Default input size for YOLO model (width, height)."""

MIN_KEYPOINT_CONF: float = 0.2
"""Minimum mean keypoint confidence to accept a detection."""


@dataclass
class PipelineConfig:
    """
    Runtime configuration for inference pipeline components.

    Passed to PoseExtractor, KinematicsAnalyzer, TelegramAlerter, etc.
    This dataclass is mutable to allow runtime configuration.

    Attributes:
        input_size: Input image size for YOLO model (width, height).
        pose_model: Path or name of the YOLO pose model.
        min_mean_keypoint_conf: Minimum mean keypoint confidence threshold.
        laydown_torso_angle_deg: Torso angle threshold for laydown detection.
        laydown_nose_ankle_angle_deg: Nose-to-ankle angle for laydown detection.
        fall_min_frames: Minimum frames in fall posture to confirm a fall.
        fall_min_seconds: Minimum duration in fall posture (seconds).
        telegram_bot_token: Bot token for Telegram alerts (auto-loaded from env).
        telegram_chat_id: Chat ID for Telegram alerts (auto-loaded from env).

    Note:
        Telegram credentials are automatically loaded from environment variables
        TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID if not provided explicitly.

    Example:
        >>> config = PipelineConfig(
        ...     input_size=(416, 416),
        ...     fall_min_seconds=15.0,
        ... )
    """
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE
    pose_model: str = YOLO_MODEL
    min_mean_keypoint_conf: float = MIN_KEYPOINT_CONF

    # Posture thresholds (degrees from vertical)
    laydown_torso_angle_deg: float = TILT_HIGH_THRESHOLD
    laydown_nose_ankle_angle_deg: float = 50.0
    fall_min_frames: int = 60
    fall_min_seconds: float | None = 10.0

    # Telegram credentials (auto-loaded from env if None)
    telegram_bot_token: str | None = field(default=None, repr=False)
    telegram_chat_id: str | None = field(default=None, repr=False)

    # Internal flag to track if credentials were loaded from environment
    _credentials_loaded: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """
        Initialize Telegram credentials from environment if not provided.

        Loads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment
        variables when not explicitly set in the constructor.
        """
        if self.telegram_bot_token is None:
            self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if self.telegram_chat_id is None:
            self.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")


# Default pipeline configuration instance
DEFAULT_CONFIG: PipelineConfig = PipelineConfig()
