"""
Unit tests for configuration module.

Tests configuration constants and dataclasses.
"""

import os
import pytest
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PipelineConfig,
    TrainingConfig,
    DEFAULT_CONFIG,
    TRAINING_CONFIG,
    KEYPOINT_DIM,
    KEYPOINT_FEATURES,
    GEOMETRIC_FEATURES,
    TOTAL_FEATURES,
    TARGET_FRAMES,
    MAX_FRAMES,
    RANDOM_SEED,
)


class TestFeatureDimensions:
    """Test feature dimension constants."""

    def test_keypoint_dim_is_17(self):
        """COCO has 17 keypoints."""
        assert KEYPOINT_DIM == 17

    def test_keypoint_features_is_51(self):
        """17 keypoints × 3 values (x, y, conf) = 51."""
        assert KEYPOINT_FEATURES == 51

    def test_geometric_features_is_9(self):
        """9 geometric features (angles + center of mass)."""
        assert GEOMETRIC_FEATURES == 9

    def test_total_features_is_60(self):
        """51 keypoint values + 9 geometric = 60."""
        assert TOTAL_FEATURES == 60
        assert TOTAL_FEATURES == KEYPOINT_FEATURES + GEOMETRIC_FEATURES


class TestPipelineConfig:
    """Test PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()
        assert config.input_size == (640, 640)
        assert config.pose_model == "yolo11n-pose.pt"
        assert config.min_mean_keypoint_conf == 0.2
        assert config.fall_min_frames == 60
        assert config.laydown_torso_angle_deg == 55.0
        assert config.laydown_nose_ankle_angle_deg == 50.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            pose_model="yolo11m-pose.pt",
            fall_min_frames=30,
            telegram_bot_token="test_token",
        )
        assert config.pose_model == "yolo11m-pose.pt"
        assert config.fall_min_frames == 30
        assert config.telegram_bot_token == "test_token"

    def test_frozen_dataclass(self):
        """Config should be immutable (frozen)."""
        config = PipelineConfig()
        with pytest.raises(AttributeError):
            config.fall_min_frames = 100

    def test_env_var_fallback_token(self):
        """Should fallback to environment variable for token."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env_token"}):
            config = PipelineConfig()
            assert config.telegram_bot_token == "env_token"

    def test_env_var_fallback_chat_id(self):
        """Should fallback to environment variable for chat_id."""
        with patch.dict(os.environ, {"TELEGRAM_CHAT_ID": "env_chat_id"}):
            config = PipelineConfig()
            assert config.telegram_chat_id == "env_chat_id"

    def test_explicit_value_overrides_env(self):
        """Explicit value should override environment variable."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env_token"}):
            config = PipelineConfig(telegram_bot_token="explicit_token")
            assert config.telegram_bot_token == "explicit_token"


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_hyperparameters(self):
        """Test default training hyperparameters."""
        config = TrainingConfig()
        assert config.INPUT_DIM == 60
        assert config.NUM_FRAMES == 60
        assert config.D_MODEL == 256
        assert config.NHEAD == 4
        assert config.NUM_LAYERS == 3
        assert config.DROPOUT == 0.1

    def test_default_training_settings(self):
        """Test default training settings."""
        config = TrainingConfig()
        assert config.EPOCHS == 100
        assert config.BATCH_SIZE == 64
        assert config.LEARNING_RATE == 5e-4
        assert config.WEIGHT_DECAY == 1e-5
        assert config.EARLY_STOPPING_PATIENCE == 25

    def test_default_augmentation(self):
        """Test default augmentation settings."""
        config = TrainingConfig()
        assert config.NOISE_STD == 0.01
        assert config.MASK_RATIO == 0.05

    def test_default_split_ratios(self):
        """Test default train/val/test split ratios."""
        config = TrainingConfig()
        assert config.TEST_SIZE == 0.2
        assert config.VAL_SIZE == 0.1
        # Train should be 70% (100% - 20% test - 10% val)
        assert config.TEST_SIZE + config.VAL_SIZE == 0.3

    def test_custom_hyperparameters(self):
        """Test custom hyperparameters."""
        config = TrainingConfig(
            D_MODEL=128,
            NHEAD=2,
            NUM_LAYERS=2,
            BATCH_SIZE=32,
        )
        assert config.D_MODEL == 128
        assert config.NHEAD == 2
        assert config.NUM_LAYERS == 2
        assert config.BATCH_SIZE == 32


class TestGlobalInstances:
    """Test global config instances."""

    def test_default_config_is_pipeline_config(self):
        """DEFAULT_CONFIG should be a PipelineConfig instance."""
        assert isinstance(DEFAULT_CONFIG, PipelineConfig)

    def test_training_config_is_training_config(self):
        """TRAINING_CONFIG should be a TrainingConfig instance."""
        assert isinstance(TRAINING_CONFIG, TrainingConfig)

    def test_random_seed_is_set(self):
        """RANDOM_SEED should be set to a fixed value for reproducibility."""
        assert RANDOM_SEED == 42
        assert isinstance(RANDOM_SEED, int)


class TestTemporalConstants:
    """Test temporal standardization constants."""

    def test_target_frames_matches_training_config(self):
        """TARGET_FRAMES should match TrainingConfig.NUM_FRAMES."""
        config = TrainingConfig()
        assert TARGET_FRAMES == config.NUM_FRAMES

    def test_max_frames_is_double_target(self):
        """MAX_FRAMES should be 2x TARGET_FRAMES for subsampling."""
        assert MAX_FRAMES == TARGET_FRAMES * 2


class TestHyperparamsDerived:
    """Test that TRAINING_HYPERPARAMS is derived from TrainingConfig."""

    def test_hyperparams_matches_config(self):
        """TRAINING_HYPERPARAMS values should match TrainingConfig."""
        from src.config import TRAINING_HYPERPARAMS, TRAINING_CONFIG
        
        assert TRAINING_HYPERPARAMS["epochs"] == TRAINING_CONFIG.EPOCHS
        assert TRAINING_HYPERPARAMS["batch_size"] == TRAINING_CONFIG.BATCH_SIZE
        assert TRAINING_HYPERPARAMS["learning_rate"] == TRAINING_CONFIG.LEARNING_RATE
        assert TRAINING_HYPERPARAMS["d_model"] == TRAINING_CONFIG.D_MODEL
        assert TRAINING_HYPERPARAMS["nhead"] == TRAINING_CONFIG.NHEAD
        assert TRAINING_HYPERPARAMS["num_layers"] == TRAINING_CONFIG.NUM_LAYERS


class TestPathConstants:
    """Test path constants (platform-aware)."""
    
    def test_output_dir_configured(self):
        """OUTPUT_DIR should be defined."""
        from src.config import OUTPUT_DIR
        assert OUTPUT_DIR is not None

    def test_model_save_dir_configured(self):
        """MODEL_SAVE_DIR should be defined."""
        from src.config import MODEL_SAVE_DIR
        assert MODEL_SAVE_DIR is not None

    def test_results_dir_configured(self):
        """RESULTS_DIR should be defined."""
        from src.config import RESULTS_DIR
        assert RESULTS_DIR is not None
