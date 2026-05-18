"""
Unit tests for data preprocessing pipeline.

Tests video feature extraction and temporal standardization.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import (
    standardize_to_60x60,
    extract_pifr_60d,
    _is_fall,
    _safe_name,
    TARGET_FRAMES,
    MAX_FRAMES,
)


class TestStandardizeTo60x60:
    """Test temporal standardization to (60, 60) shape."""

    def test_exact_60_frames(self):
        """Exactly 60 frames should pass through unchanged."""
        features = [np.random.randn(60).astype(np.float32) for _ in range(60)]
        result = standardize_to_60x60(features)
        assert result.shape == (TARGET_FRAMES, 60)

    def test_empty_input_returns_zeros(self):
        """Empty input should return zeros."""
        result = standardize_to_60x60([])
        assert result.shape == (TARGET_FRAMES, 60)
        assert np.all(result == 0)

    def test_none_input_returns_zeros(self):
        """None input should return zeros."""
        result = standardize_to_60x60(None)
        assert result.shape == (TARGET_FRAMES, 60)
        assert np.all(result == 0)

    def test_short_video_pads_with_last_frame(self):
        """Shorter videos should be padded with last frame."""
        features = [np.ones(60).astype(np.float32) * i for i in range(30)]
        result = standardize_to_60x60(features)
        
        # First 30 frames should be original
        np.testing.assert_array_almost_equal(result[:30], np.array(features))
        # Last 30 frames should be padding (last frame repeated)
        for i in range(30, 60):
            np.testing.assert_array_almost_equal(result[i], result[29])

    def test_long_video_truncates(self):
        """Videos longer than 120 frames should be truncated."""
        features = [np.random.randn(60).astype(np.float32) for _ in range(150)]
        result = standardize_to_60x60(features)
        # After truncation to 120 and subsampling by 2: 60 frames
        assert len(result) <= TARGET_FRAMES

    def test_subsampling_every_2nd_frame(self):
        """Should subsample every 2nd frame."""
        # 100 frames
        features = [np.array([i], dtype=np.float32) for i in range(100)]
        result = standardize_to_60x60(features)
        
        # After subsampling: 50 frames, need padding to 60
        # First value should be frame 0
        assert result[0, 0] == 0.0

    def test_output_dtype_is_float32(self):
        """Output should be float32."""
        features = [np.random.randn(60).astype(np.float64) for _ in range(10)]
        result = standardize_to_60x60(features)
        assert result.dtype == np.float32

    def test_output_shape_consistency(self):
        """Output should always be (60, 60)."""
        test_cases = [
            [np.random.randn(60).astype(np.float32) for _ in range(10)],   # Short
            [np.random.randn(60).astype(np.float32) for _ in range(60)],   # Exact
            [np.random.randn(60).astype(np.float32) for _ in range(100)], # Medium
        ]
        for features in test_cases:
            result = standardize_to_60x60(features)
            assert result.shape == (TARGET_FRAMES, 60), f"Failed for {len(features)} input frames"


class TestExtractPIFR60D:
    """Test 60D PIFR feature extraction."""

    def test_output_shape(self, sample_keypoints_standing):
        """Output should be 60D vector."""
        keypoints = np.random.randn(17, 3).astype(np.float32)
        keypoints[:, 2] = 0.9  # Confidence
        result = extract_pifr_60d(keypoints)
        assert result.shape == (60,)

    def test_combines_flattened_and_geometric(self, sample_keypoints_standing):
        """Should combine flattened keypoints with geometric features."""
        keypoints = sample_keypoints_standing.copy()
        result = extract_pifr_60d(keypoints)
        
        # First 51 should be flattened keypoints
        expected_flat = keypoints.flatten()
        np.testing.assert_array_almost_equal(result[:51], expected_flat)
        
        # Last 9 should be geometric features
        assert len(result) == 51 + 9

    def test_invalid_keypoints_shape(self):
        """Invalid keypoints should handle gracefully."""
        keypoints = np.zeros((10, 3), dtype=np.float32)  # Wrong shape
        result = extract_pifr_60d(keypoints)
        # Should return zeros for geometric part
        assert result.shape == (60,)


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_fall_detects_fall(self):
        """Actions with 'fall' should return 1."""
        assert _is_fall("Fall") == 1
        assert _is_fall("falling") == 1
        assert _is_fall("FALL") == 1
        assert _is_fall("walking_fall_down") == 1

    def test_is_fall_rejects_non_fall(self):
        """Actions without 'fall' should return 0."""
        assert _is_fall("walking") == 0
        assert _is_fall("sitting") == 0
        assert _is_fall("standing") == 0
        assert _is_fall("running") == 0

    def test_is_fall_case_insensitive(self):
        """Should be case insensitive."""
        assert _is_fall("Fall") == _is_fall("FALL")
        assert _is_fall("Fall") == _is_fall("fall")

    def test_safe_name_replaces_special_chars(self):
        """Should sanitize special characters."""
        assert "_" in _safe_name("Subject.1")
        assert "_" in _safe_name("Subject 1")
        assert " " not in _safe_name("Subject 1")
        assert "." not in _safe_name("Subject.1")

    def test_safe_name_handles_multiple_special_chars(self):
        """Should handle multiple special characters."""
        result = _safe_name("Subject.1/Action 2\\Test")
        assert "/" not in result
        assert "\\" not in result
        assert " " not in result
        assert "." not in result


class TestConstants:
    """Test configuration constants."""

    def test_target_frames_is_60(self):
        """TARGET_FRAMES should be 60."""
        assert TARGET_FRAMES == 60

    def test_max_frames_is_120(self):
        """MAX_FRAMES should be 120."""
        assert MAX_FRAMES == 120

    def test_max_frames_doubles_target(self):
        """MAX_FRAMES should be 2x TARGET_FRAMES (for subsampling)."""
        assert MAX_FRAMES == TARGET_FRAMES * 2
