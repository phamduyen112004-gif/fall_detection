"""
Unit Tests for PIFR Features Module
====================================
Tests for keypoint extraction, PIFR computation, and geometric features.
"""

import numpy as np
import pytest

from src.pifr_features import (
    COCO_IDX,
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE,
    LEFT_ANKLE,
    RIGHT_ANKLE,
    compute_pifr,
    compute_9_pifr_features,
)


class TestCOCOIndices:
    """Test COCO keypoint indices are correct."""

    def test_coco_idx_count(self):
        assert len(COCO_IDX) == 17

    def test_key_indices(self):
        assert COCO_IDX["nose"] == NOSE == 0
        assert COCO_IDX["left_shoulder"] == LEFT_SHOULDER == 5
        assert COCO_IDX["right_shoulder"] == RIGHT_SHOULDER == 6
        assert COCO_IDX["left_hip"] == LEFT_HIP == 11
        assert COCO_IDX["right_hip"] == RIGHT_HIP == 12
        assert COCO_IDX["left_knee"] == LEFT_KNEE == 13
        assert COCO_IDX["right_knee"] == RIGHT_KNEE == 14
        assert COCO_IDX["left_ankle"] == LEFT_ANKLE == 15
        assert COCO_IDX["right_ankle"] == RIGHT_ANKLE == 16


class TestComputePIFR:
    """Test compute_pifr function."""

    def test_output_shape(self):
        """PIFR output should be 60 dimensions."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        result = compute_pifr(keypoints, 640, 480)
        assert result.shape == (60,)

    def test_output_dtype(self):
        """Output should be float32."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        result = compute_pifr(keypoints, 640, 480)
        assert result.dtype == np.float32

    def test_none_keypoints_returns_zeros(self):
        """None input should return zeros."""
        result = compute_pifr(None, 640, 480)
        assert result.shape == (60,)
        assert np.allclose(result, 0.0)

    def test_short_keypoints_returns_zeros(self):
        """Keypoints with < 17 points should return zeros."""
        keypoints = np.random.rand(10, 3).astype(np.float32)
        result = compute_pifr(keypoints, 640, 480)
        assert result.shape == (60,)
        assert np.allclose(result, 0.0)

    def test_keypoints_flattened_in_output(self):
        """First 51 values should be flattened keypoints."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 0] = np.arange(17) / 16  # x values
        keypoints[:, 1] = np.arange(17, 34) / 32  # y values
        keypoints[:, 2] = 0.5  # confidence

        result = compute_pifr(keypoints, 1, 1)

        # Check first 51 values (flattened keypoints)
        expected_flat = keypoints.flatten()
        np.testing.assert_allclose(result[:51], expected_flat, rtol=1e-5)

    def test_geometric_features_in_output(self):
        """Last 9 values should be geometric features."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, 2] = 0.9  # High confidence
        result = compute_pifr(keypoints, 640, 480)
        assert len(result) == 60
        # Last 9 values should be angles and positions
        geometric = result[51:]
        assert len(geometric) == 9


class TestCompute9PIFRFeatures:
    """Test compute_9_pifr_features function."""

    def test_output_shape(self):
        """Should output exactly 9 features."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        result = compute_9_pifr_features(keypoints)
        assert result.shape == (9,)

    def test_output_dtype(self):
        """Output should be float32."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        result = compute_9_pifr_features(keypoints)
        assert result.dtype == np.float32

    def test_none_keypoints_returns_zeros(self):
        """None input should return zeros."""
        result = compute_9_pifr_features(None)
        assert result.shape == (9,)
        assert np.allclose(result, 0.0)

    def test_short_keypoints_returns_zeros(self):
        """Keypoints with < 17 points should return zeros."""
        keypoints = np.random.rand(5, 3).astype(np.float32)
        result = compute_9_pifr_features(keypoints)
        assert result.shape == (9,)
        assert np.allclose(result, 0.0)

    def test_center_of_mass_x_bounded(self):
        """Center of mass X should be in [0, 1] for normalized keypoints."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, 2] = 0.5  # Equal weights
        result = compute_9_pifr_features(keypoints)
        assert 0 <= result[0] <= 1

    def test_center_of_mass_y_bounded(self):
        """Center of mass Y should be in [0, 1] for normalized keypoints."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, 2] = 0.5
        result = compute_9_pifr_features(keypoints)
        assert 0 <= result[1] <= 1

    def test_angles_in_valid_range(self):
        """All angle features should be in [0, pi]."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, 2] = 0.9
        result = compute_9_pifr_features(keypoints)

        # Angles are at indices 2-8 (7 angles + 2 center of mass)
        angles = result[2:]
        assert np.all(angles >= 0)
        assert np.all(angles <= np.pi)

    def test_upright_pose_vertical_torso(self):
        """Standing pose should have vertical torso angle."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Standing person: head at top, feet at bottom
        keypoints[NOSE] = [0.5, 0.1, 1.0]
        keypoints[LEFT_SHOULDER] = [0.45, 0.25, 1.0]
        keypoints[RIGHT_SHOULDER] = [0.55, 0.25, 1.0]
        keypoints[LEFT_HIP] = [0.45, 0.5, 1.0]
        keypoints[RIGHT_HIP] = [0.55, 0.5, 1.0]
        keypoints[LEFT_KNEE] = [0.45, 0.7, 1.0]
        keypoints[RIGHT_KNEE] = [0.55, 0.7, 1.0]
        keypoints[LEFT_ANKLE] = [0.45, 0.95, 1.0]
        keypoints[RIGHT_ANKLE] = [0.55, 0.95, 1.0]

        result = compute_9_pifr_features(keypoints)

        # Torso angle (index 3) should be close to 0 (upright)
        assert result[3] < 0.3  # Nearly vertical

    def test_lying_pose_horizontal_torso(self):
        """Lying pose should have horizontal torso angle."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        # Lying person: horizontal body
        keypoints[NOSE] = [0.1, 0.5, 1.0]
        keypoints[LEFT_SHOULDER] = [0.25, 0.5, 1.0]
        keypoints[RIGHT_SHOULDER] = [0.25, 0.5, 1.0]
        keypoints[LEFT_HIP] = [0.75, 0.5, 1.0]
        keypoints[RIGHT_HIP] = [0.75, 0.5, 1.0]
        keypoints[LEFT_KNEE] = [0.85, 0.5, 1.0]
        keypoints[RIGHT_KNEE] = [0.85, 0.5, 1.0]
        keypoints[LEFT_ANKLE] = [0.95, 0.5, 1.0]
        keypoints[RIGHT_ANKLE] = [0.95, 0.5, 1.0]

        result = compute_9_pifr_features(keypoints)

        # Torso angle (index 3) should be close to pi/2 (horizontal)
        assert result[3] > 1.2  # Nearly horizontal

    def test_confidence_weighting(self):
        """Features should weight by confidence."""
        # High confidence on left side
        keypoints = np.ones((17, 3), dtype=np.float32) * 0.5
        keypoints[:8, 2] = 1.0  # High confidence for left body
        keypoints[8:, 2] = 0.1  # Low confidence for right body

        result = compute_9_pifr_features(keypoints)

        # Center of mass should shift toward left
        assert result[0] < 0.5  # Shifted left

    def test_zero_confidence_handled(self):
        """Zero confidence should not cause division by zero."""
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 0] = np.linspace(0, 1, 17)  # x values
        keypoints[:, 1] = np.linspace(0, 1, 17)  # y values
        # All zero confidence

        result = compute_9_pifr_features(keypoints)

        # Should not crash and return valid output
        assert result.shape == (9,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestPIFRConsistency:
    """Test consistency between compute_pifr and compute_9_pifr_features."""

    def test_geometric_features_match(self):
        """Geometric features in compute_pifr should match compute_9_pifr_features."""
        keypoints = np.random.rand(17, 3).astype(np.float32)
        keypoints[:, 2] = 0.8

        pifr_full = compute_pifr(keypoints, 640, 480)
        pifr_9d = compute_9_pifr_features(keypoints)

        np.testing.assert_allclose(pifr_full[51:], pifr_9d, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
