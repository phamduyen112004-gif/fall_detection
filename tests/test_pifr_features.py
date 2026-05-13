"""Unit tests for GeometricFeatureExtractor (PIFR features)."""

from __future__ import annotations

import numpy as np
import pytest


class TestGeometricFeatureExtractor:
    """Test suite for GeometricFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a fresh extractor instance for each test."""
        from src.pifr_features import GeometricFeatureExtractor
        return GeometricFeatureExtractor(conf_threshold=0.2, normalize=True)

    @pytest.fixture
    def standing_pose(self):
        """Create a standing pose (upright)."""
        return np.array([
            [0.50, 0.15, 0.95], [0.48, 0.14, 0.90], [0.52, 0.14, 0.92],
            [0.47, 0.13, 0.85], [0.53, 0.13, 0.88], [0.40, 0.30, 0.95],
            [0.60, 0.30, 0.95], [0.35, 0.45, 0.90], [0.65, 0.45, 0.88],
            [0.32, 0.60, 0.85], [0.68, 0.60, 0.82], [0.43, 0.55, 0.95],
            [0.57, 0.55, 0.95], [0.44, 0.75, 0.90], [0.56, 0.75, 0.90],
            [0.43, 0.95, 0.88], [0.57, 0.95, 0.87],
        ], dtype=np.float64)

    @pytest.fixture
    def lying_pose(self):
        """Create a lying/fallen pose (horizontal)."""
        return np.array([
            [0.95, 0.30, 0.95], [0.97, 0.32, 0.90], [0.98, 0.28, 0.92],
            [0.98, 0.30, 0.85], [0.99, 0.34, 0.88], [0.80, 0.35, 0.95],
            [0.80, 0.45, 0.95], [0.70, 0.30, 0.90], [0.70, 0.50, 0.88],
            [0.60, 0.28, 0.85], [0.60, 0.52, 0.82], [0.55, 0.32, 0.95],
            [0.55, 0.48, 0.95], [0.45, 0.30, 0.90], [0.45, 0.50, 0.90],
            [0.35, 0.32, 0.88], [0.35, 0.48, 0.87],
        ], dtype=np.float64)

    def test_extract_output_shape(self, extractor, standing_pose):
        """Feature vector should have shape (60,)."""
        features = extractor.extract(standing_pose)
        assert features.shape == (60,)

    def test_extract_standing_pose_low_torso_angle(self, extractor, standing_pose):
        """Standing pose should have low torso angle."""
        features = extractor.extract(standing_pose)
        assert features[54] < 0.3  # torso_angle index

    def test_extract_lying_pose_high_torso_angle(self, extractor, lying_pose):
        """Lying pose should have high torso angle."""
        features = extractor.extract(lying_pose)
        # Lying pose should have significantly higher torso angle than standing
        assert features[54] > 0.4  # torso_angle index

    def test_extract_batch(self, extractor, standing_pose, lying_pose):
        """Batch extraction should work correctly."""
        batch = np.stack([standing_pose, lying_pose], axis=0)
        features = extractor.extract_batch(batch)
        assert features.shape == (2, 60)

    def test_invalid_keypoints_shape(self, extractor):
        """Should raise ValueError for invalid keypoints shape."""
        with pytest.raises(ValueError, match="Expected keypoints shape"):
            extractor.extract(np.random.rand(10, 3))

    def test_invalid_batch_shape(self, extractor):
        """Should raise ValueError for invalid batch shape."""
        with pytest.raises(ValueError, match="Expected shape"):
            extractor.extract_batch(np.random.rand(2, 10, 3))

    def test_feature_names_length(self, extractor):
        """Feature names should have exactly 60 entries."""
        names = extractor.get_feature_names()
        assert len(names) == 60

    def test_standing_vs_lying_difference(self, extractor, standing_pose, lying_pose):
        """Standing and lying poses should produce different torso angles."""
        feat_standing = extractor.extract(standing_pose)
        feat_lying = extractor.extract(lying_pose)
        diff = abs(feat_standing[54] - feat_lying[54])
        assert diff > 0.3

    def test_no_nan_in_output(self, extractor, standing_pose):
        """Features should not contain NaN."""
        features = extractor.extract(standing_pose)
        assert not np.any(np.isnan(features))


class TestGeometricFeatureConstants:
    """Test constants and module-level functions."""

    def test_feature_dim_constant(self):
        """FEATURE_DIM should be 60."""
        from src.pifr_features import FEATURE_DIM
        assert FEATURE_DIM == 60

    def test_seq_len_constant(self):
        """SEQ_LEN should be 60."""
        from src.pifr_features import SEQ_LEN
        assert SEQ_LEN == 60

    def test_get_default_extractor(self):
        """Default extractor should return a GeometricFeatureExtractor."""
        from src.pifr_features import get_default_extractor
        extractor = get_default_extractor()
        assert extractor is not None

    def test_extract_pifr_features_function(self):
        """extract_pifr_features should work as a convenience function."""
        from src.pifr_features import extract_pifr_features
        keypoints = np.random.rand(17, 3)
        features = extract_pifr_features(keypoints)
        assert features.shape == (60,)


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_extract_output_shape(self):
        """Test that a dummy (17, 3) keypoint array returns exactly (60,) feature vector.

        This is the core contract of the PIFR feature extractor.
        """
        from src.pifr_features import GeometricFeatureExtractor
        extractor = GeometricFeatureExtractor()

        # Dummy (17, 3) keypoint array - standard COCO format
        dummy_keypoints = np.random.rand(17, 3).astype(np.float64)
        features = extractor.extract(dummy_keypoints)

        # Feature vector MUST be exactly (60,)
        assert features.shape == (60,), f"Expected (60,), got {features.shape}"

    def test_invalid_keypoints_all_zeros(self):
        """Test that all-zero keypoints don't crash the extractor (handles division by zero).

        The extractor should handle edge cases gracefully without raising exceptions.
        """
        from src.pifr_features import GeometricFeatureExtractor, EPS
        extractor = GeometricFeatureExtractor()

        # All zeros keypoints - edge case that could cause division by zero
        zero_keypoints = np.zeros((17, 3), dtype=np.float64)
        zero_keypoints[:, 2] = 0.0  # Zero confidence

        # Should NOT crash - extractor must handle this gracefully
        try:
            features = extractor.extract(zero_keypoints)
            # Even with zero input, should return valid shape
            assert features.shape == (60,)
            # Features may contain zeros/NaN but should not raise
        except (ZeroDivisionError, FloatingPointError):
            pytest.fail("Extractor crashed on all-zero keypoints (division by zero)")

    def test_low_confidence_keypoints(self):
        """Test that low confidence keypoints are handled correctly."""
        from src.pifr_features import GeometricFeatureExtractor
        extractor = GeometricFeatureExtractor()

        # Keypoints with very low confidence
        keypoints = np.random.rand(17, 3).astype(np.float64)
        keypoints[:, 2] = 0.05  # All low confidence

        features = extractor.extract(keypoints)
        assert features.shape == (60,)
        # Should not crash

    def test_partial_valid_keypoints(self):
        """Test extraction with some valid and some invalid keypoints."""
        from src.pifr_features import GeometricFeatureExtractor
        extractor = GeometricFeatureExtractor()

        # Mix of valid and invalid keypoints
        keypoints = np.random.rand(17, 3).astype(np.float64)
        keypoints[:5, 2] = 0.0  # First 5 invalid (low confidence)
        keypoints[5:, 2] = 0.9  # Rest valid

        features = extractor.extract(keypoints)
        assert features.shape == (60,)

    def test_frame_to_vector_60(self):
        """Test frame_to_vector_60 utility function."""
        from src.pifr_features import frame_to_vector_60
        import numpy as np

        keypoints = np.random.rand(17, 3).astype(np.float64)
        vec = frame_to_vector_60(keypoints)

        assert vec.shape == (60,), f"Expected (60,), got {vec.shape}"

    def test_resample_to_length(self):
        """Test resample_to_length utility function."""
        from src.pifr_features import resample_to_length
        import numpy as np

        # Test upsampling
        seq_short = np.random.rand(30, 60)
        resampled = resample_to_length(seq_short, 60)
        assert resampled.shape == (60, 60)

        # Test downsampling
        seq_long = np.random.rand(120, 60)
        resampled = resample_to_length(seq_long, 60)
        assert resampled.shape == (60, 60)

        # Test exact length (no change)
        seq_exact = np.random.rand(60, 60)
        resampled = resample_to_length(seq_exact, 60)
        assert resampled.shape == (60, 60)
