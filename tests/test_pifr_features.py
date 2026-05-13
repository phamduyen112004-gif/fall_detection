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
            [0.90, 0.50, 0.95], [0.92, 0.49, 0.90], [0.94, 0.51, 0.92],
            [0.95, 0.48, 0.85], [0.96, 0.52, 0.88], [0.75, 0.45, 0.95],
            [0.75, 0.55, 0.95], [0.60, 0.42, 0.90], [0.60, 0.58, 0.88],
            [0.45, 0.40, 0.85], [0.45, 0.60, 0.82], [0.40, 0.45, 0.95],
            [0.40, 0.55, 0.95], [0.20, 0.43, 0.90], [0.20, 0.57, 0.90],
            [0.05, 0.45, 0.88], [0.05, 0.55, 0.87],
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
        assert features[54] > 0.7  # torso_angle index

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
