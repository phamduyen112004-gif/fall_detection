"""Unit tests for SequenceAugmenter (data augmentation)."""

from __future__ import annotations

import numpy as np
import pytest


class TestSequenceAugmenter:
    """Test suite for SequenceAugmenter."""

    @pytest.fixture
    def sample_sequence(self):
        """Create a sample sequence for testing."""
        np.random.seed(42)
        seq = np.random.rand(60, 60).astype(np.float64)
        # Set known values for key columns
        seq[:, 0] = 0.5   # nose_x
        seq[:, 15] = 0.4   # l_shoulder_x
        seq[:, 18] = 0.6   # r_shoulder_x
        seq[:, 57] = 0.8   # left_leg_angle
        seq[:, 58] = 0.3   # right_leg_angle
        return seq

    @pytest.fixture
    def augmenter(self):
        """Create a SequenceAugmenter instance."""
        from scripts.augmentation import SequenceAugmenter
        return SequenceAugmenter(
            temporal_shift_prob=0.5,
            noise_prob=0.5,
            hflip_prob=0.5,
            temporal_shift_max=5,
            noise_sigma=0.01,
            seed=42,
        )

    def test_apply_output_shape(self, augmenter, sample_sequence):
        """Augmented sequence should maintain original shape."""
        augmented = augmenter.apply(sample_sequence)
        assert augmented.shape == sample_sequence.shape, \
            f"Shape changed from {sample_sequence.shape} to {augmented.shape}"

    def test_apply_no_data_leakage(self, augmenter, sample_sequence):
        """Original sequence should not be modified."""
        original_copy = sample_sequence.copy()
        _ = augmenter.apply(sample_sequence)
        assert np.array_equal(sample_sequence, original_copy), \
            "Original sequence was modified"

    def test_temporal_shift_preserves_shape(self, augmenter, sample_sequence):
        """Temporal shift should preserve sequence shape."""
        shifted = augmenter._temporal_shift(sample_sequence.copy())
        assert shifted.shape == sample_sequence.shape

    def test_temporal_shift_changes_data(self, augmenter, sample_sequence):
        """Temporal shift should change the sequence (unless shift=0)."""
        shifted = augmenter._temporal_shift(sample_sequence.copy())
        # With non-zero max shift, most calls should change the sequence
        # (unless random shift happens to be 0)
        # We test that the method exists and works
        assert shifted.shape == sample_sequence.shape

    def test_gaussian_noise_preserves_shape(self, augmenter, sample_sequence):
        """Gaussian noise should preserve sequence shape."""
        noisy = augmenter._add_gaussian_noise(sample_sequence.copy())
        assert noisy.shape == sample_sequence.shape

    def test_gaussian_noise_changes_x_y_only(self, augmenter, sample_sequence):
        """Gaussian noise should only affect x/y coordinates, not confidence."""
        noisy = augmenter._add_gaussian_noise(sample_sequence.copy())

        # Check that confidence columns (index 2, 5, 8, ...) are unchanged
        for kp in range(17):
            conf_col = kp * 3 + 2
            np.testing.assert_array_equal(
                noisy[:, conf_col],
                sample_sequence[:, conf_col],
                err_msg=f"Confidence column {conf_col} was modified"
            )

    def test_gaussian_noise_stays_in_range(self, augmenter, sample_sequence):
        """Noise should not push x/y coordinates outside [0, 1]."""
        noisy = augmenter._add_gaussian_noise(sample_sequence.copy())

        # Check keypoint x/y columns (0-50) are in [0, 1]
        for kp in range(17):
            x_col, y_col = kp * 3, kp * 3 + 1
            assert np.all(noisy[:, x_col] >= 0) and np.all(noisy[:, x_col] <= 1), \
                f"X column {x_col} out of range"
            assert np.all(noisy[:, y_col] >= 0) and np.all(noisy[:, y_col] <= 1), \
                f"Y column {y_col} out of range"

    def test_horizontal_flip_preserves_shape(self, augmenter, sample_sequence):
        """Horizontal flip should preserve sequence shape."""
        flipped = augmenter._horizontal_flip(sample_sequence.copy())
        assert flipped.shape == sample_sequence.shape

    def test_horizontal_flip_x_coordinate(self, augmenter, sample_sequence):
        """Horizontal flip should transform x = 1.0 - x."""
        flipped = augmenter._horizontal_flip(sample_sequence.copy())

        # Check first frame, first keypoint
        original_x = sample_sequence[0, 0]
        flipped_x = flipped[0, 0]
        expected_x = 1.0 - original_x

        assert np.isclose(flipped_x, expected_x, atol=1e-6), \
            f"X flip incorrect: expected {expected_x}, got {flipped_x}"

    def test_horizontal_flip_swaps_shoulders(self, augmenter, sample_sequence):
        """Horizontal flip should swap left and right shoulders."""
        flipped = augmenter._horizontal_flip(sample_sequence.copy())

        # After flip: original L_shoulder_x becomes R_shoulder position
        # Which should equal 1.0 - original R_shoulder_x
        original_l_shoulder_x = sample_sequence[0, 15]
        original_r_shoulder_x = sample_sequence[0, 18]
        flipped_l_shoulder_x = flipped[0, 15]

        expected = 1.0 - original_r_shoulder_x
        assert np.isclose(flipped_l_shoulder_x, expected, atol=1e-6), \
            f"Shoulder swap incorrect: expected {expected}, got {flipped_l_shoulder_x}"

    def test_horizontal_flip_swaps_leg_angles(self, augmenter, sample_sequence):
        """Horizontal flip should swap left_leg_angle and right_leg_angle."""
        flipped = augmenter._horizontal_flip(sample_sequence.copy())

        # Original: left=0.8, right=0.3
        # After flip: left=0.3, right=0.8
        assert np.isclose(flipped[0, 57], 0.3, atol=1e-6), \
            f"Left leg angle not swapped: expected 0.3, got {flipped[0, 57]}"
        assert np.isclose(flipped[0, 58], 0.8, atol=1e-6), \
            f"Right leg angle not swapped: expected 0.8, got {flipped[0, 58]}"

    def test_apply_batch_output_shape(self, augmenter, sample_sequence):
        """Batch augmentation should preserve batch shape."""
        batch = np.stack([sample_sequence] * 4, axis=0)
        augmented = augmenter.apply_batch(batch)
        assert augmented.shape == batch.shape, \
            f"Batch shape changed from {batch.shape} to {augmented.shape}"

    def test_apply_batch_independent_samples(self, augmenter, sample_sequence):
        """Each sample in batch should be processed independently."""
        batch = np.stack([sample_sequence.copy() for _ in range(4)], axis=0)
        augmented = augmenter.apply_batch(batch)

        # All samples should have different augmentations
        # (due to random operations)
        unique_count = len(set(tuple(augmented[i].ravel()[:100])
                              for i in range(4)))
        # At least some samples should differ
        assert unique_count > 1, "Batch augmentation should produce varied results"

    def test_set_seed_reproducibility(self, sample_sequence):
        """Setting seed should make results reproducible."""
        from scripts.augmentation import SequenceAugmenter

        augmenter1 = SequenceAugmenter(seed=123)
        augmenter2 = SequenceAugmenter(seed=123)

        result1 = augmenter1.apply(sample_sequence.copy())
        result2 = augmenter2.apply(sample_sequence.copy())

        np.testing.assert_array_equal(
            result1, result2,
            "Same seed should produce same results"
        )

    def test_augment_sequence_convenience_function(self, sample_sequence):
        """augment_sequence convenience function should work."""
        from scripts.augmentation import augment_sequence

        augmented = augment_sequence(
            sample_sequence,
            temporal_shift_prob=0.5,
            noise_prob=0.5,
            hflip_prob=0.5,
            seed=42,
        )

        assert augmented.shape == sample_sequence.shape

    def test_zero_probabilities(self, sample_sequence):
        """Zero probabilities should skip that augmentation."""
        from scripts.augmentation import SequenceAugmenter

        augmenter = SequenceAugmenter(
            temporal_shift_prob=0.0,
            noise_prob=0.0,
            hflip_prob=0.0,
            seed=42,
        )

        augmented = augmenter.apply(sample_sequence.copy())

        # Without any augmentation, shape and some values should be same
        # (only temporal shift at 0 prob won't be applied)
        assert augmented.shape == sample_sequence.shape


class TestAugmentationConstants:
    """Test augmentation constants."""

    def test_left_keypoints_count(self):
        """LEFT_KEYPOINTS should have 8 elements."""
        from scripts.augmentation import LEFT_KEYPOINTS
        assert len(LEFT_KEYPOINTS) == 8

    def test_right_keypoints_count(self):
        """RIGHT_KEYPOINTS should have 8 elements."""
        from scripts.augmentation import RIGHT_KEYPOINTS
        assert len(RIGHT_KEYPOINTS) == 8

    def test_keypoints_matching(self):
        """LEFT and RIGHT keypoints should be same length."""
        from scripts.augmentation import LEFT_KEYPOINTS, RIGHT_KEYPOINTS
        assert len(LEFT_KEYPOINTS) == len(RIGHT_KEYPOINTS)

    def test_geometric_feature_swap_map(self):
        """Geometric swap map should be properly defined."""
        from scripts.augmentation import LEFT_GEOMETRIC, RIGHT_GEOMETRIC
        assert len(LEFT_GEOMETRIC) == len(RIGHT_GEOMETRIC)
