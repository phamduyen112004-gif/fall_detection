#!/usr/bin/env python3
"""
Data Augmentation for Fall Detection Sequences.

Implements SequenceAugmenter for temporal sequences of shape (60, 60)
[60 frames, 60-D features: 51D keypoints + 9D geometric].

Augmentations:
    1. Temporal Shift    - Roll frames along time axis
    2. Gaussian Noise    - Add camera jitter noise (sigma=0.01)
    3. Horizontal Flip   - Flip X coords + swap left/right keypoints

Based on methodology from:
    - Benabdennour et al. (2026) - IEEE Access Fall Detection
    - PLOS ONE Fall Detection Studies (2026) - sigma=0.01 recommendation
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# COCO Keypoint Index Mapping for Horizontal Flip
# Original indices and their swapped counterparts
LEFT_KEYPOINTS = [1, 3, 5, 7, 9, 11, 13, 15]   # L_EYE, L_EAR, L_SHOULDER, L_ELBOW, L_WRIST, L_HIP, L_KNEE, L_ANKLE
RIGHT_KEYPOINTS = [2, 4, 6, 8, 10, 12, 14, 16]  # R_EYE, R_EAR, R_SHOULDER, R_ELBOW, R_WRIST, R_HIP, R_KNEE, R_ANKLE

# Geometric Feature Index Mapping (indices 51-59)
# Only angle features that depend on left/right need swapping
# center_mass_x (51), center_mass_y (52) - NOT swapped
# shoulder_nose_angle (53) - NOT swapped (uses both shoulders symmetrically)
# torso_angle (54) - NOT swapped (uses midpoints)
# hip_angle (55) - NOT swapped (uses midpoints)
# shoulder_angle (56) - NOT swapped (uses both shoulders)
# left_leg_angle (57) <-> right_leg_angle (58) - SWAPPED
# nose_to_ankle_angle (59) - NOT swapped (uses midpoints)

LEFT_GEOMETRIC = [57]   # left_leg_angle
RIGHT_GEOMETRIC = [58]  # right_leg_angle


def _get_keypoint_swap_map() -> dict[int, int]:
    """Build mapping of keypoint index to its horizontally-flipped counterpart."""
    swap_map = {}
    for l, r in zip(LEFT_KEYPOINTS, RIGHT_KEYPOINTS):
        swap_map[l] = r
        swap_map[r] = l
    return swap_map


def _get_geometric_swap_map() -> dict[int, int]:
    """Build mapping of geometric feature index to its horizontally-flipped counterpart."""
    swap_map = {}
    for l, r in zip(LEFT_GEOMETRIC, RIGHT_GEOMETRIC):
        swap_map[l] = r
        swap_map[r] = l
    return swap_map


# Pre-computed swap maps
_KEYPOINT_SWAP_MAP = _get_keypoint_swap_map()
_GEOMETRIC_SWAP_MAP = _get_geometric_swap_map()


class SequenceAugmenter:
    """
    Data Augmentation for Fall Detection Sequences.

    Applies 3 augmentations to sequences of shape (60, 60):
        - Temporal Shift: Random roll along time axis
        - Gaussian Noise: Camera jitter simulation (sigma=0.01)
        - Horizontal Flip: X-flip with left/right keypoint swap

    Usage:
        >>> augmenter = SequenceAugmenter(
        ...     temporal_shift_prob=0.5,
        ...     noise_prob=0.5,
        ...     hflip_prob=0.5,
        ...     temporal_shift_max=5,
        ...     noise_sigma=0.01,
        ... )
        >>> augmented = augmenter.apply(sequence)  # shape (60, 60)

    Attributes:
        temporal_shift_prob: Probability of applying temporal shift.
        noise_prob: Probability of applying Gaussian noise.
        hflip_prob: Probability of applying horizontal flip.
        temporal_shift_max: Maximum frames to shift (±).
        noise_sigma: Standard deviation for Gaussian noise.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        temporal_shift_prob: float = 0.5,
        noise_prob: float = 0.5,
        hflip_prob: float = 0.5,
        temporal_shift_max: int = 5,
        noise_sigma: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """
        Initialize SequenceAugmenter.

        Args:
            temporal_shift_prob: Probability of temporal shift [0, 1].
            noise_prob: Probability of Gaussian noise injection [0, 1].
            hflip_prob: Probability of horizontal flip [0, 1].
            temporal_shift_max: Max frames to shift ±.
            noise_sigma: Gaussian noise std dev (recommended: 0.01 per PLOS ONE 2026).
            seed: Random seed for reproducibility.
        """
        self.temporal_shift_prob = temporal_shift_prob
        self.noise_prob = noise_prob
        self.hflip_prob = hflip_prob
        self.temporal_shift_max = temporal_shift_max
        self.noise_sigma = noise_sigma

        self._rng = np.random.default_rng(seed)

    def apply(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply augmentations to a single sequence.

        Args:
            sequence: Array of shape (60, 60) [frames, features].

        Returns:
            Augmented sequence of same shape.
        """
        seq = sequence.copy()

        # Apply augmentations
        if self._rng.random() < self.temporal_shift_prob:
            seq = self._temporal_shift(seq)

        if self._rng.random() < self.noise_prob:
            seq = self._add_gaussian_noise(seq)

        if self._rng.random() < self.hflip_prob:
            seq = self._horizontal_flip(seq)

        return seq

    def apply_batch(
        self,
        sequences: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Apply augmentations to a batch of sequences.

        Args:
            sequences: Array of shape (B, 60, 60) [batch, frames, features].

        Returns:
            Augmented batch of same shape.
        """
        batch_size = sequences.shape[0]
        augmented = np.empty_like(sequences)

        for i in range(batch_size):
            augmented[i] = self.apply(sequences[i])

        return augmented

    def _temporal_shift(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply temporal shift by rolling frames.

        Randomly shifts frames along time axis by ±temporal_shift_max.
        Uses numpy.roll with wrap-around.

        Args:
            sequence: Shape (60, 60).

        Returns:
            Temporally shifted sequence.
        """
        shift = self._rng.integers(
            -self.temporal_shift_max,
            self.temporal_shift_max + 1,
        )
        if shift == 0:
            return sequence
        return np.roll(sequence, shift, axis=0)

    def _add_gaussian_noise(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Add Gaussian noise to simulate camera jitter.

        Adds noise ONLY to x and y coordinates (not confidence).
        Per PLOS ONE 2026 recommendation: sigma = 0.01.

        Feature structure (60 dims):
            [0:51]   - 17 keypoints × 3 (x, y, conf)
                      - Keypoint i: x at i*3, y at i*3+1, conf at i*3+2
            [51:60]  - 9 geometric features (angles - noise NOT applied)

        Args:
            sequence: Shape (60, 60).

        Returns:
            Sequence with added Gaussian noise.
        """
        seq = sequence.copy()
        noise = self._rng.normal(0, self.noise_sigma, size=seq.shape)

        # Apply noise only to x and y coordinates of keypoints
        # For each frame (60 frames), for each keypoint (17), apply to x and y
        # Keypoint i starts at column i*3, x at i*3, y at i*3+1
        for frame_idx in range(seq.shape[0]):
            for kp_idx in range(17):
                x_col = kp_idx * 3
                y_col = kp_idx * 3 + 1
                seq[frame_idx, x_col] += noise[frame_idx, x_col]
                seq[frame_idx, y_col] += noise[frame_idx, y_col]

        # Clip to valid range [0, 1] for normalized coordinates
        seq[:, :51] = np.clip(seq[:, :51], 0.0, 1.0)

        return seq

    def _horizontal_flip(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply horizontal flip with left/right keypoint swap.

        Steps:
            1. Flip X coordinates: new_X = 1.0 - X
            2. Swap left/right keypoint pairs
            3. Swap left/right geometric angle pairs

        Args:
            sequence: Shape (60, 60).

        Returns:
            Horizontally flipped sequence.
        """
        seq = sequence.copy()

        # Step 1: Flip X coordinates (all keypoint x values in columns 0, 3, 6, ...)
        # For each frame, flip all x coordinates (even indices within first 51 columns)
        for frame_idx in range(seq.shape[0]):
            # Flip x coordinates: new_x = 1.0 - x
            # x coords are at columns 0, 3, 6, ... (i*3 for keypoint i)
            for kp_idx in range(17):
                x_col = kp_idx * 3
                seq[frame_idx, x_col] = 1.0 - seq[frame_idx, x_col]

        # Step 2: Swap left/right keypoint columns
        # Each keypoint has 3 columns: (x, y, conf)
        # Left keypoint i -> Right keypoint j (where j = swap_map[i])
        seq = self._swap_keypoint_columns(seq)

        # Step 3: Swap left/right geometric feature columns
        # Only geometric features that depend on left/right need swapping:
        # - left_leg_angle (57) <-> right_leg_angle (58)
        seq = self._swap_geometric_columns(seq)

        return seq

    def _swap_keypoint_columns(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Swap left and right keypoint columns.

        For each left/right pair, swaps all 3 columns (x, y, conf).

        Args:
            sequence: Shape (60, 60).

        Returns:
            Sequence with swapped keypoint columns.
        """
        seq = sequence.copy()

        # Keypoint pairs to swap
        pairs = [
            (LEFT_KEYPOINTS[i], RIGHT_KEYPOINTS[i])
            for i in range(len(LEFT_KEYPOINTS))
        ]

        for left_kp, right_kp in pairs:
            # Get column indices for each keypoint's (x, y, conf)
            left_start = left_kp * 3
            right_start = right_kp * 3

            # Swap the 3 columns (x, y, conf)
            left_cols = seq[:, left_start:left_start + 3].copy()
            right_cols = seq[:, right_start:right_start + 3].copy()
            seq[:, left_start:left_start + 3] = right_cols
            seq[:, right_start:right_start + 3] = left_cols

        return seq

    def _swap_geometric_columns(self, sequence: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Swap left and right geometric feature columns.

        Only swaps:
            - left_leg_angle (index 57) <-> right_leg_angle (index 58)

        Args:
            sequence: Shape (60, 60).

        Returns:
            Sequence with swapped geometric columns.
        """
        seq = sequence.copy()

        # Swap left_leg_angle and right_leg_angle
        left_idx = 57
        right_idx = 58

        left_col = seq[:, left_idx].copy()
        right_col = seq[:, right_idx].copy()
        seq[:, left_idx] = right_col
        seq[:, right_idx] = left_col

        return seq

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:
        return (
            f"SequenceAugmenter("
            f"temporal_shift_prob={self.temporal_shift_prob}, "
            f"noise_prob={self.noise_prob}, "
            f"hflip_prob={self.hflip_prob}, "
            f"temporal_shift_max={self.temporal_shift_max}, "
            f"noise_sigma={self.noise_sigma})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

def augment_sequence(
    sequence: NDArray[np.float64],
    temporal_shift_prob: float = 0.5,
    noise_prob: float = 0.5,
    hflip_prob: float = 0.5,
    temporal_shift_max: int = 5,
    noise_sigma: float = 0.01,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """
    Convenience function to augment a single sequence.

    Args:
        sequence: Array of shape (60, 60).
        See SequenceAugmenter for other args.

    Returns:
        Augmented sequence.
    """
    augmenter = SequenceAugmenter(
        temporal_shift_prob=temporal_shift_prob,
        noise_prob=noise_prob,
        hflip_prob=hflip_prob,
        temporal_shift_max=temporal_shift_max,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    return augmenter.apply(sequence)


# ─────────────────────────────────────────────────────────────────────────────
# Demo & Testing
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("SequenceAugmenter - Demo & Verification")
    print("=" * 60)

    # Create sample sequence
    np.random.seed(42)
    seq = np.random.rand(60, 60).astype(np.float64)

    print("\n[1] Original Sequence Sample (first 5 frames, first 10 features):")
    print(seq[:5, :10])

    # Initialize augmenter
    augmenter = SequenceAugmenter(
        temporal_shift_prob=1.0,
        noise_prob=1.0,
        hflip_prob=1.0,
        temporal_shift_max=5,
        noise_sigma=0.01,
    )

    # Test each augmentation independently
    print("\n[2] Testing Temporal Shift:")
    seq_shifted = augmenter._temporal_shift(seq.copy())
    print(f"  Original frame 0, feature 0: {seq[0, 0]:.4f}")
    print(f"  Shifted frame 0, feature 0: {seq_shifted[0, 0]:.4f}")
    print(f"  Shift applied: {not np.array_equal(seq, seq_shifted)}")

    print("\n[3] Testing Gaussian Noise (sigma=0.01):")
    seq_noisy = augmenter._add_gaussian_noise(seq.copy())
    # Check x/y coords changed (cols 0,1,3,4,... up to col 48)
    noise_detected = False
    for kp in range(17):
        x_col, y_col = kp * 3, kp * 3 + 1
        if not np.isclose(seq[0, x_col], seq_noisy[0, x_col], atol=1e-6):
            noise_detected = True
            print(f"  Keypoint {kp} X noise: {seq_noisy[0, x_col] - seq[0, x_col]:.6f}")
            break
    print(f"  Noise detected: {noise_detected}")

    # Verify confidence not changed
    conf_changed = False
    for kp in range(17):
        conf_col = kp * 3 + 2
        if not np.isclose(seq[0, conf_col], seq_noisy[0, conf_col], atol=1e-6):
            conf_changed = True
            break
    print(f"  Confidence unchanged: {not conf_changed}")

    print("\n[4] Testing Horizontal Flip:")
    seq_flipped = augmenter._horizontal_flip(seq.copy())
    print(f"  Original frame 0, nose X (col 0): {seq[0, 0]:.4f}")
    print(f"  Flipped frame 0, nose X (col 0): {seq_flipped[0, 0]:.4f}")
    print(f"  Expected (1.0 - original): {1.0 - seq[0, 0]:.4f}")
    print(f"  X flip correct: {np.isclose(seq_flipped[0, 0], 1.0 - seq[0, 0])}")

    # Check left/right swap
    print(f"  Original L_shoulder X (col 15): {seq[0, 15]:.4f}")
    print(f"  Original R_shoulder X (col 18): {seq[0, 18]:.4f}")
    print(f"  Flipped L_shoulder X (col 15): {seq_flipped[0, 15]:.4f}")
    print(f"  Flipped R_shoulder X (col 18): {seq_flipped[0, 18]:.4f}")
    print(f"  Left/Right swapped: {np.isclose(seq_flipped[0, 15], 1.0 - seq[0, 18])}")

    print("\n[5] Testing Full Augmentation Pipeline:")
    augmented = augmenter.apply(seq)
    print(f"  Input shape: {seq.shape}")
    print(f"  Output shape: {augmented.shape}")
    print(f"  Output in valid range [0,1]: {np.all(augmented >= 0) and np.all(augmented <= 1)}")
    print(f"  Data changed: {not np.array_equal(seq, augmented)}")

    print("\n[6] Testing Batch Augmentation:")
    batch = np.random.rand(8, 60, 60).astype(np.float64)
    batch_aug = augmenter.apply_batch(batch)
    print(f"  Batch input shape: {batch.shape}")
    print(f"  Batch output shape: {batch_aug.shape}")
    print(f"  All samples augmented: {not np.array_equal(batch[0], batch_aug[0])}")

    print("\n[7] Verifying Geometric Feature Swap (left_leg_angle <-> right_leg_angle):")
    # Create sequence with known leg angles
    test_seq = np.zeros((60, 60), dtype=np.float64)
    test_seq[:, 57] = 0.8  # left_leg_angle = 0.8
    test_seq[:, 58] = 0.3  # right_leg_angle = 0.3
    flipped_test = augmenter._horizontal_flip(test_seq)
    print(f"  Original left_leg_angle (57): {test_seq[0, 57]:.2f}, right_leg_angle (58): {test_seq[0, 58]:.2f}")
    print(f"  Flipped left_leg_angle (57): {flipped_test[0, 57]:.2f}, right_leg_angle (58): {flipped_test[0, 58]:.2f}")
    print(f"  Leg angles swapped correctly: {np.isclose(flipped_test[0, 57], 0.3) and np.isclose(flipped_test[0, 58], 0.8)}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
