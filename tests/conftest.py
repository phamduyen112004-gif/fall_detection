"""
Pytest configuration and fixtures for fall-detection tests.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def sample_keypoints():
    """Generate sample keypoints for testing."""
    np.random.seed(42)
    keypoints = np.random.rand(17, 3).astype(np.float32)
    keypoints[:, 2] = 0.8  # High confidence
    return keypoints


@pytest.fixture
def upright_pose_keypoints():
    """Generate keypoints representing a standing person."""
    from src.pifr_features import (
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE
    )

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
    return keypoints


@pytest.fixture
def lying_pose_keypoints():
    """Generate keypoints representing a lying person."""
    from src.pifr_features import (
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE
    )

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
    return keypoints


@pytest.fixture
def zero_keypoints():
    """Generate keypoints with all zeros."""
    return np.zeros((17, 3), dtype=np.float32)


@pytest.fixture
def sample_video_features():
    """Generate sample video features (60 frames, 60 features)."""
    np.random.seed(42)
    return np.random.rand(60, 60).astype(np.float32)
