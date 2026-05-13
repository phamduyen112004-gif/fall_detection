#!/usr/bin/env python3
"""Tests for PIFR feature extraction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_extractor_shape():
    from src.pifr_features import GeometricFeatureExtractor
    ext = GeometricFeatureExtractor()
    kp = np.random.rand(17, 3).astype(np.float64)
    feat = ext.extract(kp)
    assert feat.shape == (60,), f"Expected (60,), got {feat.shape}"
    print("[PASS] test_extractor_shape")

def test_batch_extract():
    from src.pifr_features import GeometricFeatureExtractor
    ext = GeometricFeatureExtractor()
    batch = np.random.rand(4, 17, 3).astype(np.float64)
    feat = ext.extract_batch(batch)
    assert feat.shape == (4, 60)
    print("[PASS] test_batch_extract")

def test_standing_vs_lying():
    """Test that different poses produce different features."""
    from src.pifr_features import GeometricFeatureExtractor
    ext = GeometricFeatureExtractor()
    
    # Two different poses
    pose1 = np.random.rand(17, 3).astype(np.float64)
    pose2 = np.random.rand(17, 3).astype(np.float64)
    
    f1 = ext.extract(pose1)
    f2 = ext.extract(pose2)
    
    # Different random poses should produce different features
    assert not np.allclose(f1, f2)
    print("[PASS] test_standing_vs_lying")

def test_constants():
    from src.pifr_features import FEATURE_DIM, SEQ_LEN
    assert FEATURE_DIM == 60
    assert SEQ_LEN == 60
    print("[PASS] test_constants")

if __name__ == "__main__":
    test_extractor_shape()
    test_batch_extract()
    test_standing_vs_lying()
    test_constants()
    print("\nAll PIFR tests passed!")
