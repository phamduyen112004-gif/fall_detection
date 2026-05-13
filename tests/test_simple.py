#!/usr/bin/env python3
"""Simple test suite for Fall Detection components."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

def test_pifr_extractor():
    """Test PIFR feature extraction."""
    from src.pifr_features import GeometricFeatureExtractor, FEATURE_DIM, SEQ_LEN
    
    extractor = GeometricFeatureExtractor()
    keypoints = np.random.rand(17, 3).astype(np.float64)
    features = extractor.extract(keypoints)
    
    assert features.shape == (60,), f"Expected (60,), got {features.shape}"
    print(f"[PASS] PIFR extractor: shape={features.shape}")
    return True

def test_augmentation():
    """Test augmentation."""
    from scripts.augmentation import SequenceAugmenter
    
    augmenter = SequenceAugmenter(seed=42)
    seq = np.random.rand(60, 60).astype(np.float64)
    augmented = augmenter.apply(seq)
    
    assert augmented.shape == seq.shape, f"Shape mismatch"
    print(f"[PASS] Augmentation: shape={augmented.shape}")
    return True

def test_transformer():
    """Test transformer model."""
    from src.hybrid_fall_transformer import HybridFallTransformer
    
    model = HybridFallTransformer()
    x = torch.randn(2, 60, 60)
    output = model(x)
    
    assert output.shape == (2, 1), f"Expected (2, 1), got {output.shape}"
    print(f"[PASS] Transformer: output_shape={output.shape}")
    return True

def main():
    print("=" * 50)
    print("Running Fall Detection Tests")
    print("=" * 50)
    
    tests = [test_pifr_extractor, test_augmentation, test_transformer]
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
