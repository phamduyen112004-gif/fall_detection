#!/usr/bin/env python3
"""Sanity check for Kaggle notebook - verifies all imports and basic functionality."""

import sys

def check_imports():
    """Verify all modules can be imported."""
    print("Checking imports...")
    
    try:
        import numpy as np
        print(f"  numpy {np.__version__}")
    except ImportError:
        print("  [FAIL] numpy")
        return False
    
    try:
        import torch
        print(f"  torch {torch.__version__}")
    except ImportError:
        print("  [FAIL] torch")
        return False
    
    try:
        import cv2
        print(f"  opencv {cv2.__version__}")
    except ImportError:
        print("  [FAIL] opencv")
        return False
    
    try:
        from src.pifr_features import GeometricFeatureExtractor, extract_pifr_features
        print("  src.pifr_features")
    except ImportError as e:
        print(f"  [FAIL] src.pifr_features: {e}")
        return False
    
    try:
        from src.hybrid_fall_transformer import HybridFallTransformer
        print("  src.hybrid_fall_transformer")
    except ImportError as e:
        print(f"  [FAIL] src.hybrid_fall_transformer: {e}")
        return False
    
    try:
        from src.stage3_kinematics import compute_pose_angles, classify_posture
        print("  src.stage3_kinematics")
    except ImportError as e:
        print(f"  [FAIL] src.stage3_kinematics: {e}")
        return False
    
    try:
        from scripts.augmentation import SequenceAugmenter
        print("  scripts.augmentation")
    except ImportError as e:
        print(f"  [FAIL] scripts.augmentation: {e}")
        return False
    
    print("All imports OK\n")
    return True

def check_feature_extraction():
    """Verify feature extraction works."""
    print("Checking feature extraction...")
    
    from src.pifr_features import GeometricFeatureExtractor
    import numpy as np
    
    ext = GeometricFeatureExtractor()
    kp = np.random.rand(17, 3).astype(np.float64)
    feat = ext.extract(kp)
    
    assert feat.shape == (60,), f"Expected (60,), got {feat.shape}"
    print(f"  Feature shape: {feat.shape} OK\n")
    return True

def check_transformer():
    """Verify transformer model."""
    print("Checking transformer model...")
    
    from src.hybrid_fall_transformer import HybridFallTransformer
    import torch
    
    model = HybridFallTransformer()
    x = torch.randn(2, 60, 60)
    out = model(x)
    
    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    print(f"  Model output shape: {out.shape} OK\n")
    return True

def check_augmentation():
    """Verify augmentation."""
    print("Checking augmentation...")
    
    from scripts.augmentation import SequenceAugmenter
    import numpy as np
    
    aug = SequenceAugmenter(seed=42)
    seq = np.random.rand(60, 60)
    aug_seq = aug.apply(seq)
    
    assert aug_seq.shape == (60, 60)
    print(f"  Augmented shape: {aug_seq.shape} OK\n")
    return True

def main(strict: bool = False):
    """Run all sanity checks."""
    print("=" * 50)
    print("FALL DETECTION SANITY CHECK")
    print("=" * 50 + "\n")
    
    checks = [
        ("Imports", check_imports),
        ("Feature Extraction", check_feature_extraction),
        ("Transformer", check_transformer),
        ("Augmentation", check_augmentation),
    ]
    
    results = []
    for name, fn in checks:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            results.append((name, False))
    
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False
    
    print("=" * 50)
    
    if all_pass:
        print("All checks passed!")
        return 0
    else:
        print("Some checks FAILED!")
        if strict:
            return 1
    return 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    
    sys.exit(main(strict=args.strict))
