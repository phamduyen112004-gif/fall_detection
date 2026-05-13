#!/usr/bin/env python3
"""Sanity check cho Kaggle notebook - kiểm tra tất cả imports và chức năng cơ bản.

Kiểm tra:
  - Import các module chính
  - Feature extraction hoạt động
  - Transformer model hoạt động
  - Augmentation pipeline hoạt động
"""

import sys


def kiem_tra_imports():
    """Kiểm tra tất cả modules có thể import được."""
    print("Đang kiểm tra imports...")

    try:
        import numpy as np
        print(f"  numpy {np.__version__}")
    except ImportError:
        print("  [LOI] numpy")
        return False

    try:
        import torch
        print(f"  torch {torch.__version__}")
    except ImportError:
        print("  [LOI] torch")
        return False

    try:
        import cv2
        print(f"  opencv {cv2.__version__}")
    except ImportError:
        print("  [LOI] opencv")
        return False

    try:
        from src.pifr_features import GeometricFeatureExtractor, extract_pifr_features
        print("  src.pifr_features")
    except ImportError as e:
        print(f"  [LOI] src.pifr_features: {e}")
        return False

    try:
        from src.hybrid_fall_transformer import HybridFallTransformer
        print("  src.hybrid_fall_transformer")
    except ImportError as e:
        print(f"  [LOI] src.hybrid_fall_transformer: {e}")
        return False

    try:
        from src.stage3_kinematics import compute_pose_angles, classify_posture
        print("  src.stage3_kinematics")
    except ImportError as e:
        print(f"  [LOI] src.stage3_kinematics: {e}")
        return False

    try:
        from scripts.augmentation import SequenceAugmenter
        print("  scripts.augmentation")
    except ImportError as e:
        print(f"  [LOI] scripts.augmentation: {e}")
        return False

    print("Tat ca imports OK\n")
    return True


def kiem_tra_trich_xuat_dac_trung():
    """Kiểm tra feature extraction hoạt động đúng."""
    print("Đang kiểm tra feature extraction...")

    from src.pifr_features import GeometricFeatureExtractor
    import numpy as np

    ext = GeometricFeatureExtractor()
    kp = np.random.rand(17, 3).astype(np.float64)
    feat = ext.extract(kp)

    assert feat.shape == (60,), f"Expected (60,), got {feat.shape}"
    print(f"  Feature shape: {feat.shape} OK\n")
    return True


def kiem_tra_transformer():
    """Kiểm tra transformer model."""
    print("Đang kiểm tra transformer model...")

    from src.hybrid_fall_transformer import HybridFallTransformer
    import torch

    model = HybridFallTransformer()
    x = torch.randn(2, 60, 60)
    out = model(x)

    assert out.shape == (2, 1), f"Expected (2, 1), got {out.shape}"
    print(f"  Model output shape: {out.shape} OK\n")
    return True


def kiem_tra_augmentation():
    """Kiểm tra augmentation pipeline."""
    print("Đang kiểm tra augmentation...")

    from scripts.augmentation import SequenceAugmenter
    import numpy as np

    aug = SequenceAugmenter(seed=42)
    seq = np.random.rand(60, 60)
    aug_seq = aug.apply(seq)

    assert aug_seq.shape == (60, 60)
    print(f"  Augmented shape: {aug_seq.shape} OK\n")
    return True


def main(strict: bool = False):
    """Chạy tất cả sanity checks."""
    print("=" * 50)
    print("FALL DETECTION SANITY CHECK")
    print("=" * 50 + "\n")

    checks = [
        ("Imports", kiem_tra_imports),
        ("Feature Extraction", kiem_tra_trich_xuat_dac_trung),
        ("Transformer", kiem_tra_transformer),
        ("Augmentation", kiem_tra_augmentation),
    ]

    results = []
    for name, fn in checks:
        try:
            ok = fn()
            results.append((name, ok))
        except Exception as e:
            print(f"  [LOI] {e}\n")
            results.append((name, False))

    print("=" * 50)
    print("KET_QUA")
    print("=" * 50)

    all_pass = True
    for name, ok in results:
        status = "DAT" if ok else "LOI"
        print(f"  {name}: {status}")
        if not ok:
            all_pass = False

    print("=" * 50)

    if all_pass:
        print("Tat ca kiem tra da qua!")
        return 0
    else:
        print("Mot so kiem tra THAT_BAI!")
        if strict:
            return 1
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    sys.exit(main(strict=args.strict))
