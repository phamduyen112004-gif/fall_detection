#!/usr/bin/env python3
"""Final comprehensive test of all project imports."""
import sys
sys.path.insert(0, 'e:/fall-detection')

print("=" * 60)
print("FINAL COMPREHENSIVE TEST")
print("=" * 60)

# Test 1: Core pifr_features
print("\n[1] src.pifr_features")
try:
    from src.pifr_features import (
        EPS, IMGSZ, MIN_MEAN_CONF, FEATURE_DIM, SEQ_LEN,
        frame_to_vector_60, resample_to_length,
        GeometricFeatureExtractor
    )
    print("    [OK] All exports present")
    print(f"        EPS={EPS}, IMGSZ={IMGSZ}, FEATURE_DIM={FEATURE_DIM}")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 2: src package
print("\n[2] src package imports")
try:
    from src import (
        EPS, IMGSZ, MIN_MEAN_CONF, FEATURE_DIM, SEQ_LEN,
        frame_to_vector_60, resample_to_length,
        GeometricFeatureExtractor, HybridFallTransformer,
        PipelineConfig, KinematicsAnalyzer, Posture
    )
    print("    [OK] Package imports work")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 3: app_inference.py imports
print("\n[3] app_inference.py imports")
try:
    # Just test the imports at top of file
    import numpy as np
    import cv2
    import torch
    from src.pifr_features import (
        EPS, FEATURE_DIM, IMGSZ, MIN_MEAN_CONF, SEQ_LEN,
        frame_to_vector_60, resample_to_length
    )
    from src.hybrid_fall_transformer import HybridFallTransformer
    print("    [OK] All imports successful")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 4: gui_app.py imports
print("\n[4] gui_app.py imports")
try:
    import numpy as np
    import cv2
    from src.pipeline import HybridFallPipeline
    from src.stage2_pose import PoseExtractor
    from src.stage3_kinematics import FallTemporalFilter
    print("    [OK] All imports successful")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 5: train_transformer.py imports
print("\n[5] train_transformer.py imports")
try:
    import numpy as np
    import torch
    from src.hybrid_fall_transformer import HybridFallTransformer
    from src.pifr_features import SEQ_LEN, FEATURE_DIM
    from scripts.augmentation import SequenceAugmenter
    print("    [OK] All imports successful")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 6: data_extractor.py imports
print("\n[6] data_extractor.py imports")
try:
    import numpy as np
    import cv2
    from src.pifr_features import (
        EPS, FEATURE_DIM, IMGSZ, MIN_MEAN_CONF, SEQ_LEN,
        frame_to_vector_60, resample_to_length
    )
    from src.stage2_pose import PoseExtractor
    print("    [OK] All imports successful")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 7: Scripts
print("\n[7] Scripts imports")
try:
    from scripts.prepare_le2i_dataset import main as le2i_main
    from scripts.le2i_zone_based_extractor import ZoneExtractor
    from scripts.augmentation import SequenceAugmenter
    from scripts.export_onnx import export_to_onnx, validate_onnx
    print("    [OK] All scripts importable")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 8: kaggle_pipeline.py imports
print("\n[8] kaggle_pipeline.py imports")
try:
    import sys as _sys
    # Test basic imports
    import time
    import subprocess
    import numpy as np
    from src.pifr_features import FEATURE_DIM, SEQ_LEN
    from scripts.prepare_le2i_dataset import LE2I_FALL_ANNOTATIONS
    print("    [OK] All imports successful")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 9: kaggle_sanity.py imports
print("\n[9] kaggle_sanity.py imports")
try:
    from src.kaggle_sanity import main as sanity_main
    print("    [OK] kaggle_sanity imports")
except Exception as e:
    print(f"    [FAIL] {e}")

# Test 10: Functional test
print("\n[10] Functional test")
try:
    import numpy as np
    from src.pifr_features import frame_to_vector_60, resample_to_length
    
    # Test frame_to_vector_60
    kp = np.random.rand(17, 3).astype(np.float64)
    vec = frame_to_vector_60(kp)
    assert vec.shape == (60,), f"Expected (60,), got {vec.shape}"
    
    # Test resample_to_length
    seq = np.random.rand(30, 60)
    resampled = resample_to_length(seq, 60)
    assert resampled.shape == (60, 60), f"Expected (60, 60), got {resampled.shape}"
    
    print("    [OK] Functional tests passed")
except Exception as e:
    print(f"    [FAIL] {e}")

print("\n" + "=" * 60)
print("FINAL TEST COMPLETE")
print("=" * 60)
