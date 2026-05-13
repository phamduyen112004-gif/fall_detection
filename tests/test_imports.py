#!/usr/bin/env python3
"""Quick test for module-level exports."""
import sys
sys.path.insert(0, 'e:/fall-detection')

try:
    from src.pifr_features import (
        EPS, IMGSZ, MIN_MEAN_CONF, FEATURE_DIM, SEQ_LEN,
        frame_to_vector_60, resample_to_length
    )
    print("[OK] pifr_features imports: OK")
    print(f"     EPS={EPS}, IMGSZ={IMGSZ}, MIN_MEAN_CONF={MIN_MEAN_CONF}")
    print(f"     FEATURE_DIM={FEATURE_DIM}, SEQ_LEN={SEQ_LEN}")
except Exception as e:
    print(f"[FAIL] pifr_features: {e}")

try:
    from src import (
        EPS, IMGSZ, MIN_MEAN_CONF, FEATURE_DIM, SEQ_LEN,
        frame_to_vector_60, resample_to_length
    )
    print("[OK] src package imports: OK")
except Exception as e:
    print(f"[FAIL] src package: {e}")

try:
    import numpy as np
    from src.pifr_features import resample_to_length
    arr = np.random.rand(30, 60)
    result = resample_to_length(arr, 60)
    print(f"[OK] resample_to_length: {arr.shape} -> {result.shape}")
except Exception as e:
    print(f"[FAIL] resample_to_length: {e}")

try:
    from scripts.prepare_le2i_dataset import ZoneExtractor
    print("[OK] prepare_le2i_dataset imports: OK")
except Exception as e:
    print(f"[FAIL] prepare_le2i_dataset: {e}")

try:
    from scripts.le2i_zone_based_extractor import ZoneExtractor
    print("[OK] le2i_zone_based_extractor imports: OK")
except Exception as e:
    print(f"[FAIL] le2i_zone_based_extractor: {e}")

print("\nAll critical imports tested!")
