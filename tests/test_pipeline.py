#!/usr/bin/env python3
"""Integration tests for the complete pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_pipeline_config():
    from src.config import PipelineConfig, DEFAULT_CONFIG
    cfg = DEFAULT_CONFIG
    assert cfg.input_size == (640, 640)
    assert cfg.min_mean_keypoint_conf == 0.2
    print("[PASS] test_pipeline_config")

def test_viz_edges():
    from src.viz import COCO_EDGES
    assert len(COCO_EDGES) > 0
    print(f"[PASS] test_viz_edges ({len(COCO_EDGES)} edges)")

def test_groups():
    from src.groups import assign_groups
    paths = [f"clip_{i}" for i in range(10)]
    groups = assign_groups(paths)
    assert len(groups) == len(paths)
    print(f"[PASS] test_groups ({len(set(groups))} unique groups)")

def test_full_sequence_processing():
    """Test end-to-end: random keypoints → 60D features → sequence."""
    from src.pifr_features import GeometricFeatureExtractor
    
    ext = GeometricFeatureExtractor()
    
    # Generate 60 frames
    seq = []
    for _ in range(60):
        kp = np.random.rand(17, 3).astype(np.float64)
        feat = ext.extract(kp)
        seq.append(feat)
    
    seq = np.array(seq)
    assert seq.shape == (60, 60), f"Expected (60, 60), got {seq.shape}"
    print(f"[PASS] test_full_sequence_processing shape={seq.shape}")

if __name__ == "__main__":
    test_pipeline_config()
    test_viz_edges()
    test_groups()
    test_full_sequence_processing()
    print("\nAll integration tests passed!")
