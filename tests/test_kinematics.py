#!/usr/bin/env python3
"""Tests for kinematics analysis (angles, posture classification)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_compute_pose_angles():
    from src.stage3_kinematics import compute_pose_angles
    
    # Standing pose (vertical)
    standing_kp = np.array([
        [0.50, 0.15, 0.95], [0.48, 0.14, 0.90], [0.52, 0.14, 0.92],
        [0.47, 0.13, 0.85], [0.53, 0.13, 0.88], [0.40, 0.30, 0.95],
        [0.60, 0.30, 0.95], [0.35, 0.45, 0.90], [0.65, 0.45, 0.88],
        [0.32, 0.60, 0.85], [0.68, 0.60, 0.82], [0.43, 0.55, 0.95],
        [0.57, 0.55, 0.95], [0.44, 0.75, 0.90], [0.56, 0.75, 0.90],
        [0.43, 0.95, 0.88], [0.57, 0.95, 0.87],
    ], dtype=np.float64)
    
    torso, nose_ankle = compute_pose_angles(standing_kp)
    assert torso is not None
    assert nose_ankle is not None
    assert torso < 30, f"Standing torso should be <30 deg, got {torso}"
    print(f"[PASS] test_compute_pose_angles (torso={torso:.1f} deg)")

def test_classify_posture():
    from src.stage3_kinematics import classify_posture, Posture
    from src.config import PipelineConfig
    
    cfg = PipelineConfig()
    
    # Normal posture
    result = classify_posture(10.0, 15.0, cfg)
    assert result == Posture.NORMAL
    print("[PASS] test_classify_posture (normal)")
    
    # Laydown posture
    result = classify_posture(60.0, 60.0, cfg)
    assert result == Posture.LAYDOWN
    print("[PASS] test_classify_posture (laydown)")

def test_fall_temporal_filter():
    from src.stage3_kinematics import FallTemporalFilter, Posture
    from src.config import PipelineConfig
    
    cfg = PipelineConfig()
    flt = FallTemporalFilter(cfg)
    
    # Normal frames
    for _ in range(5):
        assert flt.update(Posture.NORMAL) == False
    
    # Laydown frames - should not trigger immediately
    for _ in range(10):
        flt.update(Posture.LAYDOWN)
    
    print("[PASS] test_fall_temporal_filter")

def test_temporal_filter_resets():
    from src.stage3_kinematics import FallTemporalFilter, Posture
    from src.config import PipelineConfig
    
    cfg = PipelineConfig(fall_min_frames=100)
    flt = FallTemporalFilter(cfg)
    
    flt.update(Posture.LAYDOWN)
    flt.update(Posture.NORMAL)  # Reset
    
    assert flt.state.laydown_frames == 0
    print("[PASS] test_temporal_filter_resets")

if __name__ == "__main__":
    test_compute_pose_angles()
    test_classify_posture()
    test_fall_temporal_filter()
    test_temporal_filter_resets()
    print("\nAll kinematics tests passed!")
