#!/usr/bin/env python3
"""
Quick Test Script - Verify all imports work correctly.
Run this on Kaggle to ensure the project is ready.

Usage:
    python scripts/test_imports.py
"""

import sys

def test_imports():
    """Test all critical imports."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    tests = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("ultralytics", "Ultralytics (YOLO)"),
        ("cv2", "OpenCV"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("tqdm", "tqdm"),
        ("thop", "THOP (FLOP calculator)"),
    ]
    
    passed = 0
    failed = 0
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"✓ {name}")
            passed += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        print("\n⚠️  Some imports failed. Install missing packages:")
        print("    pip install -r requirements.txt")
        return False
    
    return True


def test_project_imports():
    """Test project-specific imports."""
    print("\n" + "=" * 60)
    print("TESTING PROJECT IMPORTS")
    print("=" * 60)
    
    tests = [
        ("src.config", "Config module"),
        ("src.hybrid_transformer", "Transformer model"),
        ("src.pifr_features", "PIFR features"),
        ("src.utils", "Utils module"),
    ]
    
    passed = 0
    failed = 0
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"✓ {name}")
            passed += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        return False
    
    return True


def test_model_creation():
    """Test model can be created."""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    try:
        import torch
        from src.hybrid_transformer import HybridFallTransformer
        
        model = HybridFallTransformer(
            input_dim=60,
            num_frames=60,
            d_model=256,
            nhead=4,
            num_layers=3,
            dropout=0.1
        )
        
        # Test forward pass
        x = torch.randn(2, 60, 60)
        output = model(x)
        
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        
        print(f"✓ Model created successfully")
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Parameters: {num_params:.2f}M")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pifr_features():
    """Test PIFR feature extraction."""
    print("\n" + "=" * 60)
    print("TESTING PIFR FEATURES")
    print("=" * 60)
    
    try:
        import numpy as np
        from src.pifr_features import compute_pifr, extract_keypoints
        
        # Test with dummy keypoints
        dummy_kpts = np.random.rand(17, 3).astype(np.float32)
        dummy_kpts[:, 2] = 0.9  # High confidence
        
        features = compute_pifr(dummy_kpts, 640, 480)
        
        print(f"✓ PIFR features shape: {features.shape}")
        assert features.shape == (60,), f"Expected (60,), got {features.shape}"
        
        return True
        
    except Exception as e:
        print(f"✗ PIFR features failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration."""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from src.config import (
            DATA_DIR, OUTPUT_DIR, MODEL_SAVE_DIR, LOG_DIR, RESULTS_DIR,
            TRAINING_HYPERPARAMS, YOLO_MODEL
        )
        
        print(f"✓ DATA_DIR: {DATA_DIR}")
        print(f"✓ OUTPUT_DIR: {OUTPUT_DIR}")
        print(f"✓ MODEL_SAVE_DIR: {MODEL_SAVE_DIR}")
        print(f"✓ YOLO_MODEL: {YOLO_MODEL}")
        print(f"✓ Training Hyperparams: {TRAINING_HYPERPARAMS}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FALL DETECTION PROJECT - PRE-RUN VERIFICATION")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    # Test basic imports
    if not test_imports():
        all_passed = False
    
    # Test project imports
    if not test_project_imports():
        all_passed = False
    
    # Test model
    if not test_model_creation():
        all_passed = False
    
    # Test PIFR
    if not test_pifr_features():
        all_passed = False
    
    # Test config
    if not test_config():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Project is ready for Kaggle.")
    else:
        print("❌ SOME TESTS FAILED! Please fix errors before running.")
    print("=" * 60)
    
    sys.exit(0 if all_passed else 1)
