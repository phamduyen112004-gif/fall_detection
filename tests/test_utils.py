"""
Unit Tests for Utils Module
===========================
Tests for utility functions.
"""

import numpy as np
import pytest
import torch

from src.utils import (
    get_device,
    check_gpu,
    calculate_metrics,
    standardize_temporal_dim,
    safe_arccos,
)


class TestGetDevice:
    """Test device selection."""

    def test_returns_cuda_when_available(self):
        if torch.cuda.is_available():
            assert get_device().type == "cuda"
        else:
            assert get_device().type == "cpu"

    def test_returns_cpu_when_cuda_unavailable(self):
        device = get_device()
        assert device.type in ("cuda", "cpu")


class TestCheckGPU:
    """Test GPU check function."""

    def test_returns_dict(self):
        result = check_gpu()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = check_gpu()
        assert "cuda_available" in result
        assert "device" in result
        assert "gpu_name" in result
        assert "gpu_memory_gb" in result

    def test_cuda_available_matches_torch(self):
        result = check_gpu()
        assert result["cuda_available"] == torch.cuda.is_available()

    def test_device_matches_cuda_available(self):
        result = check_gpu()
        expected = "cuda" if result["cuda_available"] else "cpu"
        assert result["device"] == expected

    def test_gpu_memory_gb_is_float(self):
        result = check_gpu()
        assert isinstance(result["gpu_memory_gb"], float)

    def test_gpu_memory_gb_is_positive_when_available(self):
        if torch.cuda.is_available():
            result = check_gpu()
            assert result["gpu_memory_gb"] > 0

    def test_no_gpu_returns_zero_memory(self):
        if not torch.cuda.is_available():
            result = check_gpu()
            assert result["gpu_memory_gb"] == 0.0


class TestCalculateMetrics:
    """Test calculate_metrics function."""

    def test_perfect_predictions(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]

        metrics = calculate_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 0]

        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0.0

    def test_partial_predictions(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        metrics = calculate_metrics(y_true, y_pred)

        assert 0 < metrics["accuracy"] < 1
        assert 0 < metrics["f1"] <= 1

    def test_confusion_matrix_shape(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]

        metrics = calculate_metrics(y_true, y_pred)

        # Should be 2x2 for binary classification
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_empty_predictions_handled(self):
        """Empty predictions should return zero-division=0."""
        y_true = []
        y_pred = []

        # Should not raise, returns zeros
        metrics = calculate_metrics(y_true, y_pred)

        assert metrics["accuracy"] == 0
        assert metrics["f1"] == 0


class TestStandardizeTemporalDim:
    """Test standardize_temporal_dim function."""

    def test_exact_60_frames(self, sample_video_features):
        """Exactly 60 frames should be returned as-is."""
        result = standardize_temporal_dim(sample_video_features, target_frames=60)
        assert result.shape == (60, 60)

    def test_longer_than_max(self):
        """Frames > 120 should be truncated then subsampled."""
        features = np.random.rand(150, 60).astype(np.float32)
        result = standardize_temporal_dim(features, max_frames=120)
        # 150 -> 120 -> 60 (after subsampling)
        assert result.shape[0] == 60
        assert result.shape[1] == 60

    def test_shorter_than_60(self):
        """Frames < 60 should be padded with last frame."""
        features = np.random.rand(30, 60).astype(np.float32)
        result = standardize_temporal_dim(features, target_frames=60)
        assert result.shape == (60, 60)

    def test_subsample_by_2(self):
        """Should subsample every 2nd frame."""
        features = np.random.rand(100, 60).astype(np.float32)
        result = standardize_temporal_dim(features, target_frames=60)
        assert result.shape[0] == 60

    def test_empty_input_returns_zeros(self):
        """Empty input should return zeros."""
        result = standardize_temporal_dim(None)
        assert result.shape == (60, 60)
        assert np.allclose(result, 0.0)

    def test_empty_list_returns_zeros(self):
        """Empty list should return zeros."""
        result = standardize_temporal_dim([])
        assert result.shape == (60, 60)
        assert np.allclose(result, 0.0)

    def test_dtype_preserved(self):
        """Output dtype should be float32."""
        features = np.random.rand(30, 60).astype(np.float64)
        result = standardize_temporal_dim(features)
        assert result.dtype == np.float32

    def test_custom_target_frames(self):
        """Should support custom target frame count."""
        features = np.random.rand(20, 60).astype(np.float32)
        result = standardize_temporal_dim(features, target_frames=30)
        assert result.shape == (30, 60)


class TestSafeArccos:
    """Test safe_arccos function."""

    def test_valid_input(self):
        assert 0 <= safe_arccos(0) <= np.pi
        assert 0 <= safe_arccos(1) <= np.pi
        assert 0 <= safe_arccos(-1) <= np.pi

    def test_out_of_range_clipped(self):
        """Values outside [-1, 1] should be clipped."""
        assert safe_arccos(1.5) == safe_arccos(1.0)
        assert safe_arccos(-1.5) == safe_arccos(-1.0)

    def test_nan_input(self):
        """NaN should return 0."""
        assert safe_arccos(float('nan')) == 0.0

    def test_inf_input(self):
        """Inf should return 0."""
        assert safe_arccos(float('inf')) == 0.0
        assert safe_arccos(float('-inf')) == 0.0

    def test_array_input(self):
        """Should work with numpy arrays."""
        arr = np.array([-1, -0.5, 0, 0.5, 1])
        result = safe_arccos(arr)
        assert len(result) == 5
        assert np.all(result >= 0)
        assert np.all(result <= np.pi)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
