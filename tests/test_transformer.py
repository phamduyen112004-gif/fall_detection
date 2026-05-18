"""
Unit tests for HybridFallTransformer model.
"""

from __future__ import annotations

import torch
import pytest

from src.hybrid_transformer import HybridFallTransformer, FallDetectionModel


class TestHybridFallTransformer:
    """Test suite for HybridFallTransformer."""

    @pytest.fixture
    def batch_size(self) -> int:
        return 2

    @pytest.fixture
    def num_frames(self) -> int:
        return 60

    @pytest.fixture
    def input_dim(self) -> int:
        return 60

    @pytest.fixture
    def model(self, input_dim: int, num_frames: int) -> HybridFallTransformer:
        """Create a model instance with default SOTA hyperparameters."""
        return HybridFallTransformer(
            input_dim=input_dim,
            num_frames=num_frames,
            d_model=256,
            nhead=4,
            num_layers=3,
            dropout=0.1,
        )

    @pytest.fixture
    def dummy_input(self, batch_size: int, num_frames: int, input_dim: int) -> torch.Tensor:
        """Create mock input tensor of shape (2, 60, 60)."""
        return torch.randn(batch_size, num_frames, input_dim)

    def test_forward_pass_returns_correct_shape(
        self,
        model: HybridFallTransformer,
        dummy_input: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Assert output shape is (batch_size, 1)."""
        output = model(dummy_input)
        assert output.shape == (batch_size, 1)

    def test_forward_pass_accepts_different_batch_sizes(
        self,
        model: HybridFallTransformer,
        num_frames: int,
        input_dim: int,
    ) -> None:
        """Model should handle various batch sizes."""
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, num_frames, input_dim)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_model_with_custom_d_model(
        self,
        batch_size: int,
        num_frames: int,
        input_dim: int,
    ) -> None:
        """Model should work with custom d_model values."""
        for d_model in [128, 256, 512]:
            model = HybridFallTransformer(d_model=d_model)
            x = torch.randn(batch_size, num_frames, input_dim)
            output = model(x)
            assert output.shape == (batch_size, 1)

    def test_model_is_trainable(
        self,
        model: HybridFallTransformer,
        dummy_input: torch.Tensor,
    ) -> None:
        """Model should compute gradients during backprop."""
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_model_has_expected_parameter_count(
        self,
        model: HybridFallTransformer,
    ) -> None:
        """Model should have a reasonable number of parameters."""
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 100_000, "Model should have substantial parameters"
        assert num_params < 10_000_000, "Model should not be excessively large"


class TestFallDetectionModel:
    """Test suite for FallDetectionModel wrapper."""

    def test_forward_pass_returns_correct_shape(self) -> None:
        """FallDetectionModel should wrap HybridFallTransformer correctly."""
        model = FallDetectionModel(
            input_dim=60,
            num_frames=60,
            d_model=256,
            nhead=4,
            num_layers=3,
            dropout=0.1,
        )
        x = torch.randn(2, 60, 60)
        output = model(x)
        assert output.shape == (2, 1)

    def test_backward_pass_computes_gradients(self) -> None:
        """Wrapper should support gradient computation."""
        model = FallDetectionModel()
        x = torch.randn(2, 60, 60)
        output = model(x)
        loss = output.sum()
        loss.backward()
        assert model.model.classifier[0].weight.grad is not None
