"""Unit tests for HybridFallTransformer model."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestHybridFallTransformer:
    """Test suite for HybridFallTransformer model."""

    @pytest.fixture
    def model(self):
        """Create a fresh model instance."""
        from src.hybrid_fall_transformer import HybridFallTransformer
        return HybridFallTransformer()

    @pytest.fixture
    def sample_input(self):
        """Create a sample input tensor (batch_size=2, seq_len=60, feature_dim=60)."""
        return torch.randn(2, 60, 60)

    @pytest.fixture
    def sample_sequence(self):
        """Create a sample sequence as numpy array."""
        return np.random.rand(60, 60).astype(np.float32)

    def test_model_forward_shape(self, model, sample_input):
        """Model forward pass should output correct shape."""
        output = model(sample_input)
        assert output.shape == (2, 1), f"Expected shape (2, 1), got {output.shape}"

    def test_model_output_is_logits(self, model, sample_input):
        """Model output should be raw logits (no sigmoid applied)."""
        output = model(sample_input)
        # Logits can be any real number, not constrained to [0, 1]
        assert isinstance(output, torch.Tensor)
        assert output.numel() == 2  # batch size

    def test_model_eval_mode(self, model):
        """Model should work in eval mode."""
        model.eval()
        with torch.no_grad():
            x = torch.randn(4, 60, 60)
            output = model(x)
            assert output.shape == (4, 1)

    def test_model_train_mode(self, model):
        """Model should work in train mode."""
        model.train()
        x = torch.randn(4, 60, 60)
        output = model(x)
        assert output.shape == (4, 1)

    def test_model_single_sample(self, model):
        """Model should accept single sample input."""
        x = torch.randn(1, 60, 60)
        output = model(x)
        assert output.shape == (1, 1)

    def test_model_predict_method(self, model, sample_sequence):
        """predict() method should work with numpy input."""
        prediction = model.predict(sample_sequence)
        assert isinstance(prediction, (float, np.floating))

    def test_model_predict_batch(self, model):
        """predict() method should work with batch numpy input."""
        batch = np.random.rand(8, 60, 60).astype(np.float32)
        predictions = model.predict(batch)
        assert len(predictions) == 8

    def test_model_predict_probs(self, model, sample_sequence):
        """predict() with return_probs=True should return probability."""
        prob = model.predict(sample_sequence, return_probs=True)
        assert 0.0 <= prob <= 1.0, f"Probability should be in [0, 1], got {prob}"

    def test_model_deterministic(self, model):
        """Model should produce same output with same input (eval mode)."""
        model.eval()
        x = torch.randn(4, 60, 60)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
            torch.testing.assert_close(out1, out2)

    def test_model_large_batch(self, model):
        """Model should handle large batches."""
        large_batch = torch.randn(128, 60, 60)
        output = model(large_batch)
        assert output.shape == (128, 1)

    def test_model_gradient_flow(self, model):
        """Model should support backpropagation."""
        model.train()
        x = torch.randn(4, 60, 60, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()

        # Check that gradients exist for model parameters
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad or x.grad is not None, "No gradients computed"

    def test_model_no_nan_output(self, model):
        """Model output should not contain NaN."""
        model.eval()
        x = torch.randn(16, 60, 60)
        with torch.no_grad():
            output = model(x)
            assert not torch.isnan(output).any(), "Model output contains NaN"
            assert not torch.isinf(output).any(), "Model output contains Inf"

    def test_model_zero_input(self, model):
        """Model should handle zero input."""
        model.eval()
        x = torch.zeros(2, 60, 60)
        with torch.no_grad():
            output = model(x)
            assert output.shape == (2, 1)
            assert not torch.isnan(output).any()

    def test_model_large_input_values(self, model):
        """Model should handle large input values."""
        model.eval()
        x = torch.randn(2, 60, 60) * 10  # Large values
        with torch.no_grad():
            output = model(x)
            assert output.shape == (2, 1)
            assert not torch.isnan(output).any()


class TestSinusoidalPositionalEncoding:
    """Test suite for SinusoidalPositionalEncoding."""

    def test_pe_shape(self):
        """PE output should have correct shape."""
        from src.hybrid_fall_transformer import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(4, 100, 256)
        out = pe(x)
        assert out.shape == x.shape

    def test_pe_preserves_shape(self):
        """PE should not change sequence length or batch size."""
        from src.hybrid_fall_transformer import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=128, max_len=50)
        batch_size, seq_len = 3, 50
        x = torch.randn(batch_size, seq_len, 128)
        out = pe(x)
        assert out.shape == (batch_size, seq_len, 128)

    def test_pe_deterministic(self):
        """PE should produce same output for same input."""
        from src.hybrid_fall_transformer import SinusoidalPositionalEncoding

        pe = SinusoidalPositionalEncoding(d_model=64, max_len=30)
        x = torch.randn(2, 30, 64)

        with torch.no_grad():
            out1 = pe(x)
            out2 = pe(x)
            torch.testing.assert_close(out1, out2)


class TestModelLoading:
    """Test model weight loading."""

    def test_load_from_checkpoint(self):
        """Model should load from checkpoint dict."""
        from src.hybrid_fall_transformer import HybridFallTransformer

        model = HybridFallTransformer()
        checkpoint = model.state_dict()

        model2 = HybridFallTransformer()
        model2.load_state_dict(checkpoint)

        # Models should produce identical outputs
        model.eval()
        model2.eval()
        x = torch.randn(2, 60, 60)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
            torch.testing.assert_close(out1, out2)
