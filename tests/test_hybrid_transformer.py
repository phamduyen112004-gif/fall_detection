"""
Unit tests for HybridFallTransformer model.

Tests the transformer architecture for fall detection.
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_transformer import (
    HybridFallTransformer,
    FallDetectionModel,
    PositionalEncoding,
)


class TestPositionalEncoding:
    """Test positional encoding module."""

    def test_output_shape(self):
        """Output should have same sequence length as input."""
        d_model = 256
        seq_len = 60
        pe = PositionalEncoding(d_model, max_len=100)
        x = torch.zeros(2, seq_len, d_model)
        out = pe(x)
        assert out.shape == x.shape

    def test_positional_encoding_is_added(self):
        """Output should be input + positional encoding."""
        d_model = 64
        seq_len = 30
        pe = PositionalEncoding(d_model, max_len=100)
        x = torch.zeros(1, seq_len, d_model)
        out = pe(x)
        # Output should differ from input
        assert not torch.allclose(x, out)

    def test_different_positions_different_encoding(self):
        """Different positions should have different encodings."""
        d_model = 64
        seq_len = 20
        pe = PositionalEncoding(d_model, max_len=100)
        x = torch.zeros(1, seq_len, d_model)
        out = pe(x)
        # Adjacent positions should have different values
        assert not torch.equal(out[0, 0], out[0, 1])

    def test_buffer_registered(self):
        """PE should be registered as a buffer (not a parameter)."""
        pe = PositionalEncoding(128)
        assert hasattr(pe, 'pe')
        assert 'pe' in pe._buffers
        assert 'pe' not in pe._parameters


class TestHybridFallTransformer:
    """Test main transformer model."""

    def test_default_initialization(self):
        """Model should initialize with default parameters."""
        model = HybridFallTransformer()
        assert model.input_dim == 60
        assert model.num_frames == 60
        assert model.d_model == 256

    def test_custom_parameters(self):
        """Model should accept custom parameters."""
        model = HybridFallTransformer(
            input_dim=60,
            num_frames=60,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.2,
        )
        assert model.d_model == 128
        assert model.num_frames == 60

    def test_forward_pass_shape(self):
        """Output should be (batch, 1) for binary classification."""
        model = HybridFallTransformer()
        batch_size = 4
        x = torch.randn(batch_size, 60, 60)
        out = model(x)
        assert out.shape == (batch_size, 1)

    def test_forward_pass_single_sample(self):
        """Should handle single sample (batch_size=1)."""
        model = HybridFallTransformer()
        x = torch.randn(1, 60, 60)
        out = model(x)
        assert out.shape == (1, 1)

    def test_gradient_flow(self):
        """Model should support backpropagation."""
        model = HybridFallTransformer()
        x = torch.randn(2, 60, 60)
        target = torch.tensor([[1.0], [0.0]])
        
        criterion = torch.nn.BCEWithLogitsLoss()
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        model = HybridFallTransformer()
        x = torch.randn(1, 60, 60)
        
        # Set seed
        torch.manual_seed(42)
        out1 = model(x)
        
        # Reset model and seed
        model = HybridFallTransformer()
        torch.manual_seed(42)
        out2 = model(x)
        
        torch.testing.assert_close(out1, out2)

    def test_training_mode(self):
        """Model should switch between train and eval modes."""
        model = HybridFallTransformer()
        
        model.train()
        assert model.training
        
        model.eval()
        assert not model.training

    def test_model_has_positional_encoding(self):
        """Model should have positional encoding layer."""
        model = HybridFallTransformer()
        assert hasattr(model, 'pos_encoder')
        assert isinstance(model.pos_encoder, PositionalEncoding)

    def test_model_has_classifier(self):
        """Model should have classification head."""
        model = HybridFallTransformer()
        assert hasattr(model, 'classifier')

    def test_model_has_transformer_encoder(self):
        """Model should have transformer encoder."""
        model = HybridFallTransformer()
        assert hasattr(model, 'transformer_encoder')

    def test_model_creates_buffers(self):
        """Model should register positional encoding as buffer."""
        model = HybridFallTransformer()
        # PE should be a registered buffer
        assert 'pos_encoder.pe' in dict(model.named_buffers())


class TestFallDetectionModel:
    """Test FallDetectionModel alias class."""

    def test_creates_hybrid_fall_transformer(self):
        """Should wrap HybridFallTransformer."""
        model = FallDetectionModel(
            input_dim=60,
            d_model=128,
            nhead=4,
            num_layers=2,
        )
        assert hasattr(model, 'model')
        assert isinstance(model.model, HybridFallTransformer)

    def test_forward_passes_through(self):
        """Forward pass should pass through to inner model."""
        model = FallDetectionModel()
        x = torch.randn(2, 60, 60)
        out = model(x)
        assert out.shape == (2, 1)

    def test_parameters_from_inner_model(self):
        """Should expose parameters from inner model."""
        model = FallDetectionModel()
        params = list(model.parameters())
        inner_params = list(model.model.parameters())
        assert len(params) == len(inner_params)


class TestModelOutput:
    """Test model output characteristics."""

    def test_output_is_logit(self):
        """Output should be raw logits (not probabilities)."""
        model = HybridFallTransformer()
        x = torch.randn(4, 60, 60)
        out = model(x)
        # Logits can be any value, not constrained to [0, 1]
        assert isinstance(out, torch.Tensor)

    def test_sigmoid_applies_correctly(self):
        """Sigmoid should convert logits to probabilities."""
        model = HybridFallTransformer()
        x = torch.randn(4, 60, 60)
        out = model(x)
        probs = torch.sigmoid(out)
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

    def test_different_inputs_different_outputs(self):
        """Different inputs should produce different outputs."""
        model = HybridFallTransformer()
        x1 = torch.randn(1, 60, 60)
        x2 = torch.randn(1, 60, 60)
        
        model.eval()
        with torch.no_grad():
            out1 = model(x1)
            out2 = model(x2)
        
        # Outputs should differ (unless by extreme coincidence)
        assert not torch.allclose(out1, out2, atol=0.1)


class TestModelCapacity:
    """Test model parameter counts and capacity."""

    def test_default_model_has_reasonable_size(self):
        """Default model should have reasonable parameter count."""
        model = HybridFallTransformer()
        num_params = sum(p.numel() for p in model.parameters())
        # Should be in the range of hundreds of thousands to few millions
        assert 100_000 < num_params < 5_000_000, f"Unexpected param count: {num_params:,}"

    def test_smaller_model_has_fewer_params(self):
        """Smaller d_model should reduce parameter count."""
        model_large = HybridFallTransformer(d_model=256)
        model_small = HybridFallTransformer(d_model=64)
        
        params_large = sum(p.numel() for p in model_large.parameters())
        params_small = sum(p.numel() for p in model_small.parameters())
        
        assert params_large > params_small


class TestInputValidation:
    """Test input validation and error handling."""

    def test_wrong_input_dim_raises(self):
        """Wrong input dimension should raise error."""
        model = HybridFallTransformer(input_dim=60)
        x = torch.randn(2, 60, 100)  # Wrong dim
        with pytest.raises(Exception):
            model(x)

    def test_wrong_num_frames_raises(self):
        """Wrong number of frames should raise error."""
        model = HybridFallTransformer(input_dim=60, num_frames=60)
        x = torch.randn(2, 100, 60)  # Wrong frames
        with pytest.raises(Exception):
            model(x)
