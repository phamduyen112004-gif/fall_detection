"""
Hybrid Fall Detection Transformer

A Transformer-based model for temporal fall detection using pose keypoints.
Architecture: YOLOv11-Pose + PIFR Features + Transformer Encoder
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

# ============================================================
# MODULE-LEVEL LOGGER
# ============================================================

_logger: logging.Logger = logging.getLogger(__name__)


# ============================================================
# MAGIC NUMBER CONSTANTS — Academic Justification
# ============================================================

# SOTA model hyperparameters as documented in the thesis.
# These values are derived from hyperparameter search and are
# NOT magic numbers — they are intentionally fixed architecture constants.

# D_MODEL: Transformer embedding dimension. 256 chosen as the optimal
# balance between model capacity and inference latency for edge deployment.
D_MODEL: int = 256

# NUM_LAYERS: Number of Transformer encoder layers. 3 layers provide
# sufficient temporal modeling depth for 60-frame sequences without
# overfitting on fall detection datasets (CaucaFall, MCFD).
NUM_LAYERS: int = 3

# NHEAD: Number of parallel attention heads. Must divide D_MODEL evenly.
# 256 / 4 = 64 dimensions per head — standard practice.
NHEAD: int = 4

# NUM_FRAMES: Temporal window length — 60 frames at 30 FPS = 2 seconds
# of video history, sufficient context for fall detection.
NUM_FRAMES: int = 60

# INPUT_DIM: Per-frame PIFR feature dimension (51D COCO + 9D geometric).
INPUT_DIM: int = 60

# MAX_LEN: Maximum sequence length for sinusoidal positional encoding.
# Matched to NUM_FRAMES to encode temporal position of each frame.
MAX_LEN: int = 60

# DROPOUT: Regularization probability. 0.1 is standard for Transformer
# models with small-to-medium datasets (thesis: ~400-600 samples).
DROPOUT: float = 0.1

# DIM_FEEDFORWARD: Hidden dimension in the feed-forward sub-layer of each
# Transformer encoder layer. Set to 4× D_MODEL per Vaswani et al. (2017).
DIM_FEEDFORWARD: int = D_MODEL * 4

# POSITIONAL_ENCODING_BASE: Base for sinusoidal frequency computation.
# 10000 is the historical standard from the original Transformer paper.
POSITIONAL_ENCODING_BASE: float = 10000.0


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for Transformer models.

    Implements the absolute positional encoding scheme from Vaswani et al.
    "Attention Is All You Need" (2017). Sinusoidal encoding is preferred
    over learned embeddings for fall detection because:
      1. It generalizes to sequence lengths longer than those seen in training.
      2. It encodes relative temporal position through the periodic sine/cosine
         functions at different frequency bands.

    The encoding is registered as a buffer (not a parameter) so it is moved
    to the correct device automatically with .to(device).
    """

    def __init__(self, d_model: int, max_len: int = MAX_LEN) -> None:
        """
        Initialize sinusoidal positional encoding tables.

        Args:
            d_model: Embedding dimension — must match the Transformer's d_model.
            max_len: Maximum sequence length to pre-compute encodings for.
        """
        super().__init__()

        # Allocate (max_len, d_model) encoding matrix initialized to zeros.
        # dtype=float32 ensures numerical stability during encoding computations.
        pe: torch.Tensor = torch.zeros(max_len, d_model)

        # position: column vector [0, 1, 2, ..., max_len-1]^T
        # Shape: (max_len, 1) — broadcast-compatible with div_term.
        position: torch.Tensor = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: frequency for each embedding dimension.
        # Even indices use sin, odd indices use cos.
        # Formula: exp(-log(base) * (2i / d_model)) for i = 0, 1, ..., d_model/2-1
        # This produces exponentially decreasing frequencies from 1 to ~10^-6.
        div_term: torch.Tensor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(POSITIONAL_ENCODING_BASE) / d_model)
        )

        # Apply sin to even embedding dimensions (0, 2, 4, ...).
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd embedding dimensions (1, 3, 5, ...).
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (max_len, d_model) → (1, max_len, d_model).
        # This allows broadcasting with input tensors of shape (batch, seq, d_model).
        pe = pe.unsqueeze(0)

        # register_buffer registers pe as a non-trainable state_dict entry.
        # It is device-aware and serializable with the model checkpoint.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of the same shape with positional encoding added.
            Slicing [:, :x.size(1), :] ensures compatibility with
            sequences shorter than max_len (e.g., during partial window
            accumulation at the start of inference).
        """
        return x + self.pe[:, : x.size(1), :]


# Alias for backward compatibility with any legacy code referencing the
# original class name.
SinusoidalPositionalEncoding = PositionalEncoding


# ============================================================
# HYBRID FALL TRANSFORMER
# ============================================================

class HybridFallTransformer(nn.Module):
    """
    Hybrid Transformer for Real-Time Fall Detection.

    Processes a sequence of 60 frames, each with a 60D PIFR feature vector,
    to classify whether a fall event has occurred within the temporal window.

    Architecture pipeline:
        1. Input Projection: Linear(60 → 256) — projects raw PIFR features
           into the Transformer embedding space.
        2. Positional Encoding: Sinusoidal — injects temporal position
           information into each frame's embedding.
        3. Transformer Encoder: 3 layers, 4 heads — models long-range
           temporal dependencies across the 60-frame window.
        4. Mean Pooling: Aggregates the 60 frame-level embeddings into a
           single 256D vector (replaces [CLS] token — no learnable bias).
        5. Classification Head: MLP(256 → 128 → 32 → 1) with GELU activation.

    The Mean Pooling strategy was chosen over [CLS] token for two reasons:
      (a) It requires no architectural modification to the standard
          Transformer encoder (no prepended token).
      (b) For sequences of fixed length (60 frames), mean pooling is
          mathematically equivalent to an attention-weighted average when
          attention scores are uniform — a reasonable prior for fall detection.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        num_frames: int = NUM_FRAMES,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        """
        Initialize the Hybrid Fall Detection Transformer.

        Args:
            input_dim: Input feature dimension per frame (default: 60, PIFR).
            num_frames: Number of frames in the temporal sequence (default: 60).
            d_model: Transformer embedding dimension (default: 256).
            nhead: Number of multi-head attention heads (default: 4).
            num_layers: Number of Transformer encoder layers (default: 3).
            dropout: Dropout probability (default: 0.1).
        """
        super().__init__()

        self.input_dim: int = input_dim
        self.num_frames: int = num_frames
        self.d_model: int = d_model

        # Stage 1: Input projection — 60D → 256D.
        # This linear layer transforms raw PIFR features into the
        # higher-dimensional Transformer latent space.
        self.input_projection: nn.Linear = nn.Linear(input_dim, d_model)

        # Stage 2: Sinusoidal positional encoding.
        # Injects temporal position so the Transformer can distinguish
        # frame 0 from frame 59, which have identical PIFR features.
        self.pos_encoder: PositionalEncoding = PositionalEncoding(d_model, num_frames)

        # Stage 3: Dropout for regularization after positional encoding.
        self.dropout_layer: nn.Dropout = nn.Dropout(p=dropout)

        # Stage 4: Transformer encoder layer configuration.
        # norm_first=True: Apply LayerNorm before multi-head attention
        # and feed-forward sub-layers (Pre-LN variant) — more stable
        # training dynamics and better gradient flow than Post-LN.
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        # Stack NUM_LAYERS encoder layers to form the full Transformer.
        self.transformer_encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Stage 5: Classification MLP.
        # 256 → 128 → 32 → 1 with LayerNorm and GELU between layers.
        # LayerNorm is applied per-feature, standard for tabular-like outputs.
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.LayerNorm(128),
            nn.GELU(),     # Gaussian Error Linear Unit — smoother than ReLU,
                           # prevents dying neurons in narrow layers.
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),   # Single logit output for BCEWithLogitsLoss.
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize model weights using Xavier uniform for Linear layers
        and constant initialization for LayerNorm parameters.

        Xavier uniform ensures variance preservation across layers
        (Glorot condition), which stabilizes early training.
        LayerNorm weight=1, bias=0 is the standard default.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Hybrid Fall Detection Transformer.

        Pipeline per input (batch_size, 60, 60):
          1. Linear projection: (B, 60, 60) → (B, 60, 256)
          2. Add sinusoidal positional encoding: (B, 60, 256)
          3. Apply dropout for regularization
          4. Pass through Transformer encoder: (B, 60, 256)
          5. Mean pooling over temporal dimension: (B, 60, 256) → (B, 256)
             This aggregates all 60 frame-level embeddings into a single
             representation by computing the element-wise mean across dim=1.
          6. MLP classification head: (B, 256) → (B, 1)

        Args:
            x: Input tensor of shape (batch, num_frames, input_dim).
               For inference: (1, 60, 60) from the PIFR sliding window.

        Returns:
            Output logits of shape (batch, 1).
            Positive values (>0) indicate fall detection after sigmoid.
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.dropout_layer(x)
        x = self.transformer_encoder(x)
        # Mean pooling: average across the temporal (frame) dimension.
        # Produces a single embedding vector per sample in the batch.
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


class FallDetectionModel(nn.Module):
    """
    Alias for HybridFallTransformer for backward API compatibility.

    This class wraps HybridFallTransformer as a single-module interface,
    redirecting all forward calls through self.model. Use this when a
    unified nn.Module interface is required by external evaluation tools.
    """

    def __init__(self, **kwargs: int | float) -> None:
        """
        Initialize the FallDetectionModel wrapper.

        Args:
            **kwargs: Forwarded to HybridFallTransformer constructor.
        """
        super().__init__()
        self.model: HybridFallTransformer = HybridFallTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the wrapped model.

        Args:
            x: Input tensor of shape (batch, num_frames, input_dim).

        Returns:
            Output logits of shape (batch, 1).
        """
        return self.model(x)


# ============================================================
# STANDALONE SANITY CHECK
# ============================================================

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    _logger: _logging.Logger = _logging.getLogger(__name__)

    BATCH_SIZE: int = 4
    NUM_FRAMES_VAL: int = 60
    INPUT_DIM_VAL: int = 60

    _logger.info("Initializing HybridFallTransformer with SOTA hyperparameters:")
    _logger.info(
        f"  d_model={D_MODEL}, num_layers={NUM_LAYERS}, "
        f"nhead={NHEAD}, dropout={DROPOUT}"
    )

    model: HybridFallTransformer = HybridFallTransformer(
        input_dim=INPUT_DIM_VAL,
        num_frames=NUM_FRAMES_VAL,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    # Synthetic input: (batch=4, frames=60, features=60)
    x: torch.Tensor = torch.randn(BATCH_SIZE, NUM_FRAMES_VAL, INPUT_DIM_VAL)

    output: torch.Tensor = model(x)

    num_params: int = sum(p.numel() for p in model.parameters())

    _logger.info(f"Input shape:  {x.shape}")
    _logger.info(f"Output shape: {output.shape}")
    _logger.info(f"Output logits: {output.squeeze().tolist()}")
    _logger.info(f"Total parameters: {num_params:,}")
