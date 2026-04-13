"""HybridFallTransformer: chuỗi (B, 60, 60) → logit nhị phân."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """PE theo Vaswani et al.; buffer cố định (1, max_len, d_model)."""

    def __init__(self, d_model: int, max_len: int = 60) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, d_model)
        return self.pe[:, : x.size(1), :]


class HybridFallTransformer(nn.Module):
    """
    Đầu vào (B, 60, 60): 60 frame, 60 đặc trưng/frame.
    Chiếu Linear(60→256), nhân sqrt(256), cộng sinusoidal PE,
    TransformerEncoder ×3, mean pool, MLP → (B, 1).
    """

    def __init__(
        self,
        in_features: int = 60,
        seq_len: int = 60,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.input_proj = nn.Linear(in_features, d_model)
        self.scale = math.sqrt(float(d_model))
        self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 60, 60)
        h = self.input_proj(x) * self.scale
        h = h + self.pos_encoder(h)
        h = self.encoder(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)
