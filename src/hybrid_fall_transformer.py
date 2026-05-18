"""
Hybrid Fall Transformer Model
============================
Transformer-based fall detection using YOLOv11n-Pose extracted PIFR features.

DEPRECATED: Import from `src.hybrid_transformer` instead.
This file is kept for backward compatibility only.
"""

from src.hybrid_transformer import (
    HybridFallTransformer,
    FallDetectionModel,
    PositionalEncoding,
)

# Alias for backward compatibility
FallTransformer = HybridFallTransformer

__all__ = [
    "HybridFallTransformer",
    "FallDetectionModel",
    "FallTransformer",
    "PositionalEncoding",
]
