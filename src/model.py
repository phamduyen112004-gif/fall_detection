"""
Model exports for the Hybrid Fall Detection Transformer.

This module provides backward-compatible exports for the HybridFallTransformer.
Import from this module or directly from hybrid_transformer.
"""

from .hybrid_transformer import FallDetectionModel, HybridFallTransformer

__all__ = ["HybridFallTransformer", "FallDetectionModel"]
