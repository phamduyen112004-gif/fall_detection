#!/usr/bin/env python3
"""
INT8 Quantization Script for ONNX Runtime.
Converts PyTorch model to ONNX and applies INT8 quantization.
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

def convert_to_onnx(model_path: str, output_path: str, seq_len: int = 60, feature_dim: int = 60):
    """Convert PyTorch model to ONNX format."""
    from src.hybrid_fall_transformer import HybridFallTransformer
    
    model = HybridFallTransformer()
    
    # Load weights if checkpoint provided
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Dummy input
    dummy = torch.randn(1, seq_len, feature_dim)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=13,
    )
    print(f"[OK] ONNX model saved: {output_path}")

def quantize_onnx(onnx_path: str, output_path: str):
    """Apply INT8 quantization to ONNX model."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"[OK] Quantized model saved: {output_path}")
    except ImportError:
        print("[WARN] onnxruntime not installed. Install with: pip install onnxruntime")
        print("[INFO] Skipping quantization, ONNX model still created.")

def main():
    parser = argparse.ArgumentParser(description='INT8 Quantization for Fall Detection')
    parser.add_argument('--model', type=str, default='best_hybrid_transformer.pth',
                        help='Path to PyTorch model (.pth)')
    parser.add_argument('--output', type=str, default='model_int8.onnx',
                        help='Output ONNX path')
    parser.add_argument('--quantize', action='store_true',
                        help='Apply INT8 quantization')
    args = parser.parse_args()
    
    onnx_path = args.output.replace('.onnx', '_float.onnx') if args.quantize else args.output
    
    convert_to_onnx(args.model, onnx_path)
    
    if args.quantize:
        quantize_onnx(onnx_path, args.output)
    
    print("[DONE] Conversion complete!")

if __name__ == "__main__":
    main()
