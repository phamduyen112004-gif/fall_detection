"""INT8 Quantization Script cho ONNX Runtime.
Chuyển đổi PyTorch model sang ONNX và áp dụng INT8 quantization."""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

# Thêm comment giải thích thêm về quá trình
"""
Quá trình quantization:
1. Chuyển đổi PyTorch -> ONNX (FP32)
2. Áp dụng Dynamic Quantization -> INT8
3. Kết quả: model nhỏ hơn, inference nhanh hơn
"""


def convert_to_onnx(model_path: str, output_path: str, seq_len: int = 60, feature_dim: int = 60):
    """
    Chuyển đổi PyTorch model sang ONNX format.

    Args:
        model_path: Đường dẫn đến PyTorch model (.pth).
        output_path: Đường dẫn lưu ONNX model.
        seq_len: Độ dài sequence (mặc định: 60).
        feature_dim: Số chiều đặc trưng (mặc định: 60).
    """
    from src.hybrid_fall_transformer import HybridFallTransformer

    model = HybridFallTransformer()

    # Load weights nếu có checkpoint
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    # Dummy input cho ONNX export
    dummy = torch.randn(1, seq_len, feature_dim)

    # Export sang ONNX
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=13,
    )
    print(f"[OK] ONNX model đã lưu: {output_path}")


def quantize_onnx(onnx_path: str, output_path: str):
    """
    Áp dụng INT8 quantization lên ONNX model.

    Args:
        onnx_path: Đường dẫn ONNX model gốc (FP32).
        output_path: Đường dẫn lưu model đã quantized (INT8).
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"[OK] Quantized model đã lưu: {output_path}")
    except ImportError:
        print("[WARN] onnxruntime chưa được cài đặt. Cài với: pip install onnxruntime")
        print("[INFO] Bỏ qua quantization, ONNX model vẫn được tạo.")


def main():
    """Entry point chính."""
    parser = argparse.ArgumentParser(description='INT8 Quantization cho Fall Detection')
    parser.add_argument('--model', type=str, default='best_hybrid_transformer.pth',
                        help='Đường dẫn PyTorch model (.pth)')
    parser.add_argument('--output', type=str, default='model_int8.onnx',
                        help='Đường dẫn lưu ONNX')
    parser.add_argument('--quantize', action='store_true',
                        help='Áp dụng INT8 quantization')
    args = parser.parse_args()

    onnx_path = args.output.replace('.onnx', '_float.onnx') if args.quantize else args.output

    convert_to_onnx(args.model, onnx_path)

    if args.quantize:
        quantize_onnx(onnx_path, args.output)

    print("[HOAN_TAT] Chuyển đổi hoàn tất!")


if __name__ == "__main__":
    main()
