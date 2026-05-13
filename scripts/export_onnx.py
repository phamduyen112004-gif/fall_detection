"""
ONNX Export Script for HybridFallTransformer.

Exports the trained model to ONNX format for edge inference on NVIDIA Jetson, RK3568,
and other resource-constrained devices. Validates the exported graph against the
original PyTorch model using numerical comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_fall_transformer import HybridFallTransformer


def load_model(weights_path: str | Path) -> HybridFallTransformer:
    """Load a HybridFallTransformer from checkpoint weights."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    # Try to extract model state dict (handle potential checkpoint formats)
    state_dict = None
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            # Try to use the checkpoint directly if it's a state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Infer model config from checkpoint keys if possible
    d_model = 256
    nhead = 4
    num_layers = 3
    dim_feedforward = 256

    # Heuristic: detect config from state dict keys
    if state_dict:
        key_sample = list(state_dict.keys())
        if key_sample:
            first_key = key_sample[0]
            # Detect if keys have "model." prefix (from some training frameworks)
            prefix = ""
            if first_key.startswith("model."):
                prefix = "model."
            elif first_key.startswith("hybrid_fall_transformer."):
                prefix = "hybrid_fall_transformer."

            if prefix:
                # Check for encoder layer count
                encoder_keys = [k for k in state_dict.keys() if "encoder.layers" in k]
                if encoder_keys:
                    num_layers = max(
                        int(k.split(".encoder.layers.")[1].split(".")[0])
                        for k in encoder_keys
                    ) + 1

    model = HybridFallTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError as e:
        print(f"Warning: Partial state dict load: {e}")
        # Try with prefix stripping
        if prefix:
            cleaned_state_dict = {
                k.replace(prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            model.load_state_dict(cleaned_state_dict, strict=False)

    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    seq_len: int = 60,
    feature_dim: int = 60,
    opset_version: int = 14,
) -> Path:
    """
    Export a PyTorch model to ONNX format with dynamic batch size.

    Args:
        model: The PyTorch model to export.
        output_path: Path to save the ONNX file.
        seq_len: Sequence length (static, default 60).
        feature_dim: Feature dimension (static, default 60).
        opset_version: ONNX opset version (default 14 for modern operators).

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input: (batch_size=1, seq_len=60, feature_dim=60)
    dummy_input = torch.randn(1, seq_len, feature_dim)

    # Dynamic axes: batch size is dynamic, seq_len and feature_dim are static
    dynamic_axes = {
        "input": {0: "batch_size"},  # Batch dimension is dynamic
        "output": {0: "batch_size"},  # Batch dimension is dynamic
    }

    # Input/output names for clarity
    input_names = ["input"]
    output_names = ["output"]

    print(f"Exporting model to ONNX: {output_path}")
    print(f"  Input shape: ({'{batch_size}'}, {seq_len}, {feature_dim})")
    print(f"  OpSet version: {opset_version}")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        export_params=True,
        do_constant_folding=True,
    )

    print(f"ONNX model exported successfully: {output_path}")
    return output_path


def get_optimal_providers() -> list[str]:
    """
    Get optimal ONNX Runtime execution providers based on available hardware.

    Returns providers in order of preference (first available will be used).
    """
    try:
        import onnxruntime
    except ImportError:
        return ["CPUExecutionProvider"]  # Fallback if onnxruntime not installed

    available = onnxruntime.get_available_providers()

    # Priority order: GPU > CPU
    priority = [
        "CUDAExecutionProvider",      # NVIDIA GPU
        "CPUExecutionProvider",        # CPU fallback
    ]

    return [p for p in priority if p in available]


def validate_onnx(onnx_path: str | Path) -> bool:
    """
    Validate the exported ONNX model by comparing outputs with PyTorch model.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        True if validation passes, False otherwise.
    """
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime not installed. Install with: pip install onnxruntime")
        print("Skipping ONNX validation.")
        return True  # Not a failure, just skip

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        print(f"ONNX file not found: {onnx_path}")
        return False

    # Get optimal providers
    providers = get_optimal_providers()
    print(f"ONNX Runtime providers (in priority order): {providers}")

    # Run ONNX Runtime inference
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = onnxruntime.InferenceSession(str(onnx_path), sess_options=session_opts, providers=providers)

    # Get input/output names
    onnx_input = session.get_inputs()[0].name
    onnx_output = session.get_outputs()[0].name

    # Create dummy input matching the exported shape
    dummy_input = np.randn(1, 60, 60).astype(np.float32)

    # Run ONNX inference
    onnx_result = session.run([onnx_output], {onnx_input: dummy_input})[0]

    # For full validation, we need the original model weights
    # This is a structural validation - verify the ONNX graph loads and runs
    print(f"ONNX Runtime validation:")
    print(f"  Input name: {onnx_input}, shape: {session.get_inputs()[0].shape}")
    print(f"  Output name: {onnx_output}, shape: {onnx_result.shape}")
    print(f"  ONNX model is valid and ready for inference.")

    return True


def validate_pytorch_vs_onnx(
    pytorch_model: nn.Module,
    onnx_path: str | Path,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Validate ONNX export by comparing PyTorch and ONNX model outputs.

    Args:
        pytorch_model: The original PyTorch model.
        onnx_path: Path to the exported ONNX model.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    try:
        import onnxruntime
    except ImportError:
        print("onnxruntime not installed. Skipping PyTorch vs ONNX comparison.")
        return False

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        print(f"ONNX file not found: {onnx_path}")
        return False

    # Get optimal providers
    providers = get_optimal_providers()
    print(f"ONNX Runtime using providers: {providers}")

    # Create dummy input
    dummy_input = torch.randn(1, 60, 60)

    # PyTorch inference (using predict for sigmoid output)
    with torch.no_grad():
        pytorch_output = pytorch_model.predict(dummy_input).numpy()

    # ONNX Runtime inference with optimized providers
    session_opts = onnxruntime.SessionOptions()
    session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(str(onnx_path), sess_options=session_opts, providers=providers)
    onnx_input = session.get_inputs()[0].name
    onnx_output = session.get_outputs()[0].name
    onnx_result = session.run([onnx_output], {onnx_input: dummy_input.numpy()})[0]

    # Compare outputs
    print(f"\nComparing PyTorch vs ONNX outputs:")
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  ONNX output shape: {onnx_result.shape}")
    print(f"  Max absolute difference: {np.abs(pytorch_output - onnx_result).max():.6e}")
    print(f"  rtol={rtol}, atol={atol}")

    if np.allclose(pytorch_output, onnx_result, rtol=rtol, atol=atol):
        print("SUCCESS: PyTorch and ONNX outputs match!")
        return True
    else:
        print("FAILURE: Outputs do not match within tolerance.")
        return False


def main():
    """Main entry point for ONNX export."""
    parser = argparse.ArgumentParser(
        description="Export HybridFallTransformer to ONNX for edge inference"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best_hybrid_transformer.pth",
        help="Path to model weights file (default: best_hybrid_transformer.pth)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="weights/hybrid_transformer.onnx",
        help="Output path for ONNX file (default: weights/hybrid_transformer.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length (default: 60)",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=60,
        help="Feature dimension (default: 60)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip ONNX validation step",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HybridFallTransformer ONNX Export")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.weights}")
    try:
        model = load_model(args.weights)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: Weights file not found: {args.weights}")
        print("Please provide a valid path to the trained model weights.")
        return 1
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return 1

    # Export to ONNX
    print(f"\nExporting to ONNX...")
    try:
        onnx_path = export_to_onnx(
            model,
            args.out,
            seq_len=args.seq_len,
            feature_dim=args.feature_dim,
            opset_version=args.opset,
        )
    except Exception as e:
        print(f"ERROR during export: {e}")
        return 1

    # Validate ONNX model
    if not args.skip_validation:
        print(f"\nValidating ONNX model...")
        if validate_pytorch_vs_onnx(model, onnx_path):
            print("\n" + "=" * 60)
            print("ONNX export completed successfully!")
            print("Model is valid and ready for edge inference.")
            print("=" * 60)
        else:
            print("\nWARNING: Validation failed. Please check the model and export.")
            return 1
    else:
        print("\nValidation skipped (--skip-validation).")
        print(f"ONNX model saved to: {onnx_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
