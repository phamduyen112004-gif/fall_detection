#!/usr/bin/env python3
"""Export HybridFallTransformer sang ONNX (CPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer
from src.pifr_features import FEATURE_DIM, SEQ_LEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("best_hybrid_transformer.pth"))
    ap.add_argument("--out", type=Path, default=Path("hybrid_fall.onnx"))
    args = ap.parse_args()

    m = HybridFallTransformer()
    try:
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(args.weights, map_location="cpu")
    state = ck.get("model_state_dict", ck)
    m.load_state_dict(state, strict=True)
    m.eval()
    dummy = torch.randn(1, SEQ_LEN, FEATURE_DIM)
    torch.onnx.export(
        m,
        dummy,
        str(args.out),
        input_names=["x"],
        output_names=["logits"],
        opset_version=14,
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
