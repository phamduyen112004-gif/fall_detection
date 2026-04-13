"""
Sanity check cho Kaggle outputs:
  python -m src.kaggle_sanity --strict

Kiểm tra:
  - X_train.npy, y_train.npy, groups.npy có tồn tại và shape hợp lệ
  - checkpoint best_hybrid_transformer.pth có các key quan trọng
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description="Sanity check outputs")
    ap.add_argument("--work-root", type=Path, default=None)
    ap.add_argument("--processed", type=Path, default=None)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    work_root = args.work_root or Path(os.environ.get("FALL_WORK_ROOT", "/kaggle/working"))
    processed = args.processed or (work_root / "data" / "processed")
    ckpt = args.ckpt or (work_root / "best_hybrid_transformer.pth")

    x_path = processed / "X_train.npy"
    y_path = processed / "y_train.npy"
    g_path = processed / "groups.npy"

    missing = [p for p in (x_path, y_path, g_path, ckpt) if not p.is_file()]
    if missing:
        msg = "Thiếu file: " + ", ".join(str(p) for p in missing)
        if args.strict:
            raise SystemExit(msg)
        print("[warn]", msg)
        return

    X = np.load(x_path)
    y = np.load(y_path)
    groups = np.load(g_path, allow_pickle=True)

    print("[ok] X:", x_path, "shape=", X.shape, "dtype=", X.dtype)
    print("[ok] y:", y_path, "shape=", y.shape, "dtype=", y.dtype)
    print("[ok] g:", g_path, "shape=", groups.shape, "dtype=", groups.dtype)

    if args.strict:
        if X.ndim != 3 or X.shape[1:] != (60, 60):
            raise SystemExit(f"X shape không đúng, cần (N,60,60), nhận {X.shape}")
        if y.shape[0] != X.shape[0]:
            raise SystemExit("y và X lệch số mẫu")
        if len(groups) != X.shape[0]:
            raise SystemExit("groups và X lệch số mẫu")

    try:
        import torch

        device = "cpu"
        try:
            ck = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(ckpt, map_location=device)
        if isinstance(ck, dict):
            print("[ok] ckpt keys:", sorted(list(ck.keys()))[:20], "...")
            print("[ok] best_threshold:", ck.get("best_threshold", "(missing)"))
            print("[ok] best_val_f1_tuned:", ck.get("best_val_f1_tuned", "(missing)"))
        else:
            print("[warn] ckpt không phải dict (state_dict thuần).")
            if args.strict:
                raise SystemExit("Checkpoint không đúng định dạng dict.")
    except Exception as e:
        if args.strict:
            raise
        print("[warn] Không đọc được checkpoint:", e)


if __name__ == "__main__":
    main()

