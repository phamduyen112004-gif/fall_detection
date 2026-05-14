#!/usr/bin/env python3
# ============================================================
# URFD + GMDSA24 Fall Detection — Kaggle Full Training Pipeline
# HybridFallTransformer với Sliding Window (T=60, stride=15)
# ============================================================
# Cách chạy trên Kaggle:
#   1. Upload file này lên Kaggle (hoặc chạy trong notebook)
#   2. Đảm bảo dataset có cấu trúc:
#      /kaggle/input/<dataset-name>/URFD/...
#      /kaggle/input/<dataset-name>/GMDSA24/...
#   3. Sửa FALL_DATASET_ROOT phù hợp với dataset của bạn
#   4. Chạy: python kaggle_train.py
# ============================================================

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# CẤU HÌNH — SỬA CÁC GIÁ TRỊ DƯỚI ĐÂY
# ============================================================

# Đường dẫn đến thư mục chứa dataset trên Kaggle Input
FALL_DATASET_ROOT = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"  # <-- THAY ĐỔI NẾU CẦN

# Các thư mục con trong dataset
URFD_ROOT = os.path.join(FALL_DATASET_ROOT, "URFD")
GMDSA_ROOT = os.path.join(FALL_DATASET_ROOT, "GMDSA24")

# Thư mục làm việc
WORK_ROOT = "/kaggle/working"
REPO_NAME = "fall-detection"
WORK_DIR = os.path.join(WORK_ROOT, REPO_NAME)
AIO_ROOT = os.path.join(WORK_ROOT, "AIO_Dataset")
PROCESSED = os.path.join(WORK_ROOT, "data", "processed")
OUT_CKPT = os.path.join(WORK_ROOT, "best_hybrid_transformer.pth")
OUT_ONNX = os.path.join(WORK_ROOT, "hybrid_transformer.onnx")

# Training config
TRAIN_EPOCHS = 100
TRAIN_DEVICE = "cuda"  # "cuda" hoặc "cpu"
STRIDE = 15             # Sliding window stride
SEQ_LEN = 60            # Sliding window length

# ============================================================
# Helper Functions
# ============================================================

def log(msg: str) -> None:
    print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


def run_cmd(cmd: list[str], step: str, strict: bool = True) -> int:
    """Chạy command và in output."""
    cmd_str = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n[STEP] {step}")
    print(f"[CMD]  {cmd_str}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=WORK_DIR, capture_output=False)
    elapsed = time.perf_counter() - t0
    print(f"[TIME] {elapsed:.1f}s", flush=True)
    if result.returncode != 0 and strict:
        print(f"[ERROR] Step '{step}' failed with code {result.returncode}", flush=True)
        sys.exit(1)
    return result.returncode


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def torch_load(path: str):
    """Load torch checkpoint (cross-compat)."""
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    t_start = time.perf_counter()

    print("=" * 60)
    print("URFD + GMDSA24 FALL DETECTION — KAGGLE TRAINING PIPELINE")
    print("HybridFallTransformer + Sliding Window (T=60, stride=15)")
    print("=" * 60)
    print(f"\n[CFG] FALL_DATASET_ROOT : {FALL_DATASET_ROOT}")
    print(f"[CFG] WORK_DIR         : {WORK_DIR}")
    print(f"[CFG] URFD_ROOT       : {URFD_ROOT}")
    print(f"[CFG] GMDSA24_ROOT    : {GMDSA_ROOT}")
    print(f"[CFG] TRAIN_EPOCHS    : {TRAIN_EPOCHS}")
    print(f"[CFG] TRAIN_DEVICE    : {TRAIN_DEVICE}")
    print(f"[CFG] STRIDE          : {STRIDE}")
    print(f"[CFG] SEQ_LEN         : {SEQ_LEN}", flush=True)

    # Validate inputs
    if not os.path.isdir(FALL_DATASET_ROOT):
        print(f"[ERROR] FALL_DATASET_ROOT not found: {FALL_DATASET_ROOT}")
        print("Vui lòng kiểm tra lại đường dẫn dataset trên Kaggle.")
        sys.exit(1)

    # Ensure output dirs
    for d in [AIO_ROOT, PROCESSED]:
        ensure_dir(d)

    # ============================================================
    # STEP 1: Check Dataset Structure
    # ============================================================
    log("STEP 1: Checking Dataset Structure")

    has_urfd = os.path.isdir(URFD_ROOT)
    has_gmdsa = os.path.isdir(GMDSA_ROOT)

    print(f"  URFD:     {'OK' if has_urfd else 'MISSING'} {URFD_ROOT}")
    print(f"  GMDSA24:  {'OK' if has_gmdsa else 'MISSING'} {GMDSA_ROOT}")

    if not has_urfd and not has_gmdsa:
        print("[ERROR] Khong tim thay dataset nao!")
        sys.exit(1)

    # ============================================================
    # STEP 2: Prepare AIO_Dataset (URFD + GMDSA24)
    # ============================================================
    if has_urfd or has_gmdsa:
        log("STEP 2: Prepare AIO_Dataset (URFD + GMDSA24)")

        cmd_prepare = [
            sys.executable, "prepare_dataset.py",
            "--out", AIO_ROOT,
        ]
        if has_urfd:
            cmd_prepare.extend(["--urfd-root", URFD_ROOT])
        if has_gmdsa:
            cmd_prepare.extend(["--gmdcsa-root", GMDSA_ROOT])

        run_cmd(cmd_prepare, "prepare_dataset.py")

        # Count clips
        fall_dir = os.path.join(AIO_ROOT, "fall")
        nofall_dir = os.path.join(AIO_ROOT, "nofall")
        fall_clips = len(os.listdir(fall_dir)) if os.path.isdir(fall_dir) else 0
        nofall_clips = len(os.listdir(nofall_dir)) if os.path.isdir(nofall_dir) else 0
        print(f"  [OK] AIO_Dataset: {fall_clips} fall + {nofall_clips} nofall clips")
    else:
        print("[SKIP] Khong co dataset de chuan bi...")

    # ============================================================
    # STEP 3: Extract PIFR Features (Sliding Window T=60, stride=15)
    # ============================================================
    if has_urfd or has_gmdsa:
        log("STEP 3: Extract PIFR Features (T=60, stride=15)")

        cmd_extract = [
            sys.executable, "data_extractor.py",
            "--aio-dir", AIO_ROOT,
            "--out-dir", PROCESSED,
            "--model", "yolo11n-pose.pt",
            "--device", "cpu",  # Kaggle P100: dung CPU
            "--stride", str(STRIDE),
            "--seq-len", str(SEQ_LEN),
        ]
        run_cmd(cmd_extract, "data_extractor.py")

        # Validate output
        x_path = os.path.join(PROCESSED, "X_train.npy")
        y_path = os.path.join(PROCESSED, "y_train.npy")
        g_path = os.path.join(PROCESSED, "groups.npy")

        if os.path.isfile(x_path):
            import numpy as np
            X = np.load(x_path)
            y = np.load(y_path).reshape(-1)
            g = np.load(g_path, allow_pickle=True) if os.path.isfile(g_path) else None
            n_fall = int(np.sum(y >= 0.5))
            n_nofall = int(np.sum(y < 0.5))
            uniq_g = len(set(str(x) for x in g.tolist())) if g is not None else "N/A"
            print(f"  [OK] Features: N={len(y)} ({n_fall} fall, {n_nofall} nofall)")
            print(f"       Shape: {X.shape}, Groups: {len(g) if g is not None else 'N/A'} ({uniq_g} unique)")
        else:
            print(f"  [WARN] data_extractor output not found tai {PROCESSED}")
    else:
        print("[SKIP] Khong co dataset de trich xuat...")

    # Determine DATA_DIR
    x_path = os.path.join(PROCESSED, "X_train.npy")
    if not os.path.isfile(x_path):
        print(f"  [ERROR] Khong co data de training!")
        sys.exit(1)
    DATA_DIR = PROCESSED

    # ============================================================
    # STEP 4: Train HybridFallTransformer
    # ============================================================
    log("STEP 4: Train HybridFallTransformer")

    cmd_train = [
        sys.executable, "train_transformer.py",
        "--data-dir", DATA_DIR,
        "--out", OUT_CKPT,
        "--device", TRAIN_DEVICE,
        "--epochs", str(TRAIN_EPOCHS),
    ]
    run_cmd(cmd_train, "train_transformer.py")

    if os.path.isfile(OUT_CKPT):
        ckpt = torch_load(OUT_CKPT)
        print(f"\n  [OK] Model checkpoint: {OUT_CKPT}")
        print(f"       Best val F1:    {ckpt.get('best_val_f1', 'N/A'):.4f}")
        print(f"       Optimal thresh: {ckpt.get('best_threshold', 'N/A'):.4f}")
    else:
        print(f"  [ERROR] Training failed, no checkpoint tai {OUT_CKPT}")
        sys.exit(1)

    # ============================================================
    # STEP 5: Export ONNX (optional)
    # ============================================================
    log("STEP 5: Export ONNX (optional)")

    if os.path.isfile(OUT_CKPT):
        cmd_export = [
            sys.executable, "scripts/export_onnx.py",
            "--weights", OUT_CKPT,
            "--out", OUT_ONNX,
        ]
        ret = run_cmd(cmd_export, "export_onnx.py", strict=False)
        if ret == 0 and os.path.isfile(OUT_ONNX):
            size_mb = os.path.getsize(OUT_ONNX) / (1024 * 1024)
            print(f"  [OK] ONNX: {OUT_ONNX} ({size_mb:.2f} MB)")
        else:
            print(f"  [SKIP] ONNX export failed or script not found")
    else:
        print(f"  [SKIP] No checkpoint to export")

    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.perf_counter() - t_start

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Work directory: {WORK_DIR}")
    print(f"  Model: {OUT_CKPT}")
    if os.path.isfile(OUT_CKPT):
        ckpt = torch_load(OUT_CKPT)
        print(f"     Best val F1: {ckpt.get('best_val_f1', 'N/A'):.4f}")
        print(f"     Threshold:    {ckpt.get('best_threshold', 'N/A'):.4f}")
    print(f"\n  Data: {DATA_DIR}")
    print(f"\n  Total time: {total_time / 60:.1f} minutes")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
