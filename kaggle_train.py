#!/usr/bin/env python3
# ============================================================
# LE2I Fall Detection — Kaggle Full Training Pipeline
# Zone-based Protocol (IEEE 2026) + HybridFallTransformer
# ============================================================
# Cách chạy trên Kaggle:
#   1. Upload file này lên Kaggle (hoặc chạy trong notebook)
#   2. Đảm bảo dataset có cấu trúc:
#      /kaggle/input/<dataset-name>/URFD/...
#      /kaggle/input/<dataset-name>/GMDCSA24/...
#      /kaggle/input/<dataset-name>/LE2I/...
#   3. Sửa FALL_DATASET_ROOT phù hợp với dataset của bạn
#   4. Chạy: python kaggle_train.py
# ============================================================

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================================================
# CẤU HÌNH — SỬA CÁC GIÁ TRỊ DƯỚI ĐÂY
# ============================================================

# Đường dẫn đến thư mục chứa dataset trên Kaggle Input
# Dataset: https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia
FALL_DATASET_ROOT = "/kaggle/input/datasets/tuyenldvn/falldataset-imvia"  # <-- THAY ĐỔI NẾU CẦN

# Các thư mục con trong dataset
URFD_ROOT = os.path.join(FALL_DATASET_ROOT, "URFD")
GMDCSA_ROOT = os.path.join(FALL_DATASET_ROOT, "GMDCSA24")
LE2I_ROOT = FALL_DATASET_ROOT  # LE2I dataset ở root của Kaggle dataset

# LE2I Annotation CSV (nếu có — chứa start_fall, end_fall)
# LE2I có thể có Annotation_files folder trong mỗi scene
LE2I_ANNOTATION_CSV = ""

# Thư mục làm việc
WORK_ROOT = "/kaggle/working"
REPO_NAME = "fall-detection"
WORK_DIR = os.path.join(WORK_ROOT, REPO_NAME)
AIO_ROOT = os.path.join(WORK_ROOT, "AIO_Dataset")
PROCESSED = os.path.join(WORK_ROOT, "data", "processed")
LE2I_PROCESSED = os.path.join(WORK_ROOT, "data", "le2i_processed")
MERGED_DIR = os.path.join(WORK_ROOT, "data", "merged")
OUT_CKPT = os.path.join(WORK_ROOT, "best_hybrid_transformer.pth")
OUT_ONNX = os.path.join(WORK_ROOT, "hybrid_transformer.onnx")

# Training config
TRAIN_EPOCHS = 100
TRAIN_DEVICE = "cuda"  # "cuda" hoặc "cpu"
LE2I_VAL_SUBJECTS = 0.2
LE2I_STRIDE = 15

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


# ============================================================
# Main Pipeline
# ============================================================

def main() -> None:
    t_start = time.perf_counter()

    print("=" * 60)
    print("LE2I FALL DETECTION — KAGGLE TRAINING PIPELINE")
    print("Zone-based Protocol (IEEE 2026) + HybridFallTransformer")
    print("=" * 60)
    print(f"\n[CFG] FALL_DATASET_ROOT : {FALL_DATASET_ROOT}")
    print(f"[CFG] WORK_DIR          : {WORK_DIR}")
    print(f"[CFG] URFD_ROOT        : {URFD_ROOT}")
    print(f"[CFG] GMDCSA_ROOT      : {GMDCSA_ROOT}")
    print(f"[CFG] LE2I_ROOT        : {LE2I_ROOT}")
    print(f"[CFG] LE2I_ANNOTATION  : {LE2I_ANNOTATION_CSV}")
    print(f"[CFG] TRAIN_EPOCHS     : {TRAIN_EPOCHS}")
    print(f"[CFG] TRAIN_DEVICE     : {TRAIN_DEVICE}")
    print(f"[CFG] LE2I_VAL_SUBJECTS: {LE2I_VAL_SUBJECTS}")
    print(f"[CFG] LE2I_STRIDE      : {LE2I_STRIDE}", flush=True)

    # Validate inputs
    if not os.path.isdir(FALL_DATASET_ROOT):
        print(f"[ERROR] FALL_DATASET_ROOT not found: {FALL_DATASET_ROOT}")
        print("Vui lòng kiểm tra lại đường dẫn dataset trên Kaggle.")
        sys.exit(1)

    # Ensure output dirs
    for d in [AIO_ROOT, PROCESSED, LE2I_PROCESSED, MERGED_DIR]:
        ensure_dir(d)

    # ============================================================
    # STEP 1: Check Dataset Structure
    # ============================================================
    log("STEP 1: Checking Dataset Structure")

    has_urfd = os.path.isdir(URFD_ROOT)
    has_gmdcsa = os.path.isdir(GMDCSA_ROOT)
    has_le2i = os.path.isdir(LE2I_ROOT)

    print(f"  URFD:        {'✅' if has_urfd else '❌'} {URFD_ROOT}")
    print(f"  GMDCSA24:    {'✅' if has_gmdcsa else '❌'} {GMDCSA_ROOT}")
    print(f"  LE2I:        {'✅' if has_le2i else '❌'} {LE2I_ROOT} (Kaggle format: Annotation_files + Videos)")
    if not has_urfd and not has_gmdcsa and not has_le2i:
        print("[ERROR] Không tìm thấy dataset nào!")
        sys.exit(1)

    # ============================================================
    # STEP 2: Prepare AIO_Dataset (URFD + GMDCSA)
    # ============================================================
    if has_urfd or has_gmdcsa:
        log("STEP 2: Prepare AIO_Dataset (URFD + GMDCSA)")

        cmd_prepare = [
            sys.executable, "prepare_dataset.py",
            "--urfd-root", URFD_ROOT if has_urfd else "__skip__",
            "--gmdcsa-root", GMDCSA_ROOT if has_gmdcsa else "__skip__",
            "--out", AIO_ROOT,
        ]
        # Remove skip placeholder
        cmd_prepare = [c for c in cmd_prepare if c != "__skip__"]
        run_cmd(cmd_prepare, "prepare_dataset.py")

        # Count clips
        fall_clips = len(os.listdir(os.path.join(AIO_ROOT, "fall"))) if os.path.isdir(os.path.join(AIO_ROOT, "fall")) else 0
        nofall_clips = len(os.listdir(os.path.join(AIO_ROOT, "nofall"))) if os.path.isdir(os.path.join(AIO_ROOT, "nofall")) else 0
        print(f"  [OK] AIO_Dataset: {fall_clips} fall + {nofall_clips} nofall clips")
    else:
        print("[SKIP] URFD + GMDCSA not found, skipping...")

    # ============================================================
    # STEP 3: Prepare LE2I Dataset
    # ============================================================
    if has_le2i:
        log("STEP 3: Prepare LE2I Dataset")

        cmd_le2i = [
            sys.executable, "prepare_le2i_dataset.py",
            "--le2i-root", LE2I_ROOT,
            "--out", AIO_ROOT,
        ]
        # LE2I annotations được tự động parse từ Annotation_files folders

        run_cmd(cmd_le2i, "prepare_le2i_dataset.py", strict=False)

        # Check annotation JSON
        # NOTE: prepare_le2i_dataset.py uses label=1 for fall, label=0 for nofall (ADL)
        le2i_json = os.path.join(AIO_ROOT, "_le2i_annotations.json")
        if os.path.isfile(le2i_json):
            with open(le2i_json, encoding="utf-8") as f:
                ann = json.load(f)
            fall_ann = sum(1 for v in ann.values() if v.get("label") == 1)    # label=1 is fall
            nofall_ann = sum(1 for v in ann.values() if v.get("label") == 0)  # label=0 is nofall
            with_fall_info = sum(1 for v in ann.values() if v.get("start_fall", -1) >= 0)
            print(f"  [OK] LE2I: {len(ann)} clips ({fall_ann} fall, {nofall_ann} nofall)")
            print(f"       Với start_fall/end_fall: {with_fall_info}")
        else:
            print(f"  [WARN] _le2i_annotations.json not created")
    else:
        print("[SKIP] LE2I not found, skipping...")

    # ============================================================
    # STEP 4: Extract PIFR Features (URFD + GMDCSA)
    # ============================================================
    if has_urfd or has_gmdcsa:
        log("STEP 4: Extract PIFR Features (URFD + GMDCSA)")

        cmd_extract = [
            sys.executable, "data_extractor.py",
            "--aio-dir", AIO_ROOT,
            "--out-dir", PROCESSED,
            "--model", "yolo11n-pose.pt",
            "--device", "cpu",  # Kaggle P100: dùng CPU
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
            print(f"  [OK] AIO features: N={len(y)} ({n_fall} fall, {n_nofall} nofall)")
            print(f"       Shape: {X.shape}, Groups: {len(g) if g is not None else 'N/A'} ({uniq_g} unique)")
        else:
            print(f"  [WARN] data_extractor output not found at {PROCESSED}")
    else:
        print("[SKIP] URFD + GMDCSA not found, skipping feature extraction...")

    # ============================================================
    # STEP 5: Zone-based LE2I Extraction
    # ============================================================
    if has_le2i:
        log("STEP 5: Zone-based LE2I Extraction")

        cmd_le2i_extract = [
            sys.executable, "le2i_zone_based_extractor.py",
            "--aio-dir", AIO_ROOT,
            "--out-dir", LE2I_PROCESSED,
            "--val-subjects", str(LE2I_VAL_SUBJECTS),
            "--stride", str(LE2I_STRIDE),
            "--device", "cpu",
        ]

        le2i_json = os.path.join(AIO_ROOT, "_le2i_annotations.json")
        if os.path.isfile(le2i_json):
            cmd_le2i_extract.extend(["--annotation-json", le2i_json])

        run_cmd(cmd_le2i_extract, "le2i_zone_based_extractor.py", strict=False)

        # Stats
        stats_path = os.path.join(LE2I_PROCESSED, "_extraction_stats.json")
        if os.path.isfile(stats_path):
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
            print(f"\n  [LE2I Zone-based Stats]")
            print(f"    Total samples:   {stats.get('total_samples', 'N/A')}")
            print(f"    Fall (Class 0): {stats.get('fall_samples', 'N/A')}")
            print(f"    Non-Fall (Class 1): {stats.get('nofall_samples', 'N/A')}")
            print(f"    Train:          {stats.get('train_samples', 'N/A')}")
            print(f"    Val:            {stats.get('val_samples', 'N/A')}")
            print(f"    Unique subjects:{stats.get('unique_subjects', 'N/A')}")
    else:
        print("[SKIP] LE2I not found, skipping zone-based extraction...")

    # ============================================================
    # STEP 6: Merge Datasets
    # ============================================================
    log("STEP 6: Merge Datasets")

    import numpy as np

    def load_data(dir_path: str):
        x_p = os.path.join(dir_path, "X_train.npy")
        y_p = os.path.join(dir_path, "y_train.npy")
        g_p = os.path.join(dir_path, "groups.npy")
        if not os.path.isfile(x_p) or not os.path.isfile(y_p):
            return None, None, None
        X = np.load(x_p)
        y = np.load(y_p)
        g = np.load(g_p, allow_pickle=True) if os.path.isfile(g_p) else np.array(["unknown"] * len(y))
        return X, y, g

    X_aio, y_aio, g_aio = load_data(PROCESSED)
    X_le2i, y_le2i, g_le2i = load_data(LE2I_PROCESSED)

    print(f"  AIO:  X={X_aio.shape if X_aio is not None else 'N/A'}, y={y_aio.shape if y_aio is not None else 'N/A'}")
    print(f"  LE2I: X={X_le2i.shape if X_le2i is not None else 'N/A'}, y={y_le2i.shape if y_le2i is not None else 'N/A'}")

    DATA_DIR = PROCESSED  # Default

    if X_aio is not None and X_le2i is not None:
        X_merged = np.concatenate([X_aio, X_le2i], axis=0)
        y_merged = np.concatenate([y_aio, y_le2i], axis=0)
        g_prefixed = np.concatenate([
            np.array([f"aio_{str(x)}" for x in g_aio]),
            np.array([f"le2i_{str(x)}" for x in g_le2i]),
        ])
        np.save(os.path.join(MERGED_DIR, "X_train.npy"), X_merged)
        np.save(os.path.join(MERGED_DIR, "y_train.npy"), y_merged)
        np.save(os.path.join(MERGED_DIR, "groups.npy"), g_prefixed, allow_pickle=True)
        DATA_DIR = MERGED_DIR
        print(f"  [OK] Merged: X={X_merged.shape}, y={y_merged.shape}, groups={len(g_prefixed)}")
        print(f"       Saved to: {MERGED_DIR}")
    elif X_aio is not None:
        print(f"  [INFO] Using AIO data only (LE2I not available)")
        DATA_DIR = PROCESSED
    else:
        print(f"  [ERROR] No data available for training!")
        sys.exit(1)

    # ============================================================
    # STEP 7: Train HybridFallTransformer
    # ============================================================
    log("STEP 7: Train HybridFallTransformer")

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
        print(f"  [ERROR] Training failed, no checkpoint at {OUT_CKPT}")
        sys.exit(1)

    # ============================================================
    # STEP 8: Export ONNX
    # ============================================================
    log("STEP 8: Export ONNX (optional)")

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
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  📁 Work directory: {WORK_DIR}")
    print(f"  🧠 Model: {OUT_CKPT}")
    if os.path.isfile(OUT_CKPT):
        ckpt = torch_load(OUT_CKPT)
        print(f"     Best val F1: {ckpt.get('best_val_f1', 'N/A'):.4f}")
        print(f"     Threshold:    {ckpt.get('best_threshold', 'N/A'):.4f}")
    print(f"\n  📊 Data:")
    print(f"     AIO (URFD+GMDCSA): {PROCESSED}")
    print(f"     LE2I (Zone-based): {LE2I_PROCESSED}")
    print(f"     Merged:            {DATA_DIR}")
    print(f"\n  ⏱️  Total time: {total_time / 60:.1f} minutes")
    print("\n" + "=" * 60)


def torch_load(path: str):
    """Load torch checkpoint (cross-compat)."""
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}


if __name__ == "__main__":
    main()
