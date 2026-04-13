"""
Kaggle one-command pipeline:
  python -m src.kaggle_pipeline --strict

Mặc định đọc dataset Kaggle Input có cấu trúc:
  <DATASET_ROOT>/URFD/(Fall|fall, ADL|adl)/*.zip
  <DATASET_ROOT>/GMDCSA24/Subject */(Fall|fall, ADL|adl)/*.mp4

Biến môi trường:
  - FALL_DATASET_ROOT: ví dụ /kaggle/input/fall-detection-dataset
  - FALL_WORK_ROOT:    ví dụ /kaggle/working
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _env_path(key: str, default: str) -> Path:
    return Path(os.environ.get(key, default)).expanduser()


def _run(cmd: list[str], strict: bool) -> None:
    cmd = [c for c in cmd if c]
    print("[run]", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if strict:
            raise SystemExit(e.returncode)
        print("[warn] command failed:", e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Kaggle pipeline: prepare -> extract -> train")
    ap.add_argument("--dataset-root", type=Path, default=None)
    ap.add_argument("--work-root", type=Path, default=None)
    ap.add_argument("--strict", action="store_true", help="Fail fast nếu thiếu dữ liệu/đầu ra.")
    ap.add_argument("--pose-weights", type=str, default="yolo11n-pose.pt")
    args = ap.parse_args()

    dataset_root = args.dataset_root or _env_path("FALL_DATASET_ROOT", "/kaggle/input/fall-detection-dataset")
    work_root = args.work_root or _env_path("FALL_WORK_ROOT", "/kaggle/working")

    urfd_root = dataset_root / "URFD"
    gmdcsa_root = dataset_root / "GMDCSA24"

    aio_root = work_root / "AIO_Dataset"
    processed = work_root / "data" / "processed"
    out_ckpt = work_root / "best_hybrid_transformer.pth"

    print("[cfg] dataset_root =", dataset_root)
    print("[cfg] work_root    =", work_root)
    print("[cfg] URFD_ROOT    =", urfd_root)
    print("[cfg] GMDCSA_ROOT  =", gmdcsa_root)

    if args.strict and (not urfd_root.is_dir()) and (not gmdcsa_root.is_dir()):
        raise SystemExit(f"Không thấy URFD hoặc GMDCSA24 dưới {dataset_root}.")

    # Validate structure early (để lỗi rõ ràng, khỏi chạy lâu)
    if urfd_root.is_dir():
        has_urfd_fall = any((urfd_root / d).is_dir() for d in ("Fall", "fall", "FALL"))
        has_urfd_adl = any((urfd_root / d).is_dir() for d in ("ADL", "adl", "Adl"))
        if args.strict and (not has_urfd_fall) and (not has_urfd_adl):
            raise SystemExit(
                f"URFD có nhưng thiếu thư mục Fall/fall hoặc ADL/adl dưới {urfd_root}."
            )
        if (not has_urfd_fall) and (not has_urfd_adl):
            print(f"[warn] URFD có nhưng thiếu Fall/ADL dưới {urfd_root}.")

    if gmdcsa_root.is_dir():
        has_subject = any(p.is_dir() and "subject" in p.name.lower() for p in gmdcsa_root.iterdir())
        if args.strict and (not has_subject):
            raise SystemExit(f"GMDCSA24 có nhưng không thấy thư mục Subject* dưới {gmdcsa_root}.")
        if not has_subject:
            print(f"[warn] GMDCSA24 có nhưng không thấy Subject* dưới {gmdcsa_root}.")

    t0 = time.perf_counter()

    # 1) Prepare AIO_Dataset
    t_prepare_0 = time.perf_counter()
    _run(
        [
            sys.executable,
            "prepare_dataset.py",
            "--urfd-root",
            str(urfd_root),
            "--gmdcsa-root",
            str(gmdcsa_root),
            "--out",
            str(aio_root),
            "--strict" if args.strict else "",
        ],
        strict=args.strict,
    )
    t_prepare_1 = time.perf_counter()
    print(f"[time] prepare_dataset: {t_prepare_1 - t_prepare_0:.1f}s")

    # 2) Extract features (force CPU by default to avoid Kaggle P100 sm_60 incompatibility)
    t_extract_0 = time.perf_counter()
    _run(
        [
            sys.executable,
            "data_extractor.py",
            "--aio-dir",
            str(aio_root),
            "--out-dir",
            str(processed),
            "--model",
            args.pose_weights,
            "--device",
            "cpu",
        ],
        strict=args.strict,
    )
    t_extract_1 = time.perf_counter()
    print(f"[time] data_extractor : {t_extract_1 - t_extract_0:.1f}s")

    # Print quick dataset stats
    x_path = processed / "X_train.npy"
    y_path = processed / "y_train.npy"
    g_path = processed / "groups.npy"
    if x_path.is_file() and y_path.is_file():
        X = np.load(x_path)
        y = np.load(y_path).reshape(-1)
        n = int(X.shape[0])
        n_fall = int(np.sum(y >= 0.5))
        n_nofall = n - n_fall
        print(f"[data] N={n} fall={n_fall} nofall={n_nofall} X_shape={tuple(X.shape)}")
        if g_path.is_file():
            groups = np.load(g_path, allow_pickle=True)
            try:
                uniq = int(len(set(str(x) for x in groups.tolist())))
                print(f"[data] groups: {len(groups)} (unique={uniq})")
            except Exception:
                print(f"[data] groups: {len(groups)} (unique=n/a)")
    elif args.strict:
        raise SystemExit(f"Thiếu {x_path} hoặc {y_path} sau khi trích đặc trưng.")

    # 3) Train transformer
    t_train_0 = time.perf_counter()
    _run(
        [
            sys.executable,
            "train_transformer.py",
            "--data-dir",
            str(processed),
            "--out",
            str(out_ckpt),
        ],
        strict=args.strict,
    )
    t_train_1 = time.perf_counter()
    print(f"[time] train_transform : {t_train_1 - t_train_0:.1f}s")

    print("[done] ckpt =", out_ckpt)
    print(f"[time] total          : {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

