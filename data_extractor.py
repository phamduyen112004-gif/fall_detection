#!/usr/bin/env python3
"""
Trích đặc trưng từ AIO_Dataset (thư mục ảnh URFD hoặc video GMDCSA) -> X_train.npy, y_train.npy.

  python data_extractor.py --aio-dir AIO_Dataset --out-dir data/processed

Frame mean keypoint conf < 0.2: dùng vector frame trước trong **cùng clip** nếu có; không truyền vector **giữa các clip**.
Lưu `groups.npy` để train/val theo nhóm (subject / clip URFD).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.groups import group_id_from_clip_path
from src.pifr_features import (
    EPS,
    FEATURE_DIM,
    IMGSZ,
    MIN_MEAN_CONF,
    SEQ_LEN,
    frame_to_vector_60,
    resample_to_length,
)

VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def collect_aio_sources(aio_dir: Path) -> list[tuple[Path, int]]:
    """Các mục con trực tiếp trong fall/ (label 1) và nofall/ (label 0)."""
    out: list[tuple[Path, int]] = []
    for label, sub in [(1, "fall"), (0, "nofall")]:
        root = aio_dir / sub
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_dir():
                out.append((p, label))
            elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                out.append((p, label))
    return out


def iter_video_label_pairs(
    fall_dir: Path | None,
    normal_dir: Path | None,
    fall_label: int,
    normal_label: int,
) -> list[tuple[Path, int]]:
    """Tương thích: quét video trong hai thư mục (không dùng cấu trúc AIO)."""
    out: list[tuple[Path, int]] = []
    if fall_dir is not None and fall_dir.is_dir():
        for p in sorted(fall_dir.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                out.append((p, fall_label))
    if normal_dir is not None and normal_dir.is_dir():
        for p in sorted(normal_dir.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                out.append((p, normal_label))
    return out


def load_csv_pairs(csv_path: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    with csv_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        if not reader.fieldnames:
            raise ValueError("CSV không có header")
        lower = {h.lower(): h for h in reader.fieldnames}
        pk = next((lower[c] for c in ("video_path", "path", "file") if c in lower), None)
        lk = next((lower[c] for c in ("label", "y", "class") if c in lower), None)
        if pk is None or lk is None:
            raise ValueError("CSV cần cột video_path/path và label/y")
        for row in reader:
            raw = (row.get(pk) or "").strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = (csv_path.parent / path).resolve()
            rows.append((path, int(float(row[lk].strip()))))
    return rows


def _extract_vec_from_bgr(frame_bgr: np.ndarray, model: YOLO, device: str) -> np.ndarray | None:
    """Một khung BGR -> vector (60,) hoặc None nếu bỏ qua."""
    frame_bgr = cv2.resize(frame_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    h, w = frame_bgr.shape[:2]
    results = model.predict(frame_bgr, imgsz=IMGSZ, verbose=False, device=device)

    if not results or results[0].keypoints is None or results[0].keypoints.data is None:
        return None

    r0 = results[0]
    kall = r0.keypoints.data.cpu().numpy()
    if kall.size == 0:
        return None

    best_i = int(np.argmax([float(k[:, 2].mean()) for k in kall]))
    k = kall[best_i].astype(np.float32)
    if float(k[:, 2].mean()) < MIN_MEAN_CONF:
        return None

    kn = k.copy()
    kn[:, 0] /= float(w)
    kn[:, 1] /= float(h)

    box_xyxy = None
    if r0.boxes is not None and len(r0.boxes) > best_i:
        box_xyxy = r0.boxes.xyxy[best_i].cpu().numpy()
    if box_xyxy is not None:
        x1, y1, x2, y2 = [float(x) for x in box_xyxy.flatten()[:4]]
        bw = max(x2 - x1, EPS)
        bh = max(y2 - y1, EPS)
    else:
        bw, bh = float(w), float(h)

    return frame_to_vector_60(kn, (bw, bh))


def process_video_file(video_path: Path, model: YOLO, device: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    prev_vec: np.ndarray | None = None
    feats: list[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        vec = _extract_vec_from_bgr(frame_bgr, model, device)
        if vec is None:
            if prev_vec is not None:
                feats.append(prev_vec.copy())
        else:
            feats.append(vec)
            prev_vec = vec.copy()

    cap.release()
    if not feats:
        return None
    return np.stack(feats, axis=0)


def process_image_folder(folder: Path, model: YOLO, device: str) -> np.ndarray | None:
    files = sorted(
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    )
    if not files:
        return None

    prev_vec: np.ndarray | None = None
    feats: list[np.ndarray] = []
    for fp in files:
        frame_bgr = cv2.imread(str(fp))
        if frame_bgr is None:
            continue
        vec = _extract_vec_from_bgr(frame_bgr, model, device)
        if vec is None:
            if prev_vec is not None:
                feats.append(prev_vec.copy())
        else:
            feats.append(vec)
            prev_vec = vec.copy()

    if not feats:
        return None
    return np.stack(feats, axis=0)


def process_sample(path: Path, model: YOLO, device: str) -> np.ndarray | None:
    """
    Một clip: thư mục ảnh (URFD) hoặc file video.
    Không dùng prev_vec giữa các clip.
    """
    if path.is_dir():
        return process_image_folder(path, model, device)
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return process_video_file(path, model, device)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="AIO_Dataset / video -> X_train.npy, y_train.npy")
    parser.add_argument(
        "--aio-dir",
        type=Path,
        default=None,
        help="Thư mục gốc chứa fall/ và nofall/ (AIO_Dataset)",
    )
    parser.add_argument("--fall-dir", type=Path, default=None)
    parser.add_argument("--normal-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, default="yolo11n-pose.pt")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Thiết bị cho YOLO: "cpu", "0", "cuda:0", hoặc "auto" (mặc định).',
    )
    parser.add_argument("--fall-label", type=int, default=1)
    parser.add_argument("--normal-label", type=int, default=0)
    args = parser.parse_args()

    if args.csv:
        pairs = load_csv_pairs(args.csv)
    elif args.aio_dir is not None:
        pairs = collect_aio_sources(args.aio_dir)
    else:
        pairs = iter_video_label_pairs(
            args.fall_dir,
            args.normal_dir,
            args.fall_label,
            args.normal_label,
        )

    if not pairs:
        raise SystemExit(
            "Không có mẫu. Dùng --aio-dir AIO_Dataset hoặc --fall-dir/--normal-dir hoặc --csv."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    device = (args.device or "auto").strip().lower()
    if device == "auto":
        # An toàn mặc định: CPU (tránh lỗi CUDA trên Kaggle P100 khi torch build không hỗ trợ sm_60).
        device = "cpu"

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    group_list: list[str] = []

    for path, label in tqdm(pairs, desc="Clips", unit="clip"):
        if not path.exists():
            tqdm.write(f"[skip] không tồn tại: {path}")
            continue
        seq = process_sample(path, model, device)
        if seq is None:
            tqdm.write(f"[skip] không đủ frame hợp lệ: {path}")
            continue
        fixed = resample_to_length(seq, SEQ_LEN)
        if fixed.shape != (SEQ_LEN, FEATURE_DIM):
            raise RuntimeError(
                f"Bad shape {fixed.shape}, expected ({SEQ_LEN}, {FEATURE_DIM})"
            )
        X_list.append(fixed)
        y_list.append(label)
        group_list.append(group_id_from_clip_path(path))

    if not X_list:
        raise SystemExit("No valid samples extracted.")

    X_train = np.stack(X_list, axis=0).astype(np.float32)
    y_train = np.array(y_list, dtype=np.float32).reshape(-1, 1)
    groups = np.array(group_list, dtype=object)

    np.save(args.out_dir / "X_train.npy", X_train)
    np.save(args.out_dir / "y_train.npy", y_train)
    np.save(args.out_dir / "groups.npy", groups, allow_pickle=True)
    print(f"Đã lưu X_train {X_train.shape}, y_train {y_train.shape}, groups {groups.shape} → {args.out_dir}")


if __name__ == "__main__":
    main()
