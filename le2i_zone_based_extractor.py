#!/usr/bin/env python3
"""
LE2I Zone-based Keypoint Extraction & Sliding Window Sequence Generator.

Triển khai Zone-based Protocol (IEEE 2026) cho LE2I Fall Detection Dataset:
  - Trích đặc trưng keypoint YOLOv11-pose từ video LE2I.
  - Sinh chuỗi sliding window 60 frame theo phân vùng nghiêm ngặt:
      Class 0 (Fall): bao trùm hoàn toàn khoảng [start_fall, end_fall].
      Class 1 (Non-Fall/ADL): kết thúc >= 30 frame TRƯỚC start_fall.
      Boundary/Post-Fall: LOẠI BỎ hoàn toàn.

Annotation JSON format (từ prepare_le2i_dataset.py):
  {
    "le2i_scene_fall_nofall_video.avi": {
      "label": 1,        // 1=fall, 0=nofall (ADL)
      "start_fall": 48,  // frame bắt đầu ngã
      "end_fall": 80,    // frame kết thúc ngã
      "slug": "le2i_coffee_room_01"
    }
  }

Output:
  data/le2i_processed/
    X_train.npy, y_train.npy, groups.npy (tương thích AIO pipeline cũ)
  Hoặc: train/fall/*.npy, train/nofall/*.npy (per-class files)

Yêu cầu:
  - AIO_Dataset/ đã chứa LE2I video (chạy trước prepare_le2i_dataset.py).
  - AIO_Dataset/_le2i_annotations.json chứa metadata.
  - Model YOLO: yolo11n-pose.pt

Chạy:
  python le2i_zone_based_extractor.py --aio-dir AIO_Dataset --out-dir data/le2i_processed
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ─── Project root ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.pifr_features import (
    EPS,
    FEATURE_DIM,
    IMGSZ,
    MIN_MEAN_CONF,
    frame_to_vector_60,
    resample_to_length,
)
from src.groups import group_id_from_clip_path

# ─── COCO 17 Keypoint Indices ─────────────────────────────────────────────────
NOSE, L_EYE, R_EYE = 0, 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16
NUM_KEYPOINTS = 17

# ─── Zone-based Protocol Constants (IEEE 2026) ────────────────────────────────
SEQ_LEN = 60          # Fixed sliding window length (frames)
STRIDE = 15            # Sliding window stride (frames)
FALL_SAFETY_MARGIN = 30  # 30-frame buffer before start_fall for ADL class
CONF_THRESHOLD = 0.2   # Min mean keypoint confidence per frame
KEYPOINT_CONF_THRESHOLD = 0.2  # Min confidence per individual keypoint

# ─── Video extensions ─────────────────────────────────────────────────────────
VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"})


# ─── Data Classes ────────────────────────────────────────────────────────────

class VideoAnnotation(TypedDict, total=False):
    """Annotation cho một video LE2I."""
    video_path: str
    label: int          # 1 = fall, 0 = nofall
    source: str
    slug: str           # e.g. "le2i_subject3"
    start_fall: int     # Frame bắt đầu ngã (0-indexed hoặc 1-indexed)
    end_fall: int       # Frame kết thúc ngã
    subject: str
    fps: float


@dataclass
class ExtractedFrame:
    """Một frame đã trích đặc trưng: vector (60,) và metadata."""
    vector: np.ndarray          # shape: (60,)
    frame_index: int            # Frame index gốc trong video
    mean_confidence: float


@dataclass
class SlidingWindowSample:
    """Một mẫu sliding window 60-frame."""
    sequence: np.ndarray        # shape: (SEQ_LEN, 51) - chỉ keypoint (hoặc 60 với geometry)
    label: int                  # 0 = fall, 1 = nofall
    group_id: str               # Subject-level group (cho train/val split)
    source_video: str           # Tên video gốc
    window_start: int           # Frame index bắt đầu trong video gốc
    window_end: int             # Frame index kết thúc trong video gốc


# ─── Keypoint Extraction ──────────────────────────────────────────────────────

def normalize_keypoints(
    kpt_data: np.ndarray,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """
    Min-Max normalize keypoint (x, y) coordinates về [0, 1].
    Confidence được giữ nguyên.
    """
    k_norm = kpt_data.copy()
    k_norm[:, 0] /= float(img_width)
    k_norm[:, 1] /= float(img_height)
    return k_norm


def select_best_person(
    all_keypoints: np.ndarray,
    all_confs: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Chọn người có mean keypoint confidence cao nhất.
    all_keypoints: shape (N_people, 17, 3) - từ YOLO results.
    Returns: shape (17, 3) hoặc None nếu không có ai.
    """
    if all_keypoints.shape[0] == 0:
        return None

    if all_confs is None:
        all_confs = all_keypoints[:, :, 2]

    mean_confs = np.mean(all_confs, axis=1)
    best_idx = int(np.argmax(mean_confs))
    return all_keypoints[best_idx]


def extract_keypoints_from_frame(
    frame: np.ndarray,
    model: YOLO,
    device: str,
) -> np.ndarray | None:
    """
    Trích 17 COCO keypoints từ một frame BGR.

    Returns:
        Vector 51 chiều (17 keypoints × 3 channels: x, y, conf)
        hoặc None nếu không detect được person hoặc confidence quá thấp.
    """
    h, w = frame.shape[:2]
    # Resize về YOLO input
    frame_resized = cv2.resize(frame, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
    results = model.predict(
        frame_resized,
        imgsz=IMGSZ,
        verbose=False,
        device=device,
        conf=0.1,  # Low conf để detect nhiều người hơn, lọc sau
    )

    if not results or results[0].keypoints is None:
        return None

    r0 = results[0]
    kall = r0.keypoints.data.cpu().numpy()
    if kall.size == 0:
        return None

    # Chọn best person
    best_kpt = select_best_person(kall)
    if best_kpt is None:
        return None

    # Tính mean confidence
    mean_conf = float(np.mean(best_kpt[:, 2]))
    if mean_conf < CONF_THRESHOLD:
        return None

    # Denormalize từ IMGSZ về resolution gốc rồi normalize về [0, 1]
    k_denorm = best_kpt.copy()
    k_denorm[:, 0] *= float(w) / float(IMGSZ)
    k_denorm[:, 1] *= float(h) / float(IMGSZ)

    k_norm = normalize_keypoints(k_denorm, w, h)

    # Truncate về 51 chiều (x, y, conf cho 17 keypoints) - không cần geometry 9D ở đây
    # Vì sliding window sẽ dùng trực tiếp keypoints hoặc để pipeline khác thêm geometry
    flat = k_norm.reshape(-1).astype(np.float32)
    return flat


def extract_frames_from_video(
    video_path: Path,
    model: YOLO,
    device: str,
) -> tuple[list[np.ndarray], list[int], list[float]]:
    """
    Trích toàn bộ frame từ video thành list 51-D vectors.

    Returns:
        (vectors, frame_indices, confidences) - các list đồng bộ
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], []

    vectors: list[np.ndarray] = []
    frame_indices: list[int] = []
    confidences: list[float] = []

    prev_vec: np.ndarray | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vec = extract_keypoints_from_frame(frame, model, device)

        if vec is None:
            # Quality filter: impute với frame trước nếu có
            if prev_vec is not None:
                vectors.append(prev_vec.copy())
                confidences.append(0.0)  # Đánh dấu là imputed
            # else: bỏ qua frame đầu tiên nếu không detect được
        else:
            vectors.append(vec)
            mean_conf = float(np.mean(vec.reshape(17, 3)[:, 2]))
            confidences.append(mean_conf)
            prev_vec = vec.copy()

        frame_indices.append(frame_idx)
        frame_idx += 1

    cap.release()
    return vectors, frame_indices, confidences


def extract_frames_batch(
    videos: list[tuple[Path, dict]],
    model: YOLO,
    device: str,
    workers: int = 4,
) -> dict[str, tuple[list[np.ndarray], list[int], list[float]]]:
    """
    Trích frames từ nhiều video sử dụng multiprocessing.

    Args:
        videos: List of (video_path, annotation_dict)
        model: YOLO model
        device: compute device
        workers: số worker processes

    Returns:
        Dict mapping video_name -> (vectors, frame_indices, confidences)
    """
    if workers <= 1 or len(videos) <= 1:
        results = {}
        for vid_path, ann in tqdm(videos, desc="Extracting frames", unit="video"):
            vecs, idxs, confs = extract_frames_from_video(vid_path, model, device)
            if vecs:
                results[vid_path.name] = (vecs, idxs, confs)
        return results

    # Multiprocessing
    with mp.Pool(workers) as pool:
        args_list = [(vp, model, device) for vp, _ in videos]
        results_list = list(tqdm(
            pool.imap(_worker_extract, args_list),
            total=len(videos),
            desc="Extracting frames (parallel)",
            unit="video",
        ))

    results = {}
    for (vid_path, ann), (vecs, idxs, confs) in zip(videos, results_list):
        if vecs:
            results[vid_path.name] = (vecs, idxs, confs)
    return results


def _worker_extract(args: tuple[Path, YOLO, str]) -> tuple[list[np.ndarray], list[int], list[float]]:
    """Worker function cho multiprocessing - phải picklable."""
    video_path, model, device = args
    return extract_frames_from_video(video_path, model, device)


# ─── Zone-based Protocol ──────────────────────────────────────────────────────

def build_zone_boundaries(
    start_fall: int,
    end_fall: int,
    total_frames: int,
) -> dict[str, tuple[int, int]]:
    """
    Xây dựng các vùng phân đoạn theo Zone-based Protocol.

    Layout của một video fall LE2I:
    |---- ADL Zone ----|---- Buffer Zone ----|---- Fall Zone ----|---- Post-Fall Zone ----|
         [0                 start_fall-30         start_fall            end_fall        total]

    Returns dict với các boundary tuples (start, end) inclusive:
      - "adl": Vùng ADL hợp lệ (Class 1)
      - "buffer": Vùng buffer 30 frame (DISCARD)
      - "fall": Vùng Fall bao trùm [start_fall, end_fall] (Class 0)
      - "post_fall": Vùng sau fall (DISCARD)
    """
    adl_end = start_fall - FALL_SAFETY_MARGIN
    fall_start = start_fall
    fall_end = end_fall
    post_fall_start = end_fall + 1

    zones = {}

    # ADL Zone: [0, start_fall - 30]
    if adl_end >= 0:
        zones["adl"] = (0, max(0, adl_end))

    # Buffer Zone: [start_fall - 30, start_fall - 1] → DISCARD
    if start_fall > FALL_SAFETY_MARGIN:
        zones["buffer"] = (start_fall - FALL_SAFETY_MARGIN, start_fall - 1)

    # Fall Zone: [start_fall, end_fall] → Class 0
    if start_fall <= end_fall and start_fall < total_frames:
        zones["fall"] = (start_fall, min(end_fall, total_frames - 1))

    # Post-Fall Zone: [end_fall + 1, total_frames - 1] → DISCARD
    if post_fall_start < total_frames:
        zones["post_fall"] = (post_fall_start, total_frames - 1)

    return zones


def generate_sliding_windows(
    zone: str,
    zone_start: int,
    zone_end: int,
    total_frames: int,
) -> list[tuple[int, int]]:
    """
    Sinh các sliding window trong một vùng cho trước.

    Window phải nằm hoàn toàn trong vùng.
    Class 0 (Fall): window phải bao trùm [start_fall, end_fall]
    Class 1 (ADL): window phải kết thúc trước start_fall - 30

    Args:
        zone: tên vùng ('adl', 'fall', 'buffer', 'post_fall')
        zone_start, zone_end: boundary của vùng (inclusive)
        total_frames: tổng số frame trong video

    Yields:
        List of (window_start, window_end) tuples (inclusive)
    """
    windows: list[tuple[int, int]] = []

    if zone == "fall":
        # Class 0: Sliding window phải bao trùm hoàn toàn [start_fall, end_fall]
        # Có thể extend sang trước và sau để lấy đủ SEQ_LEN frames
        fall_start = zone_start
        fall_end = zone_end

        # Tính toán các vị trí window hợp lệ
        # Window start có thể từ max(0, fall_end - SEQ_LEN + 1) đến fall_start
        min_start = max(0, fall_end - SEQ_LEN + 1)
        max_start = fall_start

        for start in range(min_start, max_start + 1, STRIDE):
            end = start + SEQ_LEN - 1
            # Đảm bảo window nằm trong video
            if end < total_frames:
                windows.append((start, end))

    elif zone == "adl":
        # Class 1: Window phải kết thúc trước start_fall - 30
        adl_end = zone_end
        adl_start = zone_start

        for start in range(adl_start, max(adl_start, adl_end - SEQ_LEN + 2), STRIDE):
            end = start + SEQ_LEN - 1
            if end <= adl_end:
                windows.append((start, end))

    # buffer và post_fall: không sinh window (DISCARD)
    return windows


def create_window_sequences(
    vectors: list[np.ndarray],
    windows: list[tuple[int, int]],
) -> list[np.ndarray]:
    """
    Tạo các chuỗi numpy từ vectors cho các window positions.

    Args:
        vectors: list (T, 51) - đã normalize
        windows: list of (start, end) inclusive indices

    Returns:
        list of (SEQ_LEN, 51) arrays - đã resampled về SEQ_LEN nếu cần
    """
    sequences: list[np.ndarray] = []
    for start, end in windows:
        segment = vectors[start : end + 1]  # inclusive
        if len(segment) == 0:
            continue

        # Resample về SEQ_LEN nếu cần
        segment_arr = np.stack(segment, axis=0)
        if segment_arr.shape[0] != SEQ_LEN:
            segment_arr = resample_to_length(segment_arr, SEQ_LEN)

        sequences.append(segment_arr.astype(np.float32))
    return sequences


# ─── Video-level Processing ───────────────────────────────────────────────────

def process_le2i_video(
    video_path: Path,
    annotation: dict,
    model: YOLO,
    device: str,
) -> tuple[list[SlidingWindowSample], dict[str, Any]]:
    """
    Xử lý một video LE2I:
      1. Trích keypoints từng frame.
      2. Xác định zones theo start_fall/end_fall.
      3. Sinh sliding windows theo Zone-based Protocol.
      4. Tạo samples.

    Args:
        video_path: đường dẫn video
        annotation: metadata từ annotation JSON
        model: YOLO model
        device: compute device

    Returns:
        (list of SlidingWindowSample, stats dict)
    """
    # Extract frames
    vectors, frame_indices, confidences = extract_frames_from_video(video_path, model, device)

    if len(vectors) < SEQ_LEN:
        return [], {"skipped": "video_too_short", "frames": len(vectors)}

    total_frames = len(vectors)
    group_id = annotation.get("slug", group_id_from_clip_path(video_path))

    start_fall = annotation.get("start_fall", -1)
    end_fall = annotation.get("end_fall", -1)

    # Xác định label từ annotation
    base_label = annotation.get("label", 1)

    samples: list[SlidingWindowSample] = []
    stats = {
        "total_frames": total_frames,
        "valid_frames": sum(1 for c in confidences if c > 0),
        "imputed_frames": sum(1 for c in confidences if c == 0),
        "adl_windows": 0,
        "fall_windows": 0,
        "discarded_windows": 0,
    }

    # ── No-Fall Video (ADL only) ──────────────────────────────────────────────
    if base_label == 0 or (start_fall < 0 and end_fall < 0):
        # Toàn bộ video là ADL → tất cả windows là Class 1 (nofall)
        adl_zone_start = 0
        adl_zone_end = total_frames - 1

        windows = generate_sliding_windows("adl", adl_zone_start, adl_zone_end, total_frames)
        sequences = create_window_sequences(vectors, windows)

        for seq, (ws, we) in zip(sequences, windows):
            samples.append(SlidingWindowSample(
                sequence=seq,
                label=1,  # Class 1 = Non-Fall (ADL)
                group_id=group_id,
                source_video=video_path.name,
                window_start=ws,
                window_end=we,
            ))
        stats["adl_windows"] = len(samples)

    # ── Fall Video ────────────────────────────────────────────────────────────
    else:
        # Chuyển đổi 1-indexed → 0-indexed nếu cần
        sf = start_fall - 1 if start_fall > 100 else start_fall  # Heuristic: >100 = likely 1-indexed
        ef = end_fall - 1 if end_fall > 100 else end_fall

        # Validate
        if sf < 0 or ef < 0 or sf >= total_frames or ef >= total_frames or sf > ef:
            # Annotation không hợp lệ → xử lý như ADL
            windows = generate_sliding_windows("adl", 0, total_frames - 1, total_frames)
            sequences = create_window_sequences(vectors, windows)
            for seq, (ws, we) in zip(sequences, windows):
                samples.append(SlidingWindowSample(
                    sequence=seq,
                    label=1,
                    group_id=group_id,
                    source_video=video_path.name,
                    window_start=ws,
                    window_end=we,
                ))
            stats["adl_windows"] = len(samples)
            stats["annotation_invalid"] = True
        else:
            zones = build_zone_boundaries(sf, ef, total_frames)

            # ── ADL Zone (Class 1) ────────────────────────────────────────────
            if "adl" in zones:
                adl_start, adl_end = zones["adl"]
                adl_windows = generate_sliding_windows("adl", adl_start, adl_end, total_frames)
                adl_seqs = create_window_sequences(vectors, adl_windows)
                for seq, (ws, we) in zip(adl_seqs, adl_windows):
                    samples.append(SlidingWindowSample(
                        sequence=seq,
                        label=1,  # Class 1 = Non-Fall (ADL)
                        group_id=group_id,
                        source_video=video_path.name,
                        window_start=ws,
                        window_end=we,
                    ))
                stats["adl_windows"] = len(adl_seqs)

            # ── Fall Zone (Class 0) ──────────────────────────────────────────
            if "fall" in zones:
                fall_start, fall_end = zones["fall"]
                fall_windows = generate_sliding_windows("fall", fall_start, fall_end, total_frames)
                fall_seqs = create_window_sequences(vectors, fall_windows)
                for seq, (ws, we) in zip(fall_seqs, fall_windows):
                    samples.append(SlidingWindowSample(
                        sequence=seq,
                        label=0,  # Class 0 = Fall
                        group_id=group_id,
                        source_video=video_path.name,
                        window_start=ws,
                        window_end=we,
                    ))
                stats["fall_windows"] = len(fall_seqs)

            # ── Buffer & Post-Fall: DISCARD ──────────────────────────────────
            discarded = len(zones) - (1 if "adl" in zones else 0) - (1 if "fall" in zones else 0)
            stats["discarded_windows"] = discarded

    return samples, stats


# ─── Batch Processing & Output ──────────────────────────────────────────────

def collect_le2i_sources(
    aio_dir: Path,
    annotation_json: Path | None = None,
) -> list[tuple[Path, dict]]:
    """
    Thu thập tất cả video LE2I từ AIO_Dataset và merge với annotation.

    Returns:
        List of (video_path, annotation_dict)
    """
    sources: list[tuple[Path, dict]] = []

    # Load annotation JSON nếu có
    annotations: dict[str, dict] = {}
    if annotation_json and annotation_json.is_file():
        with annotation_json.open(encoding="utf-8") as f:
            annotations = json.load(f)

    # Scan fall/ và nofall/
    for label, subdir_name in [(0, "fall"), (1, "nofall")]:
        root = aio_dir / subdir_name
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                key = p.name.lower()
                ann = annotations.get(key, {})
                ann.setdefault("label", label)
                ann.setdefault("slug", group_id_from_clip_path(p))
                sources.append((p, ann))
            elif p.is_dir():
                # Image folder - skip cho zone-based extractor
                pass

    return sources


def train_val_split_by_subject(
    samples: list[SlidingWindowSample],
    val_subject_fraction: float = 0.2,
) -> tuple[list[SlidingWindowSample], list[SlidingWindowSample]]:
    """
    Chia train/val theo subject để tránh leakage.

    Args:
        samples: tất cả samples
        val_subject_fraction: tỷ lệ subject cho validation

    Returns:
        (train_samples, val_samples)
    """
    from collections import defaultdict
    by_subject: dict[str, list[SlidingWindowSample]] = defaultdict(list)
    for s in samples:
        by_subject[s.group_id].append(s)

    subjects = sorted(by_subject.keys())
    n_val = max(1, int(len(subjects) * val_subject_fraction))
    val_subjects = set(subjects[:n_val])

    train_samples = [s for s in samples if s.group_id not in val_subjects]
    val_samples = [s for s in samples if s.group_id in val_subjects]

    return train_samples, val_samples


def save_samples_to_directory(
    samples: list[SlidingWindowSample],
    output_dir: Path,
    split_name: str,
    save_format: str = "npy",
) -> None:
    """
    Lưu samples vào thư mục theo cấu trúc:
      output_dir/train/fall/
      output_dir/train/nofall/
      output_dir/val/fall/
      output_dir/val/nofall/

    Hoặc nếu save_format="merged": lưu thành X_train.npy, y_train.npy, groups.npy
    """
    output_dir = Path(output_dir)

    if save_format == "merged":
        output_dir.mkdir(parents=True, exist_ok=True)
        X_list = [s.sequence for s in samples]
        y_list = [s.label for s in samples]
        g_list = [s.group_id for s in samples]

        if not X_list:
            return

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1)
        g = np.array(g_list, dtype=object)

        np.save(output_dir / f"X_{split_name}.npy", X)
        np.save(output_dir / f"y_{split_name}.npy", y)
        np.save(output_dir / f"groups_{split_name}.npy", g, allow_pickle=True)
        print(f"  Saved {split_name}: X={X.shape}, y={y.shape}, groups={g.shape}")
        return

    # Save as individual .npy files organized by class
    for label, class_name in [(0, "fall"), (1, "nofall")]:
        class_dir = output_dir / split_name / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        class_samples = [s for s in samples if s.label == label]
        for i, s in enumerate(class_samples):
            fname = f"{s.source_video.stem}_win{i:04d}.npy"
            np.save(class_dir / fname, s.sequence)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LE2I Zone-based Keypoint Extraction & Sliding Window Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Zone-based Protocol (IEEE 2026):
  Class 1 (Non-Fall/ADL): Sequence kết thúc >= 30 frames TRƯỚC start_fall
  Class 0 (Fall):         Sequence bao trùm hoàn toàn [start_fall, end_fall]
  Discarded:              Buffer zone, post-fall zone, ambiguous overlaps

Ví dụ:
  # Chạy với annotation JSON từ prepare_le2i_dataset.py
  python le2i_zone_based_extractor.py --aio-dir AIO_Dataset \\
      --annotation-json AIO_Dataset/_le2i_annotations.json \\
      --out-dir data/le2i_processed --val-subjects 5

  # Chạy không có annotation (tất cả video xem như ADL/no-fall)
  python le2i_zone_based_extractor.py --aio-dir AIO_Dataset \\
      --out-dir data/le2i_processed

  # Dùng GPU và nhiều workers
  python le2i_zone_based_extractor.py --aio-dir AIO_Dataset \\
      --out-dir data/le2i_processed --device cuda --workers 8
        """,
    )
    ap.add_argument(
        "--aio-dir",
        type=Path,
        required=True,
        help="Thư mục AIO_Dataset đã chứa LE2I video",
    )
    ap.add_argument(
        "--annotation-json",
        type=Path,
        default=None,
        help="JSON annotation từ prepare_le2i_dataset.py (_le2i_annotations.json)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Thư mục output cho processed data",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolo11n-pose.pt",
        help="YOLO pose model (mặc định: yolo11n-pose.pt)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Thiết bị: "cpu", "0", "cuda:0", "auto" (mặc định)',
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Số worker processes cho multiprocessing (mặc định: 4)",
    )
    ap.add_argument(
        "--val-subjects",
        type=float,
        default=0.2,
        help="Tỷ lệ subject cho validation (mặc định: 0.2 = 20%%)",
    )
    ap.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help=f"Sliding window stride (mặc định: {STRIDE})",
    )
    ap.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help=f"Sliding window length (mặc định: {SEQ_LEN})",
    )
    ap.add_argument(
        "--save-format",
        type=str,
        default="npy",
        choices=["npy", "merged", "both"],
        help='Format lưu: "npy" (thư mục), "merged" (X/y.npy), "both" (cả hai)',
    )

    global STRIDE, SEQ_LEN
    args = ap.parse_args()
    STRIDE = args.stride
    SEQ_LEN = args.seq_len

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not args.aio_dir.is_dir():
        raise SystemExit(f"Không tìm thấy AIO_Dataset: {args.aio_dir}")

    # ── Load YOLO model ────────────────────────────────────────────────────────
    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    device = args.device.strip().lower()
    if device == "auto":
        device = "cuda:0" if _check_cuda_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ── Collect sources ────────────────────────────────────────────────────────
    print(f"[INFO] Scanning {args.aio_dir} for LE2I videos...")
    sources = collect_le2i_sources(args.aio_dir, args.annotation_json)
    if not sources:
        raise SystemExit(
            f"Không tìm thấy video LE2I nào trong {args.aio_dir}. "
            "Chạy trước: python prepare_le2i_dataset.py --le2i-root <path> --out AIO_Dataset"
        )
    print(f"[INFO] Found {len(sources)} videos")

    # ── Process all videos ─────────────────────────────────────────────────────
    all_samples: list[SlidingWindowSample] = []
    all_stats: list[dict] = []

    print(f"[INFO] Processing videos with Zone-based Protocol (T={SEQ_LEN}, stride={STRIDE})...")
    for video_path, annotation in tqdm(sources, desc="Videos", unit="clip"):
        samples, stats = process_le2i_video(video_path, annotation, model, device)
        if samples:
            all_samples.extend(samples)
            all_stats.append({**stats, "video": video_path.name})

    if not all_samples:
        raise SystemExit("Không tạo được sample nào. Kiểm tra annotation và video.")

    # ── Statistics ────────────────────────────────────────────────────────────
    fall_samples = [s for s in all_samples if s.label == 0]
    nofall_samples = [s for s in all_samples if s.label == 1]
    print(f"\n[STATS] Total samples: {len(all_samples)}")
    print(f"  - Fall (Class 0):     {len(fall_samples)}")
    print(f"  - Non-Fall (Class 1): {len(nofall_samples)}")

    unique_subjects = len(set(s.group_id for s in all_samples))
    print(f"  - Unique subjects:    {unique_subjects}")

    # ── Train/Val Split ────────────────────────────────────────────────────────
    print(f"[INFO] Splitting train/val (val_subjects={args.val_subjects:.0%})...")
    train_samples, val_samples = train_val_split_by_subject(all_samples, args.val_subjects)
    print(f"  - Train: {len(train_samples)} samples")
    print(f"  - Val:   {len(val_samples)} samples")

    # ── Save outputs ───────────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_format in ("npy", "both"):
        save_samples_to_directory(train_samples, args.out_dir, "train", "npy")
        save_samples_to_directory(val_samples, args.out_dir, "val", "npy")

    if args.save_format in ("merged", "both"):
        save_samples_to_directory(train_samples, args.out_dir, "train", "merged")
        save_samples_to_directory(val_samples, args.out_dir, "val", "merged")

    # ── Save stats ─────────────────────────────────────────────────────────────
    stats_path = args.out_dir / "_extraction_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(all_samples),
            "fall_samples": len(fall_samples),
            "nofall_samples": len(nofall_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "unique_subjects": unique_subjects,
            "seq_len": SEQ_LEN,
            "stride": STRIDE,
            "fall_safety_margin": FALL_SAFETY_MARGIN,
            "per_video_stats": all_stats,
        }, f, indent=2)
    print(f"[INFO] Stats saved to {stats_path}")

    print(f"\n[SUCCESS] LE2I Zone-based extraction complete!")
    print(f"  Output: {args.out_dir}")
    print(f"  Total: {len(all_samples)} windows | {unique_subjects} subjects")


def _check_cuda_available() -> bool:
    """Kiểm tra CUDA có sẵn không."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
