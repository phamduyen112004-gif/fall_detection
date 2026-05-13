#!/usr/bin/env python3
"""
Benchmark FPS for Fall Detection on Multiple Videos.

Measures per-stage latency and overall FPS across multiple videos
to compare with SOTA research (e.g., Liu et al. 2022: ~30 FPS,
Xu et al. 2024: ~25 FPS, Han et al. 2023: ~20 FPS).

Usage:
    python benchmark_fps.py --video-dir /path/to/videos --output results.csv
    python benchmark_fps.py --video-list video1.mp4 video2.mp4 ... --output results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer
from src.pifr_features import (
    EPS,
    FEATURE_DIM,
    IMGSZ,
    MIN_MEAN_CONF,
    SEQ_LEN,
    frame_to_vector_60,
    resample_to_length,
)
from ultralytics import YOLO

VIDEO_EXTS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


class FPSBenchmark:
    """Benchmark per-video FPS and per-stage latency."""

    def __init__(
        self,
        pose_weights: str = "yolo11n-pose.pt",
        cls_weights: str = "best_hybrid_transformer.pth",
        infer_stride: int = 15,
        device: str | None = None,
    ) -> None:
        self.infer_stride = infer_stride
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load YOLO
        self.pose = YOLO(pose_weights)

        # Load Transformer
        self.model = HybridFallTransformer().to(self.device)
        ckpt = torch.load(cls_weights, map_location=self.device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.threshold = float(ckpt.get("best_threshold", 0.5))

        self.stats: dict[str, list[float]] = {
            "read_ms": [],
            "pre_ms": [],
            "pose_ms": [],
            "vec_ms": [],
            "tfm_ms": [],
            "post_ms": [],
            "fps": [],
            "fall_detected": [],
            "total_frames": [],
        }

    def _extract_vec(self, frame: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Extract 60-D vector from frame. Returns (vec, latency_ms)."""
        t0 = time.perf_counter()
        frame_resized = cv2.resize(frame, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
        h, w = frame_resized.shape[:2]
        pre_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results = self.pose.predict(frame_resized, imgsz=IMGSZ, verbose=False)
        pose_ms = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        if not results or results[0].keypoints is None:
            return None, pre_ms + pose_ms + (time.perf_counter() - t2) * 1000

        kall = results[0].keypoints.data.cpu().numpy()
        if kall.size == 0:
            return None, pre_ms + pose_ms + (time.perf_counter() - t2) * 1000

        best_i = int(np.argmax([float(k[:, 2].mean()) for k in kall]))
        k = kall[best_i].astype(np.float32)

        if float(k[:, 2].mean()) < MIN_MEAN_CONF:
            return None, pre_ms + pose_ms + (time.perf_counter() - t2) * 1000

        kn = k.copy()
        kn[:, 0] /= float(w)
        kn[:, 1] /= float(h)
        vec = frame_to_vector_60(kn)
        vec_ms = (time.perf_counter() - t2) * 1000

        return vec, pre_ms + pose_ms + vec_ms

    def benchmark_video(self, video_path: str | Path) -> dict:
        """Run full pipeline on one video. Returns per-stage timings and FPS."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": f"Cannot open {video_path}"}

        feat_buffer: list[np.ndarray] = []
        total_frames = 0
        loop_times: list[float] = []

        read_times, pre_times, pose_times, vec_times, tfm_times, post_times = [], [], [], [], [], []

        t_loop_start = time.perf_counter()

        while True:
            t_read = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            read_ms = (time.perf_counter() - t_read) * 1000

            t_pre = time.perf_counter()
            frame_resized = cv2.resize(frame, (IMGSZ, IMGSZ))
            h, w = frame_resized.shape[:2]
            pre_ms = (time.perf_counter() - t_pre) * 1000

            t_pose = time.perf_counter()
            results = self.pose.predict(frame_resized, imgsz=IMGSZ, verbose=False)
            pose_ms = (time.perf_counter() - t_pose) * 1000

            vec: np.ndarray | None = None
            if results and results[0].keypoints is not None and results[0].keypoints.data is not None:
                kall = results[0].keypoints.data.cpu().numpy()
                if kall.size > 0:
                    best_i = int(np.argmax([float(k[:, 2].mean()) for k in kall]))
                    k = kall[best_i].astype(np.float32)
                    if float(k[:, 2].mean()) >= MIN_MEAN_CONF:
                        kn = k.copy()
                        kn[:, 0] /= float(w)
                        kn[:, 1] /= float(h)
                        vec = frame_to_vector_60(kn)

            t_vec = time.perf_counter()
            if vec is not None:
                feat_buffer.append(vec.astype(np.float32))
            vec_ms = (time.perf_counter() - t_vec) * 1000

            # Transformer inference
            tfm_ms = 0.0
            if len(feat_buffer) >= 8 and total_frames % self.infer_stride == 0:
                t_tfm = time.perf_counter()
                seq = np.stack(feat_buffer[-SEQ_LEN:], axis=0)
                seq_fixed = resample_to_length(seq, SEQ_LEN)
                x = torch.from_numpy(seq_fixed).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    prob = float(torch.sigmoid(self.model(x)).cpu().item())
                tfm_ms = (time.perf_counter() - t_tfm) * 1000

            t_post = time.perf_counter()
            post_ms = (time.perf_counter() - t_post) * 1000

            loop_ms = (time.perf_counter() - t_loop_start) * 1000

            read_times.append(read_ms)
            pre_times.append(pre_ms)
            pose_times.append(pose_ms)
            vec_times.append(vec_ms)
            tfm_times.append(tfm_ms)
            post_times.append(post_ms)
            loop_times.append(loop_ms)

            total_frames += 1
            t_loop_start = time.perf_counter()

        cap.release()

        total_time_ms = sum(loop_times)
        avg_fps = total_frames / (total_time_ms / 1000) if total_time_ms > 0 else 0

        return {
            "video": str(video_path),
            "total_frames": total_frames,
            "total_time_s": total_time_ms / 1000,
            "avg_fps": avg_fps,
            "read_ms_avg": np.mean(read_times),
            "pre_ms_avg": np.mean(pre_times),
            "pose_ms_avg": np.mean(pose_times),
            "vec_ms_avg": np.mean(vec_times),
            "tfm_ms_avg": np.mean(tfm_times) if tfm_times else 0,
            "post_ms_avg": np.mean(post_times),
            "pose_only_fps": 1000 / np.mean(pose_times),
        }

    def run(self, video_paths: list[str | Path]) -> list[dict]:
        """Benchmark all videos."""
        results = []
        for vp in video_paths:
            print(f"Processing: {vp}")
            r = self.benchmark_video(vp)
            if "error" in r:
                print(f"  ERROR: {r['error']}")
            else:
                print(
                    f"  FPS: {r['avg_fps']:.2f} | "
                    f"Pose: {r['pose_ms_avg']:.2f}ms | "
                    f"Transform: {r['tfm_ms_avg']:.2f}ms | "
                    f"Frames: {r['total_frames']}"
                )
            results.append(r)
        return results

    def save_csv(self, results: list[dict], output_path: str | Path) -> None:
        """Save results to CSV."""
        if not results:
            return
        fieldnames = list(results[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_path}")


def find_videos(path: str | Path) -> list[Path]:
    """Recursively find all video files in a directory."""
    path = Path(path)
    if path.is_file():
        return [path]
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(path.rglob(f"*{ext}"))
    return sorted(videos)


def print_summary(results: list[dict]) -> None:
    """Print summary table."""
    valid = [r for r in results if "error" not in r and r.get("avg_fps", 0) > 0]
    if not valid:
        print("No valid results.")
        return

    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY")
    print("=" * 100)

    # Per-video
    print(f"\n{'Video':<40} {'FPS':>8} {'Pose ms':>10} {'TFM ms':>10} {'Frames':>8}")
    print("-" * 80)
    for r in valid:
        print(
            f"{Path(r['video']).name:<40} "
            f"{r['avg_fps']:>8.2f} "
            f"{r['pose_ms_avg']:>10.2f} "
            f"{r['tfm_ms_avg']:>10.2f} "
            f"{r['total_frames']:>8}"
        )

    # Aggregate
    fps_vals = [r["avg_fps"] for r in valid]
    pose_ms_vals = [r["pose_ms_avg"] for r in valid]
    tfm_ms_vals = [r["tfm_ms_avg"] for r in valid]

    print("-" * 80)
    print(
        f"{'AVERAGE':<40} "
        f"{np.mean(fps_vals):>8.2f} "
        f"{np.mean(pose_ms_vals):>10.2f} "
        f"{np.mean(tfm_ms_vals):>10.2f} "
        f"{sum(r['total_frames'] for r in valid):>8}"
    )
    print(f"{'MIN':<40} {np.min(fps_vals):>8.2f}")
    print(f"{'MAX':<40} {np.max(fps_vals):>8.2f}")
    print(f"{'STD':<40} {np.std(fps_vals):>8.2f}")

    # SOTA comparison
    print("\n" + "=" * 100)
    print("SOTA COMPARISON (FPS)")
    print("=" * 100)
    sota = [
        ("Liu et al. (2022)", "URFD", 30.0),
        ("Han et al. (2023)", "Multiple", 20.0),
        ("Xu et al. (2024)", "Multiple", 25.0),
        ("Kurniadi et al. (2026)", "LE2I", 22.0),
        ("Benabdennour et al. (2026)", "URFD", 28.0),
        ("**Ours (YOLOv11n-Pose)**", "AIO", np.mean(fps_vals)),
    ]
    print(f"\n{'Method':<45} {'Dataset':<12} {'FPS':>8}")
    print("-" * 70)
    for name, dataset, fps in sota:
        marker = " ★" if "Ours" in name else ""
        print(f"{name:<45} {dataset:<12} {fps:>8.2f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FPS on fall detection videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_fps.py --video-dir /path/to/videos --output fps_results.csv
  python benchmark_fps.py --videos video1.mp4 video2.mp4 video3.avi --output results.csv
  python benchmark_fps.py --video-dir /path/to/videos --output results.csv --device cpu
        """,
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing videos",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        type=str,
        default=None,
        help="List of video file paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fps_benchmark_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--pose-weights",
        type=str,
        default="yolo11n-pose.pt",
        help="YOLO pose weights path",
    )
    parser.add_argument(
        "--cls-weights",
        type=str,
        default="best_hybrid_transformer.pth",
        help="Transformer classifier weights path",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Inference stride (default: 15)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "cuda:0"],
        help="Device (default: auto)",
    )

    args = parser.parse_args()

    if not args.video_dir and not args.videos:
        parser.error("Specify --video-dir or --videos")

    if args.video_dir:
        videos = find_videos(args.video_dir)
        if not videos:
            print(f"No videos found in: {args.video_dir}")
            sys.exit(1)
        print(f"Found {len(videos)} videos in: {args.video_dir}")
    else:
        videos = [Path(v) for v in args.videos]
        videos = [v for v in videos if v.exists()]
        if not videos:
            print("No valid video files provided.")
            sys.exit(1)

    print(f"Device: {args.device or 'auto'}")
    print(f"Using pose weights: {args.pose_weights}")
    print(f"Using classifier weights: {args.cls_weights}")
    print()

    bench = FPSBenchmark(
        pose_weights=args.pose_weights,
        cls_weights=args.cls_weights,
        infer_stride=args.stride,
        device=args.device,
    )

    results = bench.run(videos)
    bench.save_csv(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
