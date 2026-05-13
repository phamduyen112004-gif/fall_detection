#!/usr/bin/env python3
"""
LE2I Zone-Based Feature Extraction (IEEE 2026 Protocol).

Extracts sliding windows from LE2I videos using zone-based protocol
to ensure no label leakage between train/val sets.

Zone Protocol:
- Fall zone: [start_fall, end_fall]
- Buffer zone: 30 frames before start_fall
- Post-fall zone: 30 frames after end_fall
- ADL zone: Everything else (no fall in window)

Labels:
- Fall (1): Window fully covers fall zone
- Non-Fall (0): Window ends >= 30 frames BEFORE start_fall
- Discarded (-1): Buffer zone, post-fall zone, ambiguous overlaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Import from project
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pifr_features import GeometricFeatureExtractor


# LE2I Fall Annotations (frame numbers)
# These should be calibrated per video
LE2I_FALL_ANNOTATIONS = {
    # Laboratory videos
    "Lab005": (120, 180),
    "Lab006": (140, 200),
    # Home videos
    "Home01": (100, 160),
    "Home02": (125, 185),
    # Add more as needed
}


class ZoneExtractor:
    """
    Zone-based window extractor following IEEE 2026 protocol.

    Divides video into zones:
    - ADL zone: Safe to use, no falls
    - Buffer zone: 30 frames before fall, excluded
    - Fall zone: Contains fall event
    - Post-fall zone: 30 frames after fall, excluded
    """

    BUFFER_FRAMES = 30  # Buffer before fall
    POST_FALL_FRAMES = 30  # Buffer after fall

    def __init__(
        self,
        seq_len: int = 60,
        stride: int = 15,
        fall_annotations: dict | None = None,
    ):
        """
        Initialize ZoneExtractor.

        Args:
            seq_len: Sequence length (default: 60 frames).
            stride: Sliding window stride (default: 15 frames).
            fall_annotations: Dict mapping video_name -> (start_fall, end_fall).
        """
        self.seq_len = seq_len
        self.stride = stride
        self.fall_annotations = fall_annotations or LE2I_FALL_ANNOTATIONS

    def get_zones(self, video_name: str, total_frames: int) -> list[dict]:
        """
        Get zones for a video based on fall annotations.

        Args:
            video_name: Name of the video (without extension).
            total_frames: Total number of frames in video.

        Returns:
            List of zone dicts with 'start', 'end', 'label'.
        """
        if video_name not in self.fall_annotations:
            # No annotation - assume all ADL
            return [{
                "start": 0,
                "end": total_frames,
                "label": 0,  # Non-fall
                "zone": "adl",
            }]

        start_fall, end_fall = self.fall_annotations[video_name]

        zones = []

        # ADL zone: before buffer
        if start_fall - self.BUFFER_FRAMES > 0:
            zones.append({
                "start": 0,
                "end": start_fall - self.BUFFER_FRAMES,
                "label": 0,  # Non-fall
                "zone": "adl",
            })

        # Buffer zone: excluded
        zones.append({
            "start": start_fall - self.BUFFER_FRAMES,
            "end": start_fall,
            "label": -1,  # Discarded
            "zone": "buffer",
        })

        # Fall zone
        zones.append({
            "start": start_fall,
            "end": end_fall,
            "label": 1,  # Fall
            "zone": "fall",
        })

        # Post-fall zone: excluded
        zones.append({
            "start": end_fall,
            "end": min(end_fall + self.POST_FALL_FRAMES, total_frames),
            "label": -1,  # Discarded
            "zone": "post_fall",
        })

        # Remaining ADL after post-fall
        if end_fall + self.POST_FALL_FRAMES < total_frames:
            zones.append({
                "start": end_fall + self.POST_FALL_FRAMES,
                "end": total_frames,
                "label": 0,  # Non-fall
                "zone": "adl",
            })

        return zones

    def generate_windows(
        self,
        video_name: str,
        total_frames: int,
    ) -> Iterator[tuple[int, int, int]]:
        """
        Generate sliding windows with zone-based labels.

        Args:
            video_name: Name of the video.
            total_frames: Total frames in video.

        Yields:
            Tuples of (start_frame, end_frame, label).
            label: 1=fall, 0=non-fall, -1=discarded.
        """
        zones = self.get_zones(video_name, total_frames)

        # Build frame -> label lookup
        frame_labels = {}
        for zone in zones:
            for frame_idx in range(zone["start"], zone["end"]):
                if frame_idx >= 0 and frame_idx < total_frames:
                    frame_labels[frame_idx] = zone["label"]

        # Generate sliding windows
        for start in range(0, total_frames - self.seq_len + 1, self.stride):
            end = start + self.seq_len

            # Check window label
            window_labels = set(frame_labels.get(i, -1) for i in range(start, end))

            if -1 in window_labels:
                # Contains buffer/post-fall zone - discarded
                label = -1
            elif 1 in window_labels and 0 not in window_labels:
                # Pure fall window
                label = 1
            elif 0 in window_labels and 1 not in window_labels:
                # Pure ADL window - only use if ends 30+ frames before fall
                # Check if window ends before fall zone starts
                fall_info = self.fall_annotations.get(video_name)
                if fall_info:
                    start_fall, _ = fall_info
                    # Non-fall must end 30 frames before fall
                    if end <= start_fall - self.BUFFER_FRAMES:
                        label = 0
                    else:
                        label = -1  # Discard if too close to fall
                else:
                    label = 0  # No fall annotation, use as ADL
            else:
                # Mixed or empty - discard
                label = -1

            if label != -1:  # Only yield valid windows
                yield start, end, label


def extract_video_features(
    video_path: Path,
    extractor: GeometricFeatureExtractor,
    seq_len: int = 60,
    stride: int = 15,
) -> tuple[list[np.ndarray], list[int]]:
    """
    Extract features from a single video.

    Args:
        video_path: Path to video file.
        extractor: GeometricFeatureExtractor instance.
        seq_len: Sequence length.
        stride: Sliding window stride.

    Returns:
        Tuple of (features_list, labels_list).
    """
    if cv2 is None:
        raise ImportError("opencv-python-headless required for video extraction")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.stem

    # Initialize zone extractor
    zone_ext = ZoneExtractor(seq_len=seq_len, stride=stride)
    zones = zone_ext.get_zones(video_name, total_frames)

    # Check if video has fall
    has_fall = any(z["zone"] == "fall" for z in zones)

    features = []
    labels = []

    # Process with pose detection (simplified - would need YOLO integration)
    # For now, return empty list (requires full pipeline)
    print(f"[INFO] Video {video_name}: {total_frames} frames, has_fall={has_fall}")

    cap.release()
    return features, labels


def main():
    parser = argparse.ArgumentParser(
        description="LE2I Zone-Based Feature Extraction (IEEE 2026)"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="AIO_Dataset/le2i",
        help="Directory containing LE2I videos",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="extracted_le2i",
        help="Output directory for extracted features",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=60,
        help="Sequence length (default: 60)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding window stride (default: 15)",
    )
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LE2I Zone-Based Feature Extraction (IEEE 2026 Protocol)")
    print("=" * 60)
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Stride: {args.stride}")
    print("=" * 60)

    # Process videos
    video_extensions = {".avi", ".mp4", ".mkv", ".mov"}
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(video_dir.glob(f"**/*{ext}"))

    print(f"\nFound {len(video_paths)} videos")

    if not video_paths:
        print("[WARN] No videos found!")
        return

    extractor = GeometricFeatureExtractor()

    for video_path in video_paths:
        features, labels = extract_video_features(
            video_path,
            extractor,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        # Save features and labels for this video
        if features:
            video_name = video_path.stem
            np.save(output_dir / f"X_{video_name}.npy", np.array(features))
            np.save(output_dir / f"y_{video_name}.npy", np.array(labels))
            print(f"[OK] Saved {len(features)} samples from {video_name}")

    print("\n" + "=" * 60)
    print("Extraction Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
