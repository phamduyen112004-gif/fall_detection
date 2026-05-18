#!/usr/bin/env python
"""
Process CaucaFall dataset for Fall Detection (using centralized src modules).

Usage:
    python process_caucafall.py --input /path/to/caucafall --output /path/to/output

This script is a thin wrapper around src.data_prep.process_caucafall().
"""

import argparse
import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

from src.data_prep import (
    OUTPUT_DIR as DEFAULT_OUTPUT,
    process_video_full,
    standardize_to_60x60,
    _is_fall,
    _safe_name,
)
from src.config import (
    CAUCAFALL_DIR as DEFAULT_INPUT,
    YOLO_MODEL,
    CONF_THRESHOLD,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process CaucaFall dataset (wrapper for src.data_prep)"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help=f"Input directory (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--device", "-d", type=str, default="cpu",
        help="Device: cpu or cuda (default: cpu)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.input) if args.input else DEFAULT_INPUT
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {data_dir}")
    print(f"Output: {output_dir}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

    # Collect video list
    print("\nScanning CaucaFall...")
    video_list = []
    import numpy as np
    zero_fallback = np.zeros(60, dtype=np.float32)

    if not data_dir.exists():
        print(f"CaucaFall directory not found: {data_dir}")
        return

    for subj_dir in sorted(os.listdir(data_dir)):
        subj_path = data_dir / subj_dir
        if not subj_path.is_dir() or not subj_dir.startswith("Subject."):
            continue

        for action_dir in sorted(os.listdir(subj_path)):
            action_path = subj_path / action_dir
            if not action_path.is_dir():
                continue

            for video in sorted(os.listdir(action_path)):
                if not video.endswith(".avi"):
                    continue

                video_list.append({
                    "path": action_path / video,
                    "subject": subj_dir,
                    "action": action_dir,
                    "label": _is_fall(action_dir),
                })

    print(f"Found {len(video_list)} videos")

    # Process videos
    print("\nProcessing CaucaFall...")
    processed = skipped = errors = 0

    for item in tqdm(video_list, desc="CaucaFall"):
        x_name = f"X_cauca_{_safe_name(item['subject'])}_{_safe_name(item['action'])}.npy"
        y_name = f"y_cauca_{_safe_name(item['subject'])}_{_safe_name(item['action'])}.npy"
        x_path = output_dir / x_name
        y_path = output_dir / y_name

        if x_path.exists() and y_path.exists():
            skipped += 1
            continue

        try:
            raw_features = process_video_full(item["path"], model, zero_fallback)
            if raw_features is None or len(raw_features) == 0:
                errors += 1
                continue

            features = standardize_to_60x60(raw_features)
            np.save(x_path, features)
            np.save(y_path, np.array([item["label"]], dtype=np.int32))
            processed += 1

            if processed % 50 == 0:
                print(f"  Processed: {processed}")

        except Exception as e:
            errors += 1
            print(f"ERR: {item['subject']}/{item['action']}: {e}")

    print(f"\n=== DONE ===")
    print(f"Total: {len(video_list)}, Processed: {processed}, Skipped: {skipped}, Errors: {errors}")

    # Auto zip
    zip_path = output_dir.parent / "caucafall_processed.zip"
    print(f"\nCreating zip: {zip_path}")
    shutil.make_archive(str(output_dir), 'zip', str(output_dir))
    print("Done!")


if __name__ == "__main__":
    main()
