#!/usr/bin/env python
"""
Process MCFD (Multiple Cameras Fall Dataset) for Fall Detection (using centralized src modules).

Usage:
    python process_mcfd.py --input /path/to/mcfd --csv /path/to/csv --output /path/to/output

This script is a thin wrapper around src.data_prep.process_mcfd().
"""

import argparse
import os
import sys
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from ultralytics import YOLO

from src.data_prep import (
    OUTPUT_DIR as DEFAULT_OUTPUT,
    process_video_segment,
    standardize_to_60x60,
    _safe_name,
)
from src.config import (
    MCFD_DIR as DEFAULT_INPUT,
    MCFD_CSV as DEFAULT_CSV,
    YOLO_MODEL,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process MCFD dataset (wrapper for src.data_prep)"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help=f"Input directory (default: {DEFAULT_INPUT})"
    )
    parser.add_argument(
        "--csv", "-c", type=str, default=None,
        help=f"Annotations CSV (default: {DEFAULT_CSV})"
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
    csv_path = Path(args.csv) if args.csv else MCFD_CSV
    output_dir = Path(args.output) if args.output else DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {data_dir}")
    print(f"CSV:    {csv_path}")
    print(f"Output: {output_dir}")

    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(YOLO_MODEL)

    # Load annotations
    print("\nLoading annotations...")
    if not csv_path.exists():
        print(f"MCFD CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} annotations")

    # Process segments
    print("\nProcessing MCFD...")
    import numpy as np
    zero_fallback = np.zeros(60, dtype=np.float32)
    processed = skipped = errors = 0

    for _idx, row in tqdm(df.iterrows(), total=len(df), desc="MCFD"):
        chute = int(row["chute"])
        cam = int(row["cam"])
        start = int(row["start"])
        end = int(row["end"])
        label = int(row["label"])

        video_path = data_dir / f"chute{chute:02d}" / f"cam{cam}.avi"

        if not video_path.exists():
            errors += 1
            continue

        x_name = f"X_mcfd_c{chute:02d}_cam{cam}_{_idx}.npy"
        y_name = f"y_mcfd_c{chute:02d}_cam{cam}_{_idx}.npy"
        x_path = output_dir / x_name
        y_path = output_dir / y_name

        if x_path.exists() and y_path.exists():
            skipped += 1
            continue

        try:
            raw_features = process_video_segment(
                video_path, start, end, model, zero_fallback
            )
            if raw_features is None or len(raw_features) == 0:
                errors += 1
                continue

            features = standardize_to_60x60(raw_features)
            np.save(x_path, features)
            np.save(y_path, np.array([label], dtype=np.int32))
            processed += 1

        except Exception as e:
            errors += 1
            print(f"ERR row {_idx}: {e}")

    print(f"\n=== DONE ===")
    print(f"Total: {len(df)}, Processed: {processed}, Skipped: {skipped}, Errors: {errors}")

    # Auto zip
    zip_path = output_dir.parent / "mcfd_processed.zip"
    print(f"\nCreating zip: {zip_path}")
    shutil.make_archive(str(output_dir), 'zip', str(output_dir))
    print("Done!")


if __name__ == "__main__":
    main()
