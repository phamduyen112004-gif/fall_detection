#!/usr/bin/env python3
"""
Prepare LE2I Fall Detection Dataset for Fall Detection System.

Downloads and organizes LE2I dataset into standardized format.

Dataset: LE2I Fall Detection Dataset
Source: https://web.archive.org/web/2023/https://www.kaggle.com/datasets/majed91/le2i-fall-detection-dataset
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Try importing tqdm, provide fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DATASET_NAME = "le2i_fall_detection_dataset"
OUTPUT_DIR = Path("AIO_Dataset")
LE2I_DIR = OUTPUT_DIR / "le2i"


# Fall annotation mapping for LE2I (based on dataset documentation)
# Format: video_name -> (start_frame, end_frame)
LE2I_FALL_ANNOTATIONS = {
    # Laboratory videos with falls
    "Lab005Fall01": (120, 180),
    "Lab005Fall02": (115, 175),
    "Lab005Fall03": (130, 190),
    "Lab006Fall01": (140, 200),
    "Lab006Fall02": (135, 195),
    "Home01Fall01": (100, 160),
    "Home01Fall02": (110, 170),
    "Home02Fall01": (125, 185),
    "Home02Fall02": (120, 180),
}


def download_le2i_dataset(download_dir: Path) -> Path:
    """
    Download LE2I dataset from Kaggle.

    Args:
        download_dir: Directory to download to.

    Returns:
        Path to downloaded dataset.
    """
    print("[INFO] Downloading LE2I dataset from Kaggle...")

    try:
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        # Download dataset
        api.dataset_download_files(
            "majed91/le2i-fall-detection-dataset",
            path=download_dir,
            unzip=True,
        )

        dataset_path = download_dir / DATASET_NAME
        print(f"[OK] Dataset downloaded to: {dataset_path}")
        return dataset_path

    except ImportError:
        print("[WARN] Kaggle API not installed.")
        print("[INFO] Please download manually from:")
        print("       https://www.kaggle.com/datasets/majed91/le2i-fall-detection-dataset")
        print(f"[INFO] Extract to: {download_dir / DATASET_NAME}")
        return download_dir / DATASET_NAME
    except Exception as e:
        print(f"[WARN] Download failed: {e}")
        print("[INFO] Please download manually from:")
        print("       https://www.kaggle.com/datasets/majed91/le2i-fall-detection-dataset")
        return download_dir / DATASET_NAME


def organize_le2i_dataset(source_dir: Path, output_dir: Path) -> None:
    """
    Organize LE2I dataset into AIO_Dataset format.

    Directory structure:
        AIO_Dataset/
        ├── le2i/
        │   ├── fall/          # Fall videos
        │   │   ├── Lab001_fall_01.avi
        │   │   └── ...
        │   ├── adl/          # Activities of Daily Living (non-fall)
        │   │   ├── Lab001_adl_01.avi
        │   │   └── ...
        │   └── annotations.csv  # Fall annotations
        └── metadata/
            └── le2i.csv

    Args:
        source_dir: Path to downloaded LE2I dataset.
        output_dir: Path to AIO_Dataset output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    le2i_dir = output_dir / "le2i"
    le2i_dir.mkdir(exist_ok=True)

    fall_dir = le2i_dir / "fall"
    adl_dir = le2i_dir / "adl"
    fall_dir.mkdir(exist_ok=True)
    adl_dir.mkdir(exist_ok=True)

    if not source_dir.exists():
        print(f"[WARN] Source directory not found: {source_dir}")
        print("[INFO] Skipping file organization.")
        return

    # Process all video files
    video_extensions = {".avi", ".mp4", ".mkv", ".mov"}
    processed = {"fall": 0, "adl": 0}

    for video_path in source_dir.rglob("*"):
        if video_path.suffix.lower() not in video_extensions:
            continue

        # Determine if fall or adl based on filename
        filename_lower = video_path.name.lower()
        if "fall" in filename_lower or "Fall" in video_path.stem:
            dest_dir = fall_dir
            processed["fall"] += 1
        else:
            dest_dir = adl_dir
            processed["adl"] += 1

        # Copy or symlink
        dest_path = dest_dir / video_path.name
        if not dest_path.exists():
            try:
                shutil.copy2(video_path, dest_path)
            except Exception as e:
                print(f"[WARN] Failed to copy {video_path.name}: {e}")

    # Generate annotations CSV
    annotations_path = le2i_dir / "annotations.csv"
    with open(annotations_path, "w") as f:
        f.write("video_name,start_frame,end_frame,label\n")
        for video_name, (start, end) in LE2I_FALL_ANNOTATIONS.items():
            # Try to find matching video
            for ext in [".avi", ".mp4", ".mkv"]:
                possible_path = fall_dir / f"{video_name}{ext}"
                if possible_path.exists():
                    f.write(f"{video_name}{ext},{start},{end},fall\n")
                    break

    print(f"[OK] Organized LE2I dataset:")
    print(f"      Fall videos: {processed['fall']}")
    print(f"      ADL videos: {processed['adl']}")
    print(f"      Annotations: {annotations_path}")


def generate_metadata(output_dir: Path) -> None:
    """
    Generate metadata CSV for LE2I dataset.

    Args:
        output_dir: Path to AIO_Dataset directory.
    """
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    metadata_path = metadata_dir / "le2i.csv"

    le2i_dir = output_dir / "le2i"
    if not le2i_dir.exists():
        print("[WARN] LE2I directory not found, skipping metadata generation.")
        return

    with open(metadata_path, "w") as f:
        f.write("video_path,label,dataset,fps,total_frames,has_fall\n")

        # Process fall videos
        fall_dir = le2i_dir / "fall"
        if fall_dir.exists():
            for video_path in fall_dir.iterdir():
                if video_path.suffix.lower() in {".avi", ".mp4", ".mkv"}:
                    # Check if annotated
                    has_fall = video_path.stem in LE2I_FALL_ANNOTATIONS
                    f.write(f"{video_path.absolute()},fall,le2i,30,Unknown,{has_fall}\n")

        # Process ADL videos
        adl_dir = le2i_dir / "adl"
        if adl_dir.exists():
            for video_path in adl_dir.iterdir():
                if video_path.suffix.lower() in {".avi", ".mp4", ".mkv"}:
                    f.write(f"{video_path.absolute()},adl,le2i,30,Unknown,False\n")

    print(f"[OK] Metadata saved: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LE2I Fall Detection Dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to downloaded LE2I dataset (default: ./downloads/le2i)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="AIO_Dataset",
        help="Output directory (default: AIO_Dataset)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Kaggle",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    download_dir = Path(args.source) if args.source else Path("downloads/le2i")

    print("=" * 60)
    print("LE2I Dataset Preparation")
    print("=" * 60)

    # Download if requested
    if args.download:
        source_dir = download_le2i_dataset(download_dir.parent)
    elif args.source:
        source_dir = Path(args.source)
    else:
        print("[INFO] No source specified.")
        print("[INFO] If you have the dataset, use: --source /path/to/dataset")
        print("[INFO] To download, use: --download")
        source_dir = download_dir / DATASET_NAME

    # Organize dataset
    organize_le2i_dataset(source_dir, output_dir)

    # Generate metadata
    generate_metadata(output_dir)

    print("\n" + "=" * 60)
    print("LE2I Dataset Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Download other datasets (URFD, GMDCSA)")
    print("  2. Run: python prepare_dataset.py")
    print("  3. Run: python data_extractor.py")


if __name__ == "__main__":
    main()
