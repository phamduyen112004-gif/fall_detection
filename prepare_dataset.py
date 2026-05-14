#!/usr/bin/env python3
"""
Chuẩn bị AIO_Dataset từ URFD + GMDCSA-24 + LE2I.

Gộp tất cả dataset vào một thư mục AIO_Dataset/{fall,nofall}/:
  - URFD: extract zip hoặc copy clip folders
  - GMDCSA-24: copy video từ Subject */Fall|fall, ADL|adl

Usage:
  python prepare_dataset.py \
      --urfd-root URFD_Raw \
      --gmdcsa-root GMDCSA_Raw \
      --out AIO_Dataset

Author: Fall Detection Team
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

# --- Constants ---
FALL_FOLDER_NAMES = ("Fall", "fall", "FALL")
ADL_FOLDER_NAMES = ("ADL", "adl", "Adl")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
URFD_FALL_ZIP_DIRS = ("fall", "Fall", "FALL")
URFD_ADL_ZIP_DIRS = ("adl", "ADL", "Adl")
LE2I_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s or "clip"


# =============================================================================
# URFD Functions
# =============================================================================

def extract_urfd_clips(urfd_root: Path, aio_root: Path) -> int:
    """Extract URFD clips (zip or extracted folders) into AIO_Dataset."""

    def _has_frame_images(folder: Path) -> bool:
        if not folder.is_dir():
            return False
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                return True
        return False

    def _extract_one_src(src_dir: Path, dest_parent: Path, tag: str) -> int:
        if not src_dir.is_dir():
            return 0
        n = 0
        dest_parent.mkdir(parents=True, exist_ok=True)
        seen_stems: set[str] = set()

        # Case 1: zip clips
        for zp in sorted(src_dir.rglob("*.zip"), key=lambda x: str(x).lower()):
            stem = _safe_stem(zp.stem)
            seen_stems.add(stem)
            out_dir = dest_parent / f"urfd_{tag}_{stem}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(out_dir)
            print(f"[URFD] {zp} -> {out_dir}")
            n += 1

        # Case 2: extracted clip folders
        for p in sorted(src_dir.iterdir(), key=lambda x: x.name.lower()):
            if not p.is_dir():
                continue
            if not _has_frame_images(p):
                continue
            stem = _safe_stem(p.name)
            if stem in seen_stems:
                continue
            out_dir = dest_parent / f"urfd_{tag}_{stem}"
            if out_dir.exists():
                shutil.rmtree(out_dir)
            shutil.copytree(p, out_dir)
            print(f"[URFD] {p} -> {out_dir}")
            n += 1
        return n

    n_fall = 0
    for name in URFD_FALL_ZIP_DIRS:
        d = urfd_root / name
        if d.is_dir():
            n_fall += _extract_one_src(d, aio_root / "fall", "fall")
            break

    n_adl = 0
    for name in URFD_ADL_ZIP_DIRS:
        d = urfd_root / name
        if d.is_dir():
            n_adl += _extract_one_src(d, aio_root / "nofall", "adl")
            break

    if n_fall == 0 and n_adl == 0:
        print(f"[warn] URFD: no clips found in {urfd_root}")
    return n_fall + n_adl


# =============================================================================
# GMDCSA-24 Functions
# =============================================================================

def subject_slug(subject_dir: Path) -> str:
    m = re.search(r"(\d+)", subject_dir.name)
    if m:
        return f"subject{m.group(1)}"
    return _safe_stem(subject_dir.name).lower()


def _first_existing_subdir(parent: Path, names: tuple[str, ...]) -> Path | None:
    for n in names:
        p = parent / n
        if p.is_dir():
            return p
    return None


def _list_videos_in_dir(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    out: list[Path] = []
    for ext in ("*.mp4", "*.MP4", "*.avi", "*.AVI", "*.mov", "*.MOV"):
        out.extend(d.glob(ext))
    uniq = {p.resolve(): p for p in out}
    return sorted(uniq.values(), key=lambda p: p.name.lower())


def _filename_column(fieldnames: list[str] | None) -> str | None:
    if not fieldnames:
        return None
    stripped = [f.strip() for f in fieldnames]
    norm_to_orig = {
        re.sub(r"[^\w]+", "_", f.lower()).strip("_"): f for f in stripped
    }
    for key in ("file_name", "filename", "file"):
        if key in norm_to_orig:
            return norm_to_orig[key]
    for f in stripped:
        fl = f.lower()
        if "file" in fl and "name" in fl:
            return f
    return stripped[0]


def _video_paths_from_index_csv(
    csv_path: Path,
    subj_dir: Path,
    subdirs_first: tuple[str, ...],
) -> list[Path]:
    if not csv_path.is_file():
        return []
    out: list[Path] = []
    with csv_path.open(newline="", encoding="utf-8-sig", errors="replace") as fp:
        reader = csv.DictReader(fp)
        col = _filename_column(reader.fieldnames)
        if not col:
            return []
        seen: set[str] = set()
        for row in reader:
            name = (row.get(col) or "").strip()
            if not name:
                continue
            if not name.lower().endswith(VIDEO_SUFFIXES):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            tries: list[Path] = []
            for sd in subdirs_first:
                tries.append(subj_dir / sd / name)
            tries.append(subj_dir / name)
            for p in tries:
                if p.is_file():
                    out.append(p)
                    break
    return out


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for p in paths:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _collect_gmdcsa_subject_videos(subj_dir: Path) -> tuple[list[Path], list[Path]]:
    fall_v: list[Path] = []
    adl_v: list[Path] = []

    fd = _first_existing_subdir(subj_dir, FALL_FOLDER_NAMES)
    if fd:
        fall_v.extend(_list_videos_in_dir(fd))
    if not fall_v:
        fall_v.extend(
            _video_paths_from_index_csv(
                subj_dir / "Fall.csv",
                subj_dir,
                ("Fall", "fall", "FALL"),
            )
        )

    ad = _first_existing_subdir(subj_dir, ADL_FOLDER_NAMES)
    if ad:
        adl_v.extend(_list_videos_in_dir(ad))
    if not adl_v:
        adl_v.extend(
            _video_paths_from_index_csv(
                subj_dir / "ADL.csv",
                subj_dir,
                ("ADL", "adl", "Adl"),
            )
        )

    return _dedupe_paths(fall_v), _dedupe_paths(adl_v)


def _copy_gmdcsa_clip(src: Path, dest_parent: Path, slug: str) -> None:
    dest_parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext not in VIDEO_SUFFIXES:
        ext = ".mp4"
    new_name = f"gmdcsa_{slug}_{_safe_stem(src.stem)}{ext}"
    dest = dest_parent / new_name
    shutil.copy2(src, dest)
    print(f"[GMDCSA] {src} -> {dest}")


def copy_gmdcsa_videos(gmdcsa_root: Path, aio_root: Path) -> int:
    """Copy GMDCSA-24 videos to AIO_Dataset."""
    n_total = 0
    for subj_dir in sorted(gmdcsa_root.iterdir(), key=lambda p: p.name.lower()):
        if not subj_dir.is_dir():
            continue
        slug = subject_slug(subj_dir)
        fall_v, adl_v = _collect_gmdcsa_subject_videos(subj_dir)
        for vid in fall_v:
            _copy_gmdcsa_clip(vid, aio_root / "fall", slug)
            n_total += 1
        for vid in adl_v:
            _copy_gmdcsa_clip(vid, aio_root / "nofall", slug)
            n_total += 1
        if not fall_v and not adl_v:
            print(f"[warn] GMDCSA {subj_dir.name}: no videos found")
    if n_total == 0:
        print(f"[warn] GMDCSA: no videos copied from {gmdcsa_root}")
    return n_total


# =============================================================================
# LE2I Functions
# =============================================================================

def _normalize_video_name(name: str) -> str:
    name = name.strip()
    for ext in LE2I_VIDEO_EXTENSIONS:
        if name.lower().endswith(ext.lower()):
            name = name[: -len(ext)]
    return name.strip()


def _is_annotation_file(path: Path) -> bool:
    """Annotation files are .txt or .xml at any level in LE2I dataset."""
    ext = path.suffix.lower()
    if ext not in (".txt", ".xml", ".csv"):
        return False
    # Annotation files are typically small text files with frame numbers
    try:
        size = path.stat().st_size
        if size > 500_000:  # skip large files
            return False
    except OSError:
        return False
    return True


def _parse_le2i_annotation(ann_file: Path) -> tuple[int, int, list]:
    """Parse LE2I annotation file. Returns (start_fall, end_fall, frame_data)."""
    start_fall = -1
    end_fall = -1
    frame_data: list = []

    if not ann_file.is_file():
        return -1, -1, []

    try:
        lines = ann_file.read_text(encoding="utf-8", errors="replace").strip().split("\n")
    except Exception:
        return -1, -1, []

    if len(lines) < 3:
        return -1, -1, []

    try:
        start_fall = int(lines[0].strip())
        end_fall = int(lines[1].strip())
    except ValueError:
        pass

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            frame_idx = int(parts[0].strip())
            label = int(parts[1].strip())
            bbox = [int(p.strip()) for p in parts[2:6]] if len(parts) >= 6 else []
            frame_data.append((frame_idx, label, bbox))
        except ValueError:
            continue

    if start_fall < 0 or end_fall < 0:
        fall_frames = [f for f, l, _ in frame_data if l in (7, 8)]
        if fall_frames:
            start_fall = min(fall_frames)
            end_fall = max(fall_frames)
        else:
            start_fall, end_fall = -1, -1

    return start_fall, end_fall, frame_data


def _find_le2i_scenes_and_videos(root: Path) -> tuple[list, dict]:
    """Find all LE2I scenes, videos and annotations."""
    results: list = []
    annotations_info: dict[str, tuple[int, int]] = {}

    # Build annotation lookup: file stem -> annotation file path
    ann_lookup: dict[str, Path] = {}
    for af in root.rglob("*"):
        if af.is_file() and _is_annotation_file(af):
            ann_lookup[_normalize_video_name(af.stem)] = af

    # Scan all directories up to 2 levels deep for videos
    for scene_parent in root.iterdir():
        if not scene_parent.is_dir():
            continue

        # Try: Home_01/video.avi OR Home_01/Video/video.avi
        for f in scene_parent.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in LE2I_VIDEO_EXTENSIONS:
                continue

            norm_name = _normalize_video_name(f.stem)
            ann_file = ann_lookup.get(norm_name)
            if ann_file:
                del ann_lookup[norm_name]  # consume to avoid re-use

            results.append((scene_parent, f, ann_file))
            if ann_file:
                start, end, _ = _parse_le2i_annotation(ann_file)
                annotations_info[norm_name] = (start, end)

    return results, annotations_info


def _classify_video(start_fall: int, end_fall: int) -> int:
    """Classify video: 1=fall, 0=nofall"""
    if start_fall > 0 and end_fall > 0 and end_fall > start_fall:
        return 1
    return 0


def extract_le2i_clips(
    le2i_root: Path,
    aio_root: Path,
) -> tuple[int, dict[str, Any]]:
    """Extract LE2I dataset into AIO_Dataset/{fall,nofall}/."""
    (aio_root / "fall").mkdir(parents=True, exist_ok=True)
    (aio_root / "nofall").mkdir(parents=True, exist_ok=True)

    video_mapping: dict[str, dict[str, Any]] = {}
    total_copied = 0

    print(f"[LE2I] Scanning: {le2i_root}")

    items, _ = _find_le2i_scenes_and_videos(le2i_root)

    if not items:
        print("[LE2I] No videos found!")
        return 0, {}

    print(f"[LE2I] Found {len(items)} videos")

    scenes: dict[str, list] = {}
    for scene_path, video_path, ann_file in items:
        scene_name = _safe_stem(scene_path.name).lower()
        if scene_name not in scenes:
            scenes[scene_name] = []
        scenes[scene_name].append((scene_path, video_path, ann_file))

    for scene_name, scene_items in sorted(scenes.items()):
        print(f"\n[LE2I] Scene: {scene_name} ({len(scene_items)} videos)")

        for scene_path, video_path, ann_file in scene_items:
            norm_name = _normalize_video_name(video_path.name)

            start_fall, end_fall = -1, -1
            frame_data: list = []
            if ann_file:
                start_fall, end_fall, frame_data = _parse_le2i_annotation(ann_file)

            label = _classify_video(start_fall, end_fall)

            label_tag = "fall" if label == 1 else "nofall"
            stem = _safe_stem(video_path.stem)
            ext = video_path.suffix.lower()
            if ext not in LE2I_VIDEO_EXTENSIONS:
                ext = ".avi"
            new_name = f"le2i_{scene_name}_{label_tag}_{stem}{ext}"

            dest_dir = aio_root / ("fall" if label == 1 else "nofall")
            dest_path = dest_dir / new_name

            if dest_path.exists():
                status = "FALL" if label == 1 else "ADL "
                print(f"  [SKIP] {new_name} (already exists)")
                video_mapping[new_name.lower()] = {
                    "path": str(dest_path),
                    "label": label,
                    "source": str(video_path),
                    "slug": f"le2i_{scene_name}",
                    "scene": scene_name,
                    "start_fall": start_fall,
                    "end_fall": end_fall,
                    "total_frames": len(frame_data),
                }
                total_copied += 1
                continue

            try:
                # Try hardlink first (no extra disk space on same filesystem)
                try:
                    import os as _os
                    _os.link(video_path, dest_path)
                    status = "FALL" if label == 1 else "ADL "
                    if start_fall > 0 and end_fall > 0:
                        print(f"  [{status}] {video_path.name} -> {new_name} [hardlink]")
                        print(f"           Fall frames: {start_fall} - {end_fall}")
                    else:
                        print(f"  [{status}] {video_path.name} -> {new_name} [hardlink] (no annotation)")
                except OSError:
                    # Fallback to copy
                    shutil.copy2(video_path, dest_path)
                    status = "FALL" if label == 1 else "ADL "
                    if start_fall > 0 and end_fall > 0:
                        print(f"  [{status}] {video_path.name} -> {new_name}")
                        print(f"           Fall frames: {start_fall} - {end_fall}")
                    else:
                        print(f"  [{status}] {video_path.name} -> {new_name} (no annotation)")
            except Exception as e:
                print(f"  [ERROR] Failed to copy {video_path.name}: {e}")
                continue

            video_mapping[new_name.lower()] = {
                "path": str(dest_path),
                "label": label,
                "source": str(video_path),
                "slug": f"le2i_{scene_name}",
                "scene": scene_name,
                "start_fall": start_fall,
                "end_fall": end_fall,
                "total_frames": len(frame_data),
            }

            total_copied += 1

    # Save metadata
    meta_path = aio_root / "_le2i_annotations.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(video_mapping, f, indent=2, ensure_ascii=False)

    fall_count = sum(1 for v in video_mapping.values() if v["label"] == 1)
    nofall_count = sum(1 for v in video_mapping.values() if v["label"] == 0)
    with_ann = sum(1 for v in video_mapping.values() if v["start_fall"] >= 0)

    print(f"\n[LE2I] === Summary ===")
    print(f"  Total videos: {total_copied}")
    print(f"  Fall videos:   {fall_count}")
    print(f"  ADL videos:    {nofall_count}")
    print(f"  With annotation: {with_ann}")
    print(f"  Metadata saved: {meta_path}")

    return total_copied, video_mapping


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(
        description="Chuẩn bị AIO_Dataset từ URFD + GMDCSA-24 + LE2I",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Datasets:
  URFD: University of Rome Fall Dataset (zip clips hoặc extracted folders)
  GMDCSA-24: GMDCSA24 Dataset (video files)
  LE2I: LE2I Fall Detection Dataset (video + annotation files)

Ví dụ:
  python prepare_dataset.py \\
      --urfd-root data/raw/URFD \\
      --gmdcsa-root data/raw/GMDCSA24 \\
      --le2i-root data/raw/LE2I \\
      --out AIO_Dataset

  # URFD và GMDCSA (không có LE2I):
  python prepare_dataset.py --urfd-root URFD --gmdcsa-root GMDCSA --out AIO_Dataset
        """,
    )
    ap.add_argument(
        "--urfd-root",
        type=Path,
        default=None,
        help="Thư mục cha chứa Fall|fall và ADL|adl (URFD)",
    )
    ap.add_argument(
        "--gmdcsa-root",
        type=Path,
        default=None,
        help="Thư mục cha GMDCSA-24 (chứa Subject */Fall|ADL)",
    )
    ap.add_argument(
        "--le2i-root",
        type=Path,
        default=None,
        help="Thư mục gốc LE2I dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("AIO_Dataset"),
        help="Thư mục output (AIO_Dataset)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail nếu không chuẩn bị được clip nào",
    )

    args = ap.parse_args()

    aio = args.out
    (aio / "fall").mkdir(parents=True, exist_ok=True)
    (aio / "nofall").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AIO Dataset Preparation")
    print("=" * 60)

    n_urfd = 0
    if args.urfd_root:
        if args.urfd_root.is_dir():
            n_urfd = extract_urfd_clips(args.urfd_root, aio)
        else:
            print(f"[warn] URFD root not found: {args.urfd_root}")

    n_gmdcsa = 0
    if args.gmdcsa_root:
        if args.gmdcsa_root.is_dir():
            n_gmdcsa = copy_gmdcsa_videos(args.gmdcsa_root, aio)
        else:
            print(f"[warn] GMDCSA root not found: {args.gmdcsa_root}")

    n_le2i = 0
    if args.le2i_root:
        if args.le2i_root.is_dir():
            n_le2i, _ = extract_le2i_clips(args.le2i_root, aio)
        else:
            print(f"[warn] LE2I root not found: {args.le2i_root}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  URFD clips:  {n_urfd}")
    print(f"  GMDCSA videos: {n_gmdcsa}")
    print(f"  LE2I videos:   {n_le2i}")
    print(f"  Output: {aio}")

    if args.strict and (n_urfd + n_gmdcsa + n_le2i) == 0:
        raise SystemExit("No clips prepared (GMDCSA+LE2I empty).")


if __name__ == "__main__":
    main()
