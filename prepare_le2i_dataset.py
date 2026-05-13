#!/usr/bin/env python3
"""
Chuẩn bị LE2I Fall Detection Dataset (Kaggle format) vào AIO_Dataset/{fall,nofall}/.

Cấu trúc Kaggle LE2I:
  Scene/
    Scene/
      Annotation_files/
        video (1).txt
        video (2).txt
      Videos/
        video (1).avi
        video (2).avi

Annotation format (LE2I):
  Line 1: start_fall (frame number bắt đầu ngã)
  Line 2: end_fall (frame number kết thúc ngã)
  Line 3+: frame, label, x1, y1, x2, y2
    - label=1: standing/normal (ADL)
    - label=7: lying down (post-fall)
    - label=8: falling (transition)
    - label=0: unknown/no person

Script này:
  1. Quét tất cả scene folders (Coffee_room_01, Dormitory, ...)
  2. Parse annotation để lấy start_fall/end_fall
  3. Copy video vào AIO_Dataset/{fall,nofall}/
  4. Lưu metadata vào _le2i_annotations.json

Output:
  AIO_Dataset/
    fall/   le2i_<scene>_<videoname>.avi
    nofall/ le2i_<scene>_<videoname>.avi
  AIO_Dataset/_le2i_annotations.json

Ví dụ:
  python prepare_le2i_dataset.py \\
      --le2i-root /kaggle/input/datasets/tuyenldvn/falldataset-imvia \\
      --out AIO_Dataset
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

# --- Cấu hình ---
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def _safe_stem(name: str) -> str:
    """Tạo stem an toàn từ tên file."""
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s or "clip"


def _normalize_video_name(name: str) -> str:
    """Chuẩn hóa tên video để match với annotation."""
    name = name.strip()
    for ext in VIDEO_EXTENSIONS:
        if name.lower().endswith(ext.lower()):
            name = name[: -len(ext)]
    return name.strip()


def _parse_le2i_annotation(ann_file: Path) -> tuple[int, int, list[tuple[int, int, list[int]]]]:
    """
    Parse LE2I annotation file (Kaggle format).

    Format:
      Line 1: start_fall (int)
      Line 2: end_fall (int)
      Line 3+: frame, label, x1, y1, x2, y2 (CSV)

    Returns:
        (start_fall, end_fall, frame_data)
        - start_fall: frame bắt đầu ngã (từ line 1)
        - end_fall: frame kết thúc ngã (từ line 2)
        - frame_data: list of (frame_idx, label, [x1,y1,x2,y2])
    """
    start_fall = -1
    end_fall = -1
    frame_data: list[tuple[int, int, list[int]]] = []

    if not ann_file.is_file():
        return -1, -1, []

    try:
        lines = ann_file.read_text(encoding="utf-8", errors="replace").strip().split("\n")
    except Exception:
        return -1, -1, []

    if len(lines) < 3:
        return -1, -1, []

    # Parse first two lines: start_fall and end_fall
    try:
        start_fall = int(lines[0].strip())
        end_fall = int(lines[1].strip())
    except ValueError:
        # Fallback: try to parse from frame data
        pass

    # Parse frame data (from line 3 onwards)
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

    # If start_fall/end_fall not in first 2 lines, try to infer from frame data
    if start_fall < 0 or end_fall < 0:
        # Find frames with label 7 or 8 (falling/lying)
        fall_frames = [f for f, l, _ in frame_data if l in (7, 8)]
        if fall_frames:
            start_fall = min(fall_frames)
            end_fall = max(fall_frames)
        else:
            start_fall, end_fall = -1, -1

    return start_fall, end_fall, frame_data


def _find_scenes_and_videos(root: Path) -> tuple[list[tuple[Path, Path, Path]], dict[str, tuple[int, int]]]:
    """
    Tìm tất cả scenes, videos và annotations.

    Returns:
        List of (scene_path, video_path, annotation_path)
        Dict mapping normalized video name -> (start_fall, end_fall)
    """
    results: list[tuple[Path, Path, Path]] = []
    annotations_info: dict[str, tuple[int, int]] = {}

    # Tìm tất cả thư mục chứa "Videos" hoặc "Annotation_files"
    for scene_parent in root.iterdir():
        if not scene_parent.is_dir():
            continue

        # Tìm nested folder cùng tên (LE2I format: Scene/Scene/Videos/)
        videos_dir = None
        ann_dir = None

        for child in scene_parent.iterdir():
            if not child.is_dir():
                continue

            child_name = child.name.lower()
            if "video" in child_name:
                videos_dir = child
            elif "annotation" in child_name:
                ann_dir = child

        # Nếu không có nested folder, kiểm tra trực tiếp
        if videos_dir is None:
            for child in scene_parent.iterdir():
                if child.is_dir() and "video" in child.name.lower():
                    videos_dir = child
                    break

        if ann_dir is None:
            for child in scene_parent.iterdir():
                if child.is_dir() and "annotation" in child.name.lower():
                    ann_dir = child
                    break

        # Nếu videos ở cùng cấp với scene parent
        if videos_dir is None:
            # Kiểm tra xem có video files trực tiếp không
            for f in scene_parent.iterdir():
                if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                    videos_dir = scene_parent
                    break

        if videos_dir is None:
            continue

        # Tìm annotation folder
        if ann_dir is None:
            # Tìm trong parent
            for sibling in root.iterdir():
                if sibling.is_dir() and sibling != scene_parent:
                    for sub in sibling.iterdir():
                        if sub.is_dir() and "annotation" in sub.name.lower():
                            ann_dir = sub
                            break

        # Collect videos và annotations
        if videos_dir.is_dir():
            for f in videos_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                norm_name = _normalize_video_name(f.name)

                # Tìm annotation file
                ann_file = None
                if ann_dir and ann_dir.is_dir():
                    for af in ann_dir.iterdir():
                        if af.is_file():
                            ann_norm = _normalize_video_name(af.name)
                            if ann_norm == norm_name:
                                ann_file = af
                                break

                results.append((scene_parent, f, ann_file))

                # Parse annotation
                if ann_file:
                    start, end, _ = _parse_le2i_annotation(ann_file)
                    annotations_info[norm_name] = (start, end)

    return results, annotations_info


def _classify_video(start_fall: int, end_fall: int) -> int:
    """
    Phân loại video dựa trên annotation.
    Returns: 1=fall, 0=nofall (ADL)
    """
    if start_fall > 0 and end_fall > 0 and end_fall > start_fall:
        return 1  # Có fall event
    return 0  # Không có fall (ADL)


def _make_output_name(scene: str, video_path: Path, label: int) -> str:
    """Tạo tên file output chuẩn: le2i_<scene>_<label>_<videoname>.avi"""
    label_tag = "fall" if label == 1 else "nofall"
    stem = _safe_stem(video_path.stem)
    ext = video_path.suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        ext = ".avi"
    return f"le2i_{scene}_{label_tag}_{stem}{ext}"


def extract_le2i_clips(
    le2i_root: Path,
    aio_root: Path,
) -> tuple[int, dict[str, Any]]:
    """
    Hàm chính: extract LE2I dataset vào AIO_Dataset/{fall,nofall}/.
    """
    (aio_root / "fall").mkdir(parents=True, exist_ok=True)
    (aio_root / "nofall").mkdir(parents=True, exist_ok=True)

    video_mapping: dict[str, dict[str, Any]] = {}
    total_copied = 0

    print(f"[LE2I] Scanning: {le2i_root}")

    # Tìm scenes, videos, annotations
    items, annotations_info = _find_scenes_and_videos(le2i_root)

    if not items:
        print("[LE2I] Không tìm thấy video nào!")
        return 0, {}

    print(f"[LE2I] Found {len(items)} videos")

    # Group by scene
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

            # Parse annotation
            start_fall, end_fall = -1, -1
            frame_data: list = []
            if ann_file:
                start_fall, end_fall, frame_data = _parse_le2i_annotation(ann_file)

            # Classify
            label = _classify_video(start_fall, end_fall)

            # Create output name
            new_name = _make_output_name(scene_name, video_path, label)
            dest_dir = aio_root / ("fall" if label == 1 else "nofall")
            dest_path = dest_dir / new_name

            # Copy video
            try:
                shutil.copy2(video_path, dest_path)
                status = "FALL" if label == 1 else "ADL "
                if start_fall > 0 and end_fall > 0:
                    print(f"  [{status}] {video_path.name} -> {new_name}")
                    print(f"           Fall frames: {start_fall} - {end_fall}")
                else:
                    print(f"  [{status}] {video_path.name} -> {new_name} (no fall annotation)")
            except Exception as e:
                print(f"  [ERROR] Failed to copy {video_path.name}: {e}")
                continue

            # Save metadata
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

    # Statistics
    fall_count = sum(1 for v in video_mapping.values() if v["label"] == 1)
    nofall_count = sum(1 for v in video_mapping.values() if v["label"] == 0)
    with_ann = sum(1 for v in video_mapping.values() if v["start_fall"] >= 0)

    print(f"\n[LE2I] === Summary ===")
    print(f"  Total videos: {total_copied}")
    print(f"  Fall videos:   {fall_count}")
    print(f"  ADL videos:    {nofall_count}")
    print(f"  With fall annotation: {with_ann}")
    print(f"  Metadata saved: {meta_path}")

    return total_copied, video_mapping


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Chuẩn bị LE2I Fall Detection Dataset vào AIO_Dataset/{fall,nofall}/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Dataset Kaggle LE2I
  python prepare_le2i_dataset.py \\
      --le2i-root /kaggle/input/datasets/tuyenldvn/falldataset-imvia \\
      --out AIO_Dataset
        """,
    )
    ap.add_argument(
        "--le2i-root",
        type=Path,
        required=True,
        help="Thư mục gốc LE2I dataset",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("AIO_Dataset"),
        help="Thư mục output (AIO_Dataset)",
    )
    args = ap.parse_args()

    if not args.le2i_root.is_dir():
        raise SystemExit(f"Không tìm thấy LE2I root: {args.le2i_root}")

    n, _ = extract_le2i_clips(args.le2i_root, args.out)
    if n == 0:
        raise SystemExit(
            f"Không tìm thấy video nào trong {args.le2i_root}. "
            "Kiểm tra lại cấu trúc thư mục."
        )
    print(f"\n[OK] Done. {n} videos prepared.")


if __name__ == "__main__":
    main()
