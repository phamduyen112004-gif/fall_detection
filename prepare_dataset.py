#!/usr/bin/env python3
"""
Gộp URFD (zip khung hình) và GMDCSA-24 vào AIO_Dataset/{fall,nofall}/.

URFD: zip thường nằm trong `Fall`/`fall` và `ADL`/`adl` dưới thư mục gốc (ví dụ `data/raw/URFD/ADL/*.zip`).

GMDCSA-24 (Zenodo): mỗi subject có Fall/ADL (hoặc fall/adl) chứa video, kèm Fall.csv / ADL.csv.
Có thể trỏ thẳng bản đã giải nén, ví dụ:
  python prepare_dataset.py --skip-urfd --gmdcsa-root data/raw/GMDCSA24 --out AIO_Dataset

  python prepare_dataset.py --urfd-root URFD_Raw --gmdcsa-root GMDCSA_Raw --out AIO_Dataset
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from pathlib import Path

# Thư mục video GMDCSA24 trên Zenodo: Subject N/Fall, Subject N/ADL (hoặc fall/adl chữ thường).
FALL_FOLDER_NAMES = ("Fall", "fall", "FALL")
ADL_FOLDER_NAMES = ("ADL", "adl", "Adl")
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")

# URFD: thư mục zip theo từng bản phát hành (chữ hoa / thường).
URFD_FALL_ZIP_DIRS = ("fall", "Fall", "FALL")
URFD_ADL_ZIP_DIRS = ("adl", "ADL", "Adl")


def _safe_stem(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    return s or "clip"


def extract_urfd_zips(urfd_root: Path, aio_root: Path) -> None:
    """<urfd_root>/Fall|fall/*.zip -> AIO fall/; <urfd_root>/ADL|adl/*.zip -> AIO nofall/."""

    def _extract_one_src(src_dir: Path, dest_parent: Path, tag: str) -> int:
        if not src_dir.is_dir():
            return 0
        n = 0
        dest_parent.mkdir(parents=True, exist_ok=True)
        for zp in sorted(src_dir.glob("*.zip")):
            stem = _safe_stem(zp.stem)
            out_dir = dest_parent / f"urfd_{tag}_{stem}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(out_dir)
            print(f"[URFD] {zp.name} -> {out_dir}")
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
        print(
            f"[warn] URFD: không thấy thư mục zip (fall/Fall hoặc adl/ADL) dưới {urfd_root}. "
            "Dùng --urfd-root trỏ tới thư mục cha chứa ADL và Fall, ví dụ: data/raw/URFD"
        )


def subject_slug(subject_dir: Path) -> str:
    """Subject_1 -> subject1"""
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
    """Đọc Fall.csv / ADL.csv; tìm file video trong subj_dir/<Fall|ADL>/ hoặc trực tiếp trong subject."""
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
    """(fall_videos, adl_videos): ưu tiên thư mục Fall/ADL, nếu trống thì theo Fall.csv / ADL.csv."""
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


def copy_gmdcsa_videos(gmdcsa_root: Path, aio_root: Path) -> None:
    """GMDCSA-24: Subject */Fall|fall hoặc ADL|adl, hoặc chỉ số từ Fall.csv / ADL.csv (Zenodo)."""
    for subj_dir in sorted(gmdcsa_root.iterdir(), key=lambda p: p.name.lower()):
        if not subj_dir.is_dir():
            continue
        slug = subject_slug(subj_dir)
        fall_v, adl_v = _collect_gmdcsa_subject_videos(subj_dir)
        for vid in fall_v:
            _copy_gmdcsa_clip(vid, aio_root / "fall", slug)
        for vid in adl_v:
            _copy_gmdcsa_clip(vid, aio_root / "nofall", slug)
        if not fall_v and not adl_v:
            print(
                f"[warn] GMDCSA {subj_dir.name}: không thấy video. "
                "Cần thư mục Fall/ADL chứa .mp4 hoặc file .mp4 trùng tên trong Fall.csv/ADL.csv."
            )


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Chuẩn bị AIO_Dataset từ URFD + GMDCSA-24")
    ap.add_argument(
        "--urfd-root",
        type=Path,
        default=Path("URFD_Raw"),
        help="Thư mục cha chứa Fall|fall và ADL|adl với file .zip (vd. data/raw/URFD)",
    )
    ap.add_argument("--gmdcsa-root", type=Path, default=Path("GMDCSA_Raw"))
    ap.add_argument("--out", type=Path, default=Path("AIO_Dataset"))
    ap.add_argument("--skip-urfd", action="store_true")
    ap.add_argument("--skip-gmdcsa", action="store_true")
    args = ap.parse_args()

    aio = args.out
    (aio / "fall").mkdir(parents=True, exist_ok=True)
    (aio / "nofall").mkdir(parents=True, exist_ok=True)

    if not args.skip_urfd and args.urfd_root.is_dir():
        extract_urfd_zips(args.urfd_root, aio)
    elif not args.skip_urfd:
        print(f"[warn] Không thấy {args.urfd_root} — bỏ qua URFD.")

    if not args.skip_gmdcsa and args.gmdcsa_root.is_dir():
        copy_gmdcsa_videos(args.gmdcsa_root, aio)
    elif not args.skip_gmdcsa:
        print(f"[warn] Không thấy {args.gmdcsa_root} — bỏ qua GMDCSA.")

    print(f"Hoàn tất. Cấu trúc: {aio}/fall, {aio}/nofall")


if __name__ == "__main__":
    main()
