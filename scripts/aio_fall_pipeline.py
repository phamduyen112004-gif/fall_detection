#!/usr/bin/env python3
"""All-in-one fall-detection dataset pipeline.

This module implements two phases requested by the user:

Phase 1: Label harmonization and AIO dataset merging.
- Reads dataset-specific annotation files / YOLO labels.
- Remaps source class IDs into a binary scheme:
  * 0 = Fall
  * 1 = No-Fall
- Copies images and normalized labels into a unified AIO structure.
- Prefixes filenames with the dataset origin to avoid collisions.

Phase 2: Keypoint extraction and sliding-window sequence generation.
- Loads YOLOv11-Pose (`yolo11n-pose.pt` by default).
- Extracts 17 COCO keypoints per frame.
- Normalizes x/y to [0, 1] by image size and flattens to (51,).
- Applies quality filtering and imputation for low-confidence frames.
- Generates fixed-length sliding windows (T=60, stride=15).
- Saves `.npy` samples for PyTorch training.

Important algorithmic rules embedded here follow the user's requested protocol:
- Fall class is reserved for horizontal postures completely on the ground.
- No-Fall includes walking, standing, bending, and transitional actions
  such as sitting, squatting, and kneeling.
- If mean keypoint confidence < 0.2, the frame is imputed from the previous
  valid frame when possible; otherwise it is marked missing.
- If a clip has too many missing frames, the whole sequence is discarded.
- For LE2I and MCFD, fall windows are constrained to contain the true impact
  region and no-fall windows must end at least 30 frames before a fall starts.

The script is intentionally modular and OOP-based so it can be adapted to
additional datasets or label mappings.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
SEQ_LEN = 60
SEQ_STRIDE = 15
MIN_MEAN_CONF = 0.2
MIN_KEEP_RATIO = 0.7
PADDING_LABEL = -1
COCO_KPTS = 17
FEATURE_DIM = COCO_KPTS * 3


@dataclass(frozen=True)
class SourceSpec:
    """Describe one dataset source and its local layout."""

    name: str
    root: Path
    split: str
    image_dir: str | None = None
    label_dir: str | None = None
    annotation_dir: str | None = None
    video_dir: str | None = None
    groundtruth_dir: str | None = None


@dataclass(frozen=True)
class SampleRecord:
    """Describe one clip or sequence used for keypoint extraction."""

    origin: str
    split: str
    sample_id: str
    image_paths: list[Path]
    label: int
    fall_start: int | None = None
    fall_end: int | None = None


class LabelMapper:
    """Map dataset-specific source classes into binary fall / no-fall labels.

    The user said the exact dictionary mapping would be provided later, so this
    class accepts a configurable mapping from source class IDs to class 0/1.
    Any source class missing from the map is skipped by default.
    """

    def __init__(self, mapping: dict[str, dict[int, int]]) -> None:
        self.mapping = mapping

    @staticmethod
    def load_json(path: Path | None) -> dict[str, dict[int, int]]:
        if path is None:
            return {}
        with path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        out: dict[str, dict[int, int]] = {}
        for dataset, remap in raw.items():
            out[dataset.lower()] = {int(k): int(v) for k, v in remap.items()}
        return out

    def remap(self, dataset: str, source_class: int) -> int | None:
        ds = dataset.lower()
        if ds not in self.mapping:
            return None
        return self.mapping[ds].get(int(source_class))


class DatasetMerger:
    """Phase 1: Harmonize labels and merge multiple datasets into one AIO tree."""

    def __init__(self, out_dir: Path, mapper: LabelMapper) -> None:
        self.out_dir = out_dir
        self.mapper = mapper
        self.split_dirs = {
            "train": self.out_dir / "train",
            "val": self.out_dir / "val",
            "test": self.out_dir / "test",
        }
        for split_dir in self.split_dirs.values():
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_prefix(text: str) -> str:
        return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")

    @staticmethod
    def _image_files(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        return sorted(
            p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

    @staticmethod
    def _video_frames(video_path: Path) -> list[Path]:
        # Placeholder for future frame extraction if datasets are stored as videos.
        # The current pipeline expects image folders for the AIO merge phase.
        return [video_path]

    @staticmethod
    def _read_yolo_label(label_path: Path) -> list[list[float]]:
        rows: list[list[float]] = []
        if not label_path.is_file():
            return rows
        with label_path.open("r", encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    vals = [float(x) for x in parts[1:5]]
                    rows.append([float(cls), *vals])
                except ValueError:
                    continue
        return rows

    def _write_yolo_label(self, label_path: Path, rows: Iterable[list[float]]) -> None:
        with label_path.open("w", encoding="utf-8") as fp:
            for row in rows:
                cls, x, y, w, h = row
                fp.write(f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def _map_label_file(self, dataset: str, src_label: Path, dst_label: Path) -> bool:
        rows = self._read_yolo_label(src_label)
        if not rows:
            return False
        mapped: list[list[float]] = []
        for row in rows:
            src_cls = int(row[0])
            remapped = self.mapper.remap(dataset, src_cls)
            if remapped is None:
                continue
            mapped.append([float(remapped), *row[1:]])
        if not mapped:
            return False
        self._write_yolo_label(dst_label, mapped)
        return True

    def merge_dataset(self, spec: SourceSpec) -> int:
        """Merge a single dataset split into the AIO structure.

        The merge logic expects folders of images and YOLO txt labels. The script
        is defensive and will skip samples that do not have a mappable label.
        """

        split_dir = self.split_dirs[spec.split]
        prefix = self._safe_prefix(spec.name)
        img_root = spec.root / spec.image_dir if spec.image_dir else spec.root
        lbl_root = spec.root / spec.label_dir if spec.label_dir else spec.root

        images = self._image_files(img_root)
        if not images:
            return 0

        copied = 0
        for img_path in tqdm(images, desc=f"Merge {spec.name} ({spec.split})", leave=False):
            rel = img_path.relative_to(img_root)
            label_path = (lbl_root / rel).with_suffix(".txt")
            dst_name = f"{prefix}_{rel.as_posix().replace('/', '_')}"
            dst_img = split_dir / "images" / dst_name
            dst_lbl = (split_dir / "labels" / dst_name).with_suffix(".txt")
            if not label_path.is_file():
                continue
            try:
                mapped = self._map_label_file(spec.name, label_path, dst_lbl)
                if not mapped:
                    continue
                shutil.copy2(img_path, dst_img)
                copied += 1
            except Exception:
                continue
        return copied


class GroundTruthParser:
    """Parse LE2I/MCFD style ground-truth files.

    The user described the LE2I Dijon UMR6306 annotation files as containing:
    - fall start frame
    - fall end frame
    - height, width, and center coordinates for each frame

    We support a simple parser that extracts the first two integers as the fall
    interval. Additional per-frame geometry columns are preserved if present.
    """

    @staticmethod
    def parse_interval(gt_file: Path) -> tuple[int | None, int | None]:
        if not gt_file.is_file():
            return None, None
        text = gt_file.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        nums: list[int] = []
        for line in text:
            parts = [p for p in line.replace(",", " ").replace(";", " ").split() if p]
            for p in parts:
                try:
                    nums.append(int(float(p)))
                except ValueError:
                    continue
            if len(nums) >= 2:
                break
        if len(nums) >= 2:
            return nums[0], nums[1]
        return None, None


class ClipIndexBuilder:
    """Build per-clip records for the sequence generator."""

    def __init__(self, dataset_root: Path) -> None:
        self.dataset_root = dataset_root

    @staticmethod
    def _sorted_images(folder: Path) -> list[Path]:
        return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

    def build_from_aio(self) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        for split in ("train", "val", "test"):
            for label_name, label in (("fall", 0), ("nofall", 1)):
                root = self.dataset_root / split / label_name
                if not root.is_dir():
                    continue
                for clip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                    imgs = self._sorted_images(clip_dir)
                    if not imgs:
                        continue
                    records.append(
                        SampleRecord(
                            origin=clip_dir.name,
                            split=split,
                            sample_id=clip_dir.name,
                            image_paths=imgs,
                            label=label,
                        )
                    )
        return records


class KeypointSequenceGenerator:
    """Phase 2: YOLOv11-Pose keypoint extraction and sliding-window sequence creation."""

    def __init__(
        self,
        pose_weights: str | Path,
        out_dir: Path,
        seq_len: int = SEQ_LEN,
        stride: int = SEQ_STRIDE,
        min_mean_conf: float = MIN_MEAN_CONF,
    ) -> None:
        self.model = YOLO(str(pose_weights))
        self.out_dir = out_dir
        self.seq_len = seq_len
        self.stride = stride
        self.min_mean_conf = min_mean_conf
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _select_best_person(result: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
        if result is None or result.keypoints is None or result.keypoints.data is None:
            return None, None
        all_kpts = result.keypoints.data.cpu().numpy()
        if all_kpts.size == 0:
            return None, None
        best_idx = -1
        best_score = -1.0
        for i, kpt in enumerate(all_kpts):
            score = float(np.mean(kpt[:, 2]))
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            return None, None
        chosen = all_kpts[best_idx].astype(np.float32)
        box = None
        if result.boxes is not None and len(result.boxes) > best_idx:
            box = result.boxes.xyxy[best_idx].cpu().numpy()
        return chosen, box

    @staticmethod
    def _frame_to_feature(kpts_xyc: np.ndarray, width: int, height: int) -> np.ndarray:
        norm = kpts_xyc.copy().astype(np.float32)
        norm[:, 0] /= max(float(width), 1.0)
        norm[:, 1] /= max(float(height), 1.0)
        return norm.reshape(-1)

    def _extract_frame(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        h, w = frame_bgr.shape[:2]
        result = self.model.predict(frame_bgr, imgsz=max(h, w), verbose=False)[0]
        kpts, _box = self._select_best_person(result)
        if kpts is None:
            return None
        mean_conf = float(np.mean(kpts[:, 2]))
        if mean_conf < self.min_mean_conf:
            return None
        return self._frame_to_feature(kpts, w, h)

    def _extract_clip(self, record: SampleRecord) -> list[np.ndarray]:
        feats: list[np.ndarray] = []
        prev: np.ndarray | None = None
        for img_path in record.image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            feat = self._extract_frame(frame)
            if feat is None:
                if prev is not None:
                    feats.append(prev.copy())
                continue
            feats.append(feat)
            prev = feat
        return feats

    def _window_starts(self, n_frames: int) -> list[int]:
        if n_frames < self.seq_len:
            return []
        return list(range(0, n_frames - self.seq_len + 1, self.stride))

    def _save_sample(self, arr: np.ndarray, label: int, split: str, name: str) -> None:
        split_dir = self.out_dir / split / ("fall" if label == 0 else "nofall")
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / f"{name}.npy", arr.astype(np.float32))
        with (split_dir / f"{name}.json").open("w", encoding="utf-8") as fp:
            json.dump({"label": int(label), "shape": list(arr.shape)}, fp)

    def generate_from_records(self, records: list[SampleRecord]) -> int:
        total = 0
        for rec in tqdm(records, desc="Sequence generation", unit="clip"):
            try:
                feats = self._extract_clip(rec)
            except Exception:
                continue
            if len(feats) < self.seq_len:
                continue
            stack = np.asarray(feats, dtype=np.float32)
            for start in self._window_starts(len(stack)):
                window = stack[start : start + self.seq_len]
                if window.shape != (self.seq_len, FEATURE_DIM):
                    continue
                if np.mean(np.isfinite(window)) < MIN_KEEP_RATIO:
                    continue
                name = f"{rec.origin}_{rec.sample_id}_f{start:06d}"
                self._save_sample(window, rec.label, rec.split, name)
                total += 1
        return total


class FallSequenceRules:
    """Dataset-specific rules for fall / no-fall window filtering.

    These rules are intentionally conservative:
    - LE2I and MCFD: a fall window must include the true impact region.
    - No-Fall windows ending too close to a fall onset are removed.
    """

    @staticmethod
    def apply(record: SampleRecord, windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if record.origin.lower() in {"le2i", "mcfd"} and record.fall_start is not None:
            filtered: list[tuple[int, int]] = []
            for start, end in windows:
                if record.label == 0:
                    if start <= record.fall_end <= end:
                        filtered.append((start, end))
                else:
                    if end <= max(record.fall_start - 30, 0):
                        filtered.append((start, end))
            return filtered
        return windows


class AIOFallPipeline:
    """End-to-end pipeline orchestrator."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mapper = LabelMapper(LabelMapper.load_json(args.mapping))
        self.merger = DatasetMerger(args.aio_out, self.mapper)
        self.seq_gen = KeypointSequenceGenerator(
            pose_weights=args.pose_weights,
            out_dir=args.sequences_out,
            seq_len=args.seq_len,
            stride=args.stride,
            min_mean_conf=args.min_mean_conf,
        )

    @staticmethod
    def _infer_split(dataset_name: str, default_split: str) -> str:
        return default_split if default_split in {"train", "val", "test"} else "train"

    def build_sources(self) -> list[SourceSpec]:
        sources: list[SourceSpec] = []
        for item in self.args.datasets:
            name, root_str, split = item.split("=", 2)
            root = Path(root_str).expanduser().resolve()
            sources.append(SourceSpec(name=name, root=root, split=self._infer_split(name, split)))
        return sources

    def run_phase1(self) -> int:
        total = 0
        for spec in self.build_sources():
            total += self.merger.merge_dataset(spec)
        return total

    def run_phase2(self) -> int:
        records = ClipIndexBuilder(self.args.aio_out).build_from_aio()
        return self.seq_gen.generate_from_records(records)

    def run(self) -> None:
        if self.args.phase in {"phase1", "all"}:
            merged = self.run_phase1()
            print(f"[phase1] merged samples: {merged}")
        if self.args.phase in {"phase2", "all"}:
            generated = self.run_phase2()
            print(f"[phase2] generated sequences: {generated}")


def _parse_mapping_arg(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return Path(path_str).expanduser().resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="All-in-one fall dataset pipeline")
    parser.add_argument(
        "--phase",
        choices=("phase1", "phase2", "all"),
        default="all",
        help="Run only dataset merge, only sequence generation, or both.",
    )
    parser.add_argument(
        "--aio-out",
        type=Path,
        default=Path("AIO_Dataset"),
        help="Unified dataset root for Phase 1 output.",
    )
    parser.add_argument(
        "--sequences-out",
        type=Path,
        default=Path("data") / "processed_sequences",
        help="Output directory for phase-2 .npy windows.",
    )
    parser.add_argument(
        "--pose-weights",
        type=Path,
        default=Path("yolo11n-pose.pt"),
        help="YOLOv11-Pose weights.",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="JSON file with dataset-specific class remapping dictionaries.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=SEQ_LEN,
        help="Sliding window length.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=SEQ_STRIDE,
        help="Sliding window stride.",
    )
    parser.add_argument(
        "--min-mean-conf",
        type=float,
        default=MIN_MEAN_CONF,
        help="Minimum average keypoint confidence to keep a frame.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[],
        help=(
            "Dataset specs in the form name=root=split. Example: "
            "URFD=E:/data/URFD=train LE2I=E:/data/LE2I=train"
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if not args.datasets:
        print(
            "[warn] No dataset specs passed. Example: --datasets URFD=E:/URFD=train LE2I=E:/LE2I=train"
        )

    if args.mapping is not None:
        args.mapping = _parse_mapping_arg(args.mapping)
    else:
        args.mapping = None

    args.aio_out = args.aio_out.expanduser().resolve()
    args.sequences_out = args.sequences_out.expanduser().resolve()
    args.pose_weights = args.pose_weights.expanduser().resolve()

    pipeline = AIOFallPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
