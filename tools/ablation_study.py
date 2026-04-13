"""
Ablation Study — quét lưới 3 tham số (Torso × Fall frames × Pose conf).

Chạy offline trên video, không gửi Telegram. Xuất CSV + JSON (frame báo động).

Ví dụ:
  python tools/ablation_study.py --video data/fall.mp4 --video data/normal.mp4
  python tools/ablation_study.py --fetch-samples
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# Thư mục gốc dự án
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig
from src.pipeline import HybridFallPipeline
from src.stage2_pose import PoseExtractor


@dataclass
class AblationResult:
    video_name: str
    video_path: str
    torso_deg: float
    fall_min_frames: int
    min_mean_keypoint_conf: float
    total_frames: float
    fps: float
    num_fall_alerts: int
    alert_frame_indices: list[int]
    wall_seconds: float


def run_one_video(
    video_path: Path,
    torso: float,
    fall_frames: int,
    min_conf: float,
    nose_ankle_fixed: float,
    mirror: bool,
    stride: int,
    max_frames: int | None,
    shared_pose: PoseExtractor,
) -> AblationResult:
    cfg = PipelineConfig(
        laydown_torso_angle_deg=torso,
        laydown_nose_ankle_angle_deg=nose_ankle_fixed,
        fall_min_frames=fall_frames,
        fall_min_seconds=None,
        min_mean_keypoint_conf=min_conf,
    )
    pipe = HybridFallPipeline(cfg, pose_extractor=shared_pose)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    alerts: list[int] = []
    frame_i = 0
    t0 = time.perf_counter()

    while True:
        if max_frames is not None and frame_i >= max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        if stride > 1 and frame_i % stride != 0:
            frame_i += 1
            continue
        diag = pipe.process_frame(frame)
        if diag.fall_confirmed:
            alerts.append(frame_i)
        frame_i += 1

    cap.release()
    wall = time.perf_counter() - t0

    return AblationResult(
        video_name=video_path.name,
        video_path=str(video_path.resolve()),
        torso_deg=torso,
        fall_min_frames=fall_frames,
        min_mean_keypoint_conf=min_conf,
        total_frames=float(frame_i),
        fps=fps,
        num_fall_alerts=len(alerts),
        alert_frame_indices=alerts,
        wall_seconds=round(wall, 2),
    )


def fetch_sample(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Tải mẫu: {url} -> {dest}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; AblationStudy/1.0)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        dest.write_bytes(resp.read())
    print("Xong.")


def write_synthetic_video(dest: Path, frames: int = 240, w: int = 640, h: int = 360) -> None:
    """Video tối thiểu (không có người) — chỉ để smoke-test pipeline/ablation."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dest), fourcc, 24.0, (w, h))
    for i in range(frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, int((i * 3) % 200), 80)
        cv2.putText(
            frame,
            f"SYNTH {i}",
            (40, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        out.write(frame)
    out.release()
    print(f"Đã tạo video tổng hợp: {dest}")


def main() -> None:
    p = argparse.ArgumentParser(description="Ablation study (torso × frames × conf)")
    p.add_argument(
        "--video",
        action="append",
        default=[],
        help="Đường dẫn video (có thể lặp lại nhiều lần).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "ablation",
        help="Thư mục ghi CSV/JSON.",
    )
    p.add_argument(
        "--nose-ankle-deg",
        type=float,
        default=50.0,
        help="Giữ cố định ngưỡng góc mũi–cổ chân trong suốt ablation.",
    )
    p.add_argument("--mirror", action="store_true")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Chỉ xử lý mỗi N frame (tăng tốc, kết quả chỉ mang tính tương đối).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Giới hạn số frame đọc từ mỗi video (None = hết video).",
    )
    p.add_argument(
        "--fetch-samples",
        action="store_true",
        help="Tải 2 clip mẫu ngắn (proxy) vào data/samples/ rồi thoát.",
    )
    args = p.parse_args()

    data_dir = ROOT / "data" / "samples"
    if args.fetch_samples:
        # Thử tải clip công khai; nếu lỗi mạng → tạo 2 video tổng hợp để vẫn chạy được ablation.
        samples: list[tuple[str | None, Path]] = [
            (
                "https://archive.org/download/BigBuckBunny_356/BigBuckBunny_356kb.mp4",
                data_dir / "sample_activity_A.mp4",
            ),
            (
                "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
                data_dir / "sample_activity_B.mp4",
            ),
        ]
        for url, dest in samples:
            if dest.exists():
                print(f"Đã có sẵn: {dest}")
                continue
            if url:
                try:
                    fetch_sample(url, dest)
                    continue
                except Exception as e:
                    print(f"Tải thất bại ({url}): {e}")
            write_synthetic_video(dest)
        print(
            "Thêm video thật (vd. '50 ways to fall', sinh hoạt) vào data/ và chạy:\n"
            "  python tools/ablation_study.py --video data/ten_video.mp4"
        )
        return

    videos = [Path(v) for v in args.video]
    if not videos:
        p.error("Cần ít nhất một --video hoặc dùng --fetch-samples trước.")

    torsos = (45.0, 60.0, 75.0)
    fall_frames_list = (10, 30, 60)
    confs = (0.2, 0.4, 0.6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.out_dir / f"ablation_grid_{stamp}.csv"
    json_path = args.out_dir / f"ablation_grid_{stamp}.json"

    rows: list[dict] = []
    all_json: list[dict] = []

    print(
        "Lưới: torso",
        torsos,
        "× fall_min_frames",
        fall_frames_list,
        "× min_mean_keypoint_conf",
        confs,
        "| fall_min_seconds=None (chỉ đếm frame)",
    )
    print("Đang tải mô hình pose (một lần)…", flush=True)
    shared_pose = PoseExtractor(PipelineConfig())

    for vp in videos:
        if not vp.is_file():
            raise SystemExit(f"Không tìm thấy file: {vp}")
        for torso in torsos:
            for ff in fall_frames_list:
                for conf in confs:
                    tag = f"{vp.stem}_t{int(torso)}_f{ff}_c{conf}"
                    print(f"Chạy: {tag} ...", flush=True)
                    r = run_one_video(
                        vp,
                        torso=torso,
                        fall_frames=ff,
                        min_conf=conf,
                        nose_ankle_fixed=args.nose_ankle_deg,
                        mirror=args.mirror,
                        stride=args.stride,
                        max_frames=args.max_frames,
                        shared_pose=shared_pose,
                    )
                    d = asdict(r)
                    d["alert_frame_indices"] = ",".join(
                        str(x) for x in r.alert_frame_indices
                    )
                    rows.append(d)
                    all_json.append(
                        {
                            **asdict(r),
                            "note_tp_fp": "Điền TP/FP thủ công sau khi đối chiếu video.",
                        }
                    )

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_json, f, ensure_ascii=False, indent=2)

    tpl = args.out_dir / f"ablation_tp_fp_template_{stamp}.csv"
    with tpl.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "video",
                "torso_deg",
                "fall_min_frames",
                "min_conf",
                "num_alerts",
                "TP",
                "FP",
                "FN",
                "ghi_chu",
            ]
        )
        for r in all_json:
            w.writerow(
                [
                    r["video_name"],
                    r["torso_deg"],
                    r["fall_min_frames"],
                    r["min_mean_keypoint_conf"],
                    r["num_fall_alerts"],
                    "",
                    "",
                    "",
                    "",
                ]
            )

    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Mẫu ghi TP/FP: {tpl}")


if __name__ == "__main__":
    main()
