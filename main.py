"""
CLI chạy Hybrid Fall Detection Pipeline.

Ví dụ:
  python main.py --source 0
  python main.py --gui-transformer
  python main.py --source path/to/video.mp4 --no-show
"""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.pipeline import HybridFallPipeline, run_on_video


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid Fall Detection (YOLOv11-pose + hình học)")
    p.add_argument(
        "--gui",
        action="store_true",
        help="Mở giao diện Tkinter (Webcam / Tải video / Settings).",
    )
    p.add_argument(
        "--gui-transformer",
        action="store_true",
        help="PyQt5 + HybridFallTransformer (cần best_hybrid_transformer.pth).",
    )
    p.add_argument(
        "--source",
        default="0",
        help="Đường dẫn video hoặc chỉ số camera (mặc định 0).",
    )
    p.add_argument("--no-show", action="store_true", help="Không hiển thị cửa sổ OpenCV.")
    p.add_argument("--mirror", action="store_true", help="Lật ngang (selfie cam).")
    p.add_argument(
        "--fall-frames",
        type=int,
        default=None,
        help="Số frame tối thiểu ở tư thế nằm ngang trước khi báo ngã.",
    )
    p.add_argument(
        "--fall-seconds",
        type=float,
        default=None,
        help="Thời gian tối thiểu (giây) hoặc -1 để tắt ngưỡng giây.",
    )
    args = p.parse_args()

    if args.gui_transformer:
        import app_inference

        app_inference.main()
        return

    if args.gui:
        import gui_app

        gui_app.main()
        return

    src: str | int = args.source
    if isinstance(src, str) and src.isdigit():
        src = int(src)

    cfg_kw = {}
    if args.fall_frames is not None:
        cfg_kw["fall_min_frames"] = args.fall_frames
    if args.fall_seconds is not None:
        cfg_kw["fall_min_seconds"] = (
            None if args.fall_seconds < 0 else args.fall_seconds
        )
    cfg = PipelineConfig(**cfg_kw) if cfg_kw else PipelineConfig()

    if not args.no_show:
        run_on_video(src, config=cfg, show=True, mirror=args.mirror)
        return

    # Chế độ không hiển thị: chỉ log
    pipe = HybridFallPipeline(cfg)
    import cv2

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được nguồn: {src}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)
        diag = pipe.process_frame(frame)
        if diag.fall_confirmed:
            print("[FALL]", diag)
    cap.release()


if __name__ == "__main__":
    main()
