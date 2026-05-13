"""Orchestrator: 4 giai đoạn Hybrid Fall Detection.

Pipeline hoàn chỉnh bao gồm:
    Giai đoạn 1: Tiền xử lý frame (resize về 640x640)
    Giai đoạn 2: Trích xuất pose từ YOLOv11-Pose
    Giai đoạn 3: Phân tích động học (tính góc, phân loại tư thế)
    Giai đoạn 4: Cảnh báo Telegram khi phát hiện ngã

Module này điều phối luồng xử lý từ đầu vào (video/camera) đến đầu ra (cảnh báo).
Hỗ trợ cả chế độ heuristic (góc) và hybrid (Transformer + heuristic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .config import PipelineConfig
from .stage1_preprocess import preprocess_frame
from .stage2_pose import PoseExtractor
from .stage3_kinematics import (
    FallTemporalFilter,
    Posture,
    classify_posture,
    compute_pose_angles,
)
from .stage4_alert import TelegramAlerter
from .viz import draw_pose_overlay
from .types import FrameDiag


def _annotate_status(
    display_bgr: np.ndarray,
    diag: FrameDiag,
    extra: str | None = None,
) -> None:
    mc = (
        f"{diag.mean_kpt_conf:.2f}"
        if diag.mean_kpt_conf is not None
        else "—"
    )
    label = (
        f"{diag.posture} | thân={diag.torso_deg} | mui_cc={diag.nose_ankle_deg} "
        f"| conf_tb={mc}"
    )
    if extra:
        label = f"{label} | {extra}"
    cv2.putText(
        display_bgr,
        label[:96],
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )


class HybridFallPipeline:
    """
    Bộ điều phối pipeline phát hiện ngã lai ghép.

    Xử lý 4 giai đoạn:
        1. Tiền xử lý frame (resize về 640x640)
        2. Trích xuất pose từ YOLOv11-Pose
        3. Phân tích động học (tính góc, phân loại tư thế)
        4. Cảnh báo Telegram khi phát hiện ngã
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        pose_extractor: PoseExtractor | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.pose = pose_extractor or PoseExtractor(self.config)
        self.pose.config = self.config
        self.temporal = FallTemporalFilter(self.config)
        self.alerter = TelegramAlerter(self.config)

    def process_frame_with_display(
        self,
        frame_bgr: np.ndarray,
        on_fall: Callable[[np.ndarray, FrameDiag], None] | None = None,
    ) -> tuple[FrameDiag, np.ndarray]:
        """
        Xử lý một khung BGR; trả về (diag, ảnh BGR 640×640 để hiển thị).

        Args:
            frame_bgr: Frame đầu vào dạng BGR.
            on_fall: Callback được gọi khi phát hiện ngã.

        Returns:
            Tuple (FrameDiag, display_image).
        """
        # Giai đoạn 1: Tiền xử lý
        try:
            small, _, _ = preprocess_frame(frame_bgr, self.config.input_size)
        except Exception:
            # Nếu resize thất bại, trả về frame gốc đã resize cơ bản
            small = cv2.resize(frame_bgr, self.config.input_size, interpolation=cv2.INTER_LINEAR)

        # Giai đoạn 2: Trích xuất pose
        try:
            pose = self.pose.extract(small)
        except Exception:
            pose = None

        # Xử lý khi không phát hiện người
        if pose is None:
            diag = FrameDiag(
                mean_kpt_conf=None,
                torso_deg=None,
                nose_ankle_deg=None,
                posture=Posture.NORMAL.value,
                fall_confirmed=False,
            )
            self.temporal.update(Posture.NORMAL)
            display = small.copy()
            _annotate_status(display, diag, extra="khong_phat_hien_nguoi")
            return diag, display

        # Giai đoạn 3: Phân tích động học
        try:
            torso_deg, na_deg = compute_pose_angles(pose.keypoints_norm)
            posture = classify_posture(torso_deg, na_deg, self.config)
            fall_now = self.temporal.update(posture)
        except Exception:
            # Nếu tính góc thất bại, coi như tư thế bình thường
            torso_deg, na_deg = None, None
            posture = Posture.NORMAL
            fall_now = False

        diag = FrameDiag(
            mean_kpt_conf=pose.mean_confidence,
            torso_deg=torso_deg,
            nose_ankle_deg=na_deg,
            posture=posture.value,
            fall_confirmed=fall_now,
        )

        try:
            display = draw_pose_overlay(small, pose)
        except Exception:
            display = small.copy()

        _annotate_status(display, diag)

        # Giai đoạn 4: Cảnh báo
        if fall_now:
            if on_fall is not None:
                try:
                    on_fall(display.copy(), diag)
                except Exception:
                    pass  # Bỏ qua lỗi callback
            elif self.alerter.enabled():
                try:
                    self.alerter.send_fall_alert(
                        display,
                        extra_text=(
                            f"conf_tb={diag.mean_kpt_conf:.3f}, "
                            f"goc_than={diag.torso_deg}, "
                            f"goc_mui_co-chan={diag.nose_ankle_deg}"
                        ),
                    )
                except Exception:
                    pass  # Bỏ qua lỗi Telegram
            self.temporal.acknowledge_fall()

        return diag, display

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        on_fall: Callable[[np.ndarray, FrameDiag], None] | None = None,
    ) -> FrameDiag:
        """
        Xử lý một khung BGR; `on_fall` khi vừa xác nhận ngã.

        Args:
            frame_bgr: Frame đầu vào dạng BGR.
            on_fall: Callback được gọi khi phát hiện ngã.

        Returns:
            FrameDiag chứa thông tin chẩn đoán.
        """
        diag, _ = self.process_frame_with_display(frame_bgr, on_fall=on_fall)
        return diag


def run_on_video(
    source: str | int,
    config: PipelineConfig | None = None,
    show: bool = True,
    mirror: bool = False,
) -> None:
    """
    Vòng lặp demo: mở video hoặc chỉ số camera.

    Args:
        source: Đường dẫn video hoặc chỉ số camera (0 cho webcam).
        config: Cấu hình pipeline.
        show: Hiển thị cửa sổ OpenCV.
        mirror: Lật ngang frame (cho selfie camera).
    """
    cfg = config or PipelineConfig()
    pipe = HybridFallPipeline(cfg)

    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn: {source}")
    except Exception as e:
        raise RuntimeError(f"Lỗi khi mở nguồn video: {e}") from e

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))

    alert = TelegramAlerter(cfg)

    def on_fall(vis: np.ndarray, diag: FrameDiag) -> None:
        if alert.enabled():
            try:
                alert.send_fall_alert(
                    vis,
                    extra_text=(
                        f"conf_tb={diag.mean_kpt_conf}, "
                        f"goc_than={diag.torso_deg}, goc_mui_co-chan={diag.nose_ankle_deg}"
                    ),
                )
            except Exception:
                pass
        print(f"[CANH_BAO_NGÃ] {diag}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            try:
                diag, display = pipe.process_frame_with_display(frame, on_fall=on_fall)
            except Exception:
                continue

            if show:
                cv2.imshow("Phat hien nga - Hybrid Fall Detection (640)", display)
                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
