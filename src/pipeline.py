"""Orchestrator: 4 giai đoạn Hybrid Fall Detection."""

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


@dataclass
class FrameDiag:
    mean_kpt_conf: float | None
    torso_deg: float | None
    nose_ankle_deg: float | None
    posture: str
    fall_confirmed: bool


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
        Xử lý một khung BGR; trả về (diag, ảnh BGR 640×640 để hiển thị GUI/OpenCV).
        """
        small, _, _ = preprocess_frame(frame_bgr, self.config.input_size)
        pose = self.pose.extract(small)

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
            _annotate_status(display, diag, extra="khong_pose")
            return diag, display

        torso_deg, na_deg = compute_pose_angles(pose.keypoints_norm)
        posture = classify_posture(torso_deg, na_deg, self.config)
        fall_now = self.temporal.update(posture)

        diag = FrameDiag(
            mean_kpt_conf=pose.mean_confidence,
            torso_deg=torso_deg,
            nose_ankle_deg=na_deg,
            posture=posture.value,
            fall_confirmed=fall_now,
        )

        display = draw_pose_overlay(small, pose)
        _annotate_status(display, diag)

        if fall_now:
            if on_fall is not None:
                on_fall(display.copy(), diag)
            elif self.alerter.enabled():
                self.alerter.send_fall_alert(
                    display,
                    extra_text=(
                        f"conf_tb={diag.mean_kpt_conf:.3f}, "
                        f"góc_thân={diag.torso_deg}, "
                        f"góc_mũi_cổ_chân={diag.nose_ankle_deg}"
                    ),
                )
            self.temporal.acknowledge_fall()

        return diag, display

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        on_fall: Callable[[np.ndarray, FrameDiag], None] | None = None,
    ) -> FrameDiag:
        """Xử lý một khung BGR; `on_fall` khi vừa xác nhận ngã (ảnh đã vẽ pose + box)."""
        diag, _ = self.process_frame_with_display(frame_bgr, on_fall=on_fall)
        return diag


def run_on_video(
    source: str | int,
    config: PipelineConfig | None = None,
    show: bool = True,
    mirror: bool = False,
) -> None:
    """Vòng lặp demo: mở video hoặc chỉ số camera."""
    cfg = config or PipelineConfig()
    pipe = HybridFallPipeline(cfg)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(1, int(1000 / fps))

    alert = TelegramAlerter(cfg)

    def on_fall(vis: np.ndarray, diag: FrameDiag) -> None:
        if alert.enabled():
            alert.send_fall_alert(
                vis,
                extra_text=(
                    f"conf_tb={diag.mean_kpt_conf}, "
                    f"góc_thân={diag.torso_deg}, góc_mũi_cổ_chân={diag.nose_ankle_deg}"
                ),
            )
        print("[FALL]", diag)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mirror:
            frame = cv2.flip(frame, 1)
        diag, display = pipe.process_frame_with_display(frame, on_fall=on_fall)

        if show:
            cv2.imshow("Hybrid Fall Detection (640)", display)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()
