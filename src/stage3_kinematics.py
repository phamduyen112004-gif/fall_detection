"""Giai đoạn 3: Góc hình học, phân loại tư thế, bộ lọc thời gian ngắn."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time

import numpy as np

from .config import PipelineConfig
from .stage2_pose import (
    L_ANKLE,
    L_HIP,
    L_SHOULDER,
    NOSE,
    R_ANKLE,
    R_HIP,
    R_SHOULDER,
)


class Posture(str, Enum):
    NORMAL = "normal"  # đứng / ngồi
    LAYDOWN = "laydown"


def _angle_with_vertical_deg(vx: float, vy: float) -> float:
    """Góc (0–90°) giữa vector và trục dọc màn hình; trục dọc hướng lên = (0, -1)."""
    n = math.hypot(vx, vy)
    if n < 1e-6:
        return 0.0
    vx /= n
    vy /= n
    # Trục dọc lên
    ux, uy = 0.0, -1.0
    c = max(-1.0, min(1.0, vx * ux + vy * uy))
    rad = math.acos(c)
    deg = math.degrees(rad)
    # Góc nhỏ với trục dọc: đứng ~0°, nằm ~90°
    if deg > 90.0:
        deg = 180.0 - deg
    return deg


def _mid(
    k: np.ndarray,
    a: int,
    b: int,
    min_conf: float = 0.15,
) -> tuple[float, float] | None:
    """Trung điểm 2 keypoint (tọa độ đã chuẩn hóa 0–1)."""
    ca, cb = float(k[a, 2]), float(k[b, 2])
    if ca < min_conf and cb < min_conf:
        return None
    if ca < min_conf:
        return float(k[b, 0]), float(k[b, 1])
    if cb < min_conf:
        return float(k[a, 0]), float(k[a, 1])
    return (float(k[a, 0] + k[b, 0]) * 0.5, float(k[a, 1] + k[b, 1]) * 0.5)


def compute_pose_angles(k: np.ndarray) -> tuple[float | None, float | None]:
    """
    Torso: từ mid-hip → mid-shoulder so với trục dọc.
    Nose–ankle: vector nose → mid-ankle so với trục dọc.
    """
    mid_hip = _mid(k, L_HIP, R_HIP)
    mid_sh = _mid(k, L_SHOULDER, R_SHOULDER)
    torso_deg: float | None = None
    if mid_hip and mid_sh:
        vx = mid_sh[0] - mid_hip[0]
        vy = mid_sh[1] - mid_hip[1]
        torso_deg = _angle_with_vertical_deg(vx, vy)

    na = _mid(k, L_ANKLE, R_ANKLE)
    nose_ankle_deg: float | None = None
    if na and float(k[NOSE, 2]) > 0.1:
        vx = float(k[NOSE, 0]) - na[0]
        vy = float(k[NOSE, 1]) - na[1]
        nose_ankle_deg = _angle_with_vertical_deg(vx, vy)

    return torso_deg, nose_ankle_deg


def classify_posture(
    torso_deg: float | None,
    nose_ankle_deg: float | None,
    cfg: PipelineConfig,
) -> Posture:
    """If–else: nằm ngang nếu ít nhất một góc vượt ngưỡng (có dữ liệu)."""
    t_ok = torso_deg is not None and torso_deg >= cfg.laydown_torso_angle_deg
    n_ok = (
        nose_ankle_deg is not None
        and nose_ankle_deg >= cfg.laydown_nose_ankle_angle_deg
    )
    if t_ok or n_ok:
        return Posture.LAYDOWN
    return Posture.NORMAL


@dataclass
class TemporalState:
    laydown_frames: int = 0
    laydown_start_mono: float | None = None
    fall_announced: bool = False

    def reset_laydown(self) -> None:
        self.laydown_frames = 0
        self.laydown_start_mono = None


class FallTemporalFilter:
    """Ngưỡng thời gian / số frame (OR) để giảm báo động giả."""

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.state = TemporalState()

    def update(self, posture: Posture, dt: float | None = None) -> bool:
        """
        Trả về True một lần khi xác nhận ngã.
        dt: dự phòng (FPS thấp có thể mở rộng logic theo thời gian thực).
        """
        if posture != Posture.LAYDOWN:
            self.state.reset_laydown()
            self.state.fall_announced = False
            return False

        self.state.laydown_frames += 1
        now = time.monotonic()
        if self.state.laydown_start_mono is None:
            self.state.laydown_start_mono = now

        elapsed = now - self.state.laydown_start_mono
        frame_ok = self.state.laydown_frames >= self.cfg.fall_min_frames
        time_ok = (
            self.cfg.fall_min_seconds is not None
            and elapsed >= self.cfg.fall_min_seconds
        )

        if frame_ok or time_ok:
            if not self.state.fall_announced:
                self.state.fall_announced = True
                return True
        return False

    def acknowledge_fall(self) -> None:
        """Giữ trạng thái đã báo; reset khi bệnh nhân trở lại tư thế bình thường (trong update)."""
        self.state.reset_laydown()
