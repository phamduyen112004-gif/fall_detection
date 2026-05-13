"""Các dataclass dùng chung trong pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameDiag:
    """
    Thông tin chẩn đoán cho mỗi frame.

    Thuộc tính:
        mean_kpt_conf: Độ tin cậy trung bình của các keypoints (0-1).
        torso_deg: Góc thân (torso) so với trục dọc, tính bằng độ.
        nose_ankle_deg: Góc mũi-mắt cá chân so với trục dọc, tính bằng độ.
        posture: Tư thế hiện tại ('normal' hoặc 'laydown').
        fall_confirmed: True nếu sự kiện ngã đã được xác nhận.
        fall_prob: Xác suất ngã từ Transformer (None nếu dùng heuristic).
    """
    mean_kpt_conf: float | None
    torso_deg: float | None
    nose_ankle_deg: float | None
    posture: str
    fall_confirmed: bool
    fall_prob: float | None = None

    def __str__(self) -> str:
        conf_str = f"{self.mean_kpt_conf:.3f}" if self.mean_kpt_conf is not None else "—"
        torso_str = f"{self.torso_deg:.1f}°" if self.torso_deg is not None else "—"
        ankle_str = f"{self.nose_ankle_deg:.1f}°" if self.nose_ankle_deg is not None else "—"
        prob_str = f" p={self.fall_prob:.3f}" if self.fall_prob is not None else ""
        return (
            f"FrameDiag(posture={self.posture}, conf={conf_str}, "
            f"torso={torso_str}, nose_ankle={ankle_str}, "
            f"fall={self.fall_confirmed}{prob_str})"
        )
