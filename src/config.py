"""Tham số pipeline — chỉnh theo môi trường triển khai."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class PipelineConfig:
    # Giai đoạn 1
    input_size: tuple[int, int] = (640, 640)

    # Giai đoạn 2
    pose_model: str = "yolo11n-pose.pt"
    min_mean_keypoint_conf: float = 0.2

    # Giai đoạn 3 — góc (độ) so với trục dọc màn hình
    laydown_torso_angle_deg: float = 55.0
    laydown_nose_ankle_angle_deg: float = 50.0
    # Duy trì tư thế nằm ngang trước khi xác nhận ngã
    fall_min_frames: int = 60
    fall_min_seconds: float | None = 10.0  # None = chỉ dùng frame

    # Giai đoạn 4
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "telegram_bot_token",
            self.telegram_bot_token or os.environ.get("TELEGRAM_BOT_TOKEN"),
        )
        object.__setattr__(
            self,
            "telegram_chat_id",
            self.telegram_chat_id or os.environ.get("TELEGRAM_CHAT_ID"),
        )


DEFAULT_CONFIG = PipelineConfig()
