"""Giai đoạn 4: Cảnh báo Telegram (Edge — không cần cloud xử lý video)."""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import requests

from .config import PipelineConfig


def encode_jpeg_bgr(frame_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("Không thể mã hóa JPEG")
    return buf.tobytes()


class TelegramAlerter:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def enabled(self) -> bool:
        return bool(self.config.telegram_bot_token and self.config.telegram_chat_id)

    def send_fall_alert(
        self,
        snapshot_bgr: np.ndarray,
        extra_text: str = "",
    ) -> dict[str, Any]:
        if not self.enabled():
            raise RuntimeError(
                "Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong môi trường / config."
            )
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        assert token is not None and chat_id is not None

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        img_bytes = encode_jpeg_bgr(snapshot_bgr)
        caption = "🚨 CẢNH BÁO: Phát hiện NGÃ (Fall Detected) — kiểm tra bệnh nhân ngay."
        if extra_text:
            caption = f"{caption}\n{extra_text}"

        files = {"photo": ("fall_snapshot.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(url, data=data, files=files, timeout=60)
        r.raise_for_status()
        return r.json()
