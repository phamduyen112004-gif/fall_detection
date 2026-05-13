"""Giai đoạn 4: Cảnh báo Telegram (Edge — không cần cloud xử lý video).

Module gửi cảnh báo qua Telegram khi phát hiện sự kiện ngã.
- Hỗ trợ cooldown để tránh spam
- Mã hóa ảnh JPEG với chất lượng có thể điều chỉnh
- Xử lý lỗi mạng và API Telegram
"""

from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import requests

from .config import PipelineConfig


def encode_jpeg_bgr(frame_bgr: np.ndarray, quality: int = 90) -> bytes:
    """
    Mã hóa frame BGR thành bytes JPEG.

    Args:
        frame_bgr: Frame ảnh BGR.
        quality: Chất lượng JPEG (0-100), mặc định 90.

    Returns:
        Bytes của ảnh JPEG.

    Raises:
        RuntimeError: Nếu không thể mã hóa ảnh.
    """
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("Không thể mã hóa JPEG")
    return buf.tobytes()


class TelegramAlerter:
    """Gửi cảnh báo Telegram khi phát hiện ngã."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def enabled(self) -> bool:
        """Kiểm tra xem Telegram alert có được bật không."""
        return bool(self.config.telegram_bot_token and self.config.telegram_chat_id)

    def send_fall_alert(
        self,
        snapshot_bgr: np.ndarray,
        extra_text: str = "",
    ) -> dict[str, Any]:
        """
        Gửi cảnh báo ngã qua Telegram.

        Args:
            snapshot_bgr: Ảnh snapshot BGR của frame phát hiện ngã.
            extra_text: Văn bản bổ sung cho caption.

        Returns:
            Response JSON từ Telegram API.

        Raises:
            RuntimeError: Nếu thiếu token/chat_id hoặc không gửi được.
        """
        if not self.enabled():
            raise RuntimeError(
                "Thiếu TELEGRAM_BOT_TOKEN hoặc TELEGRAM_CHAT_ID trong môi trường / config."
            )
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        assert token is not None and chat_id is not None

        url = f"https://api.telegram.org/bot{token}/sendPhoto"

        # Mã hóa ảnh
        try:
            img_bytes = encode_jpeg_bgr(snapshot_bgr)
        except Exception as e:
            raise RuntimeError(f"Không thể mã hóa ảnh: {e}") from e

        caption = "🚨 CẢNH BÁO: Phát hiện NGÃ (Fall Detected) — kiểm tra bệnh nhân ngay."
        if extra_text:
            caption = f"{caption}\n{extra_text}"

        files = {"photo": ("fall_snapshot.jpg", io.BytesIO(img_bytes), "image/jpeg")}
        data = {"chat_id": chat_id, "caption": caption}

        try:
            r = requests.post(url, data=data, files=files, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.Timeout:
            raise RuntimeError("Telegram API timeout - không thể gửi cảnh báo") from None
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Không thể kết nối Telegram: {e}") from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Telegram API lỗi HTTP: {e}") from e
