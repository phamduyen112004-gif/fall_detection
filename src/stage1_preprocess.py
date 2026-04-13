"""Giai đoạn 1: Thu nhận & tiền xử lý — resize 640×640 cho YOLOv11."""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(
    frame_bgr: np.ndarray,
    size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, float]:
    """
    Trả về:
        resized: khung BGR 640×640
        sx, sy: tỉ lệ scale so với khung gốc (dùng khi cần map lại tọa độ)
    """
    h, w = frame_bgr.shape[:2]
    tw, th = size
    sx = tw / float(w)
    sy = th / float(h)
    resized = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized, sx, sy
