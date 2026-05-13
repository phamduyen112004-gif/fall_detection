"""Giai đoạn 1: Thu nhận & tiền xử lý — resize 640×640 cho YOLOv11.

Module tiền xử lý frame trước khi đưa vào YOLOv11-Pose:
- Resize về kích thước input chuẩn (mặc định 640x640)
- Tính tỷ lệ scale để map tọa độ ngược lại nếu cần
"""

from __future__ import annotations

import cv2
import numpy as np


def preprocess_frame(
    frame_bgr: np.ndarray,
    size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, float]:
    """
    Tiền xử lý một frame: resize về kích thước chuẩn.

    Args:
        frame_bgr: Frame đầu vào dạng BGR.
        size: Kích thước resize mong muốn (width, height).

    Returns:
        Tuple gồm:
            - resized: Frame BGR đã resize
            - sx: Tỷ lệ scale theo chiều ngang (width_mới / width_gốc)
            - sy: Tỷ lệ scale theo chiều dọc (height_mới / height_gốc)

    Ví dụ:
        >>> frame = cv2.imread("test.jpg")
        >>> resized, sx, sy = preprocess_frame(frame)
        >>> print(f"Scale: sx={sx:.2f}, sy={sy:.2f}")
    """
    h, w = frame_bgr.shape[:2]
    tw, th = size
    sx = tw / float(w)
    sy = th / float(h)
    resized = cv2.resize(frame_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    return resized, sx, sy
