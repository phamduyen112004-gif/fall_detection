"""Vẽ bounding box + khung xương lên snapshot cảnh báo."""

from __future__ import annotations

import cv2
import numpy as np

from .stage2_pose import PoseFrame


COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_pose_overlay(frame_bgr: np.ndarray, pose: PoseFrame) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    k = pose.keypoints_norm
    pts = np.zeros((17, 2), dtype=np.int32)
    for i in range(17):
        pts[i, 0] = int(np.clip(k[i, 0] * w, 0, w - 1))
        pts[i, 1] = int(np.clip(k[i, 1] * h, 0, h - 1))

    for a, b in COCO_EDGES:
        if float(k[a, 2]) < 0.1 or float(k[b, 2]) < 0.1:
            continue
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 180), 2, cv2.LINE_AA)

    for i in range(17):
        if float(k[i, 2]) < 0.1:
            continue
        cv2.circle(out, tuple(pts[i]), 4, (0, 128, 255), -1, cv2.LINE_AA)

    if pose.boxes_xyxy is not None and len(pose.boxes_xyxy) > 0:
        x1, y1, x2, y2 = pose.boxes_xyxy[0].astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)

    return out
