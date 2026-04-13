"""Giai đoạn 2: YOLOv11n-pose — 17 keypoint COCO, chuẩn hóa, lọc confidence."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ultralytics import YOLO

from .config import PipelineConfig


# COCO pose (17 điểm)
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16


@dataclass
class PoseFrame:
    """keypoints_norm: (17, 3) — x,y ∈ [0,1], conf gốc [0,1]."""

    keypoints_norm: np.ndarray  # (17, 3)
    mean_confidence: float
    boxes_xyxy: np.ndarray | None  # pixel trên ảnh 640 (nếu có)
    model_img_shape: tuple[int, int]


class PoseExtractor:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        model: YOLO | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._model = model if model is not None else YOLO(self.config.pose_model)

    def extract(self, frame_bgr_640: np.ndarray) -> PoseFrame | None:
        h, w = frame_bgr_640.shape[:2]
        results = self._model.predict(
            frame_bgr_640,
            imgsz=self.config.input_size[0],
            verbose=False,
        )
        if not results:
            return None
        r0 = results[0]
        if r0.keypoints is None or r0.keypoints.data is None:
            return None
        # data: (num_person, 17, 3) — x,y pixel, conf
        kall = r0.keypoints.data.cpu().numpy()
        if kall.size == 0:
            return None
        # Chọn người có độ tin cậy trung bình keypoint cao nhất
        best_i = int(np.argmax([k[:, 2].mean() for k in kall]))
        k = kall[best_i].astype(np.float32)

        mean_conf = float(k[:, 2].mean())
        if mean_conf < self.config.min_mean_keypoint_conf:
            return None

        kn = k.copy()
        kn[:, 0] /= float(w)
        kn[:, 1] /= float(h)
        # conf giữ nguyên

        boxes = None
        if r0.boxes is not None and len(r0.boxes) > best_i:
            boxes = r0.boxes.xyxy[best_i : best_i + 1].cpu().numpy()

        return PoseFrame(
            keypoints_norm=kn,
            mean_confidence=mean_conf,
            boxes_xyxy=boxes,
            model_img_shape=(h, w),
        )
