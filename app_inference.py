#!/usr/bin/env python3
"""PyQt5 real-time fall detection app.

Pipeline:
- Video source (webcam / local video / RTSP-HTTP stream)
- YOLOv11-Pose keypoint extraction
- 60-D frame feature vector (51 flattened keypoints + 9 PIFR features)
- Sliding-window inference with stride
- Hybrid heuristic filtering to suppress false alarms
- Async Telegram alerting with cooldown
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import requests
import torch
from PyQt5.QtCore import QObject, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.hybrid_fall_transformer import HybridFallTransformer
from src.pifr_features import (
    EPS,
    FEATURE_DIM,
    IMGSZ,
    MIN_MEAN_CONF,
    SEQ_LEN,
    frame_to_vector_60,
    resample_to_length,
)
from src.types import FrameDiag
from src.viz import COCO_EDGES
from ultralytics import YOLO


class PerformanceProfiler:
    """Collect per-stage latency and model complexity metrics."""

    def __init__(self) -> None:
        self.read_ms = 0.0
        self.preprocess_ms = 0.0
        self.pose_ms = 0.0
        self.transformer_ms = 0.0
        self.post_ms = 0.0
        self.total_ms = 0.0
        self.frames = 0
        self.pose_params_m = 0.0
        self.pose_gflops = 0.0
        self.transformer_params_m = 0.0
        self.transformer_gflops = 0.0
        self.pose_model_mb = 0.0
        self.transformer_model_mb = 0.0

    @staticmethod
    def _safe_model_info(model: torch.nn.Module, imgsz: int = IMGSZ) -> tuple[float, float]:
        params_m = 0.0
        gflops = 0.0
        try:
            info = model.info(verbose=False, imgsz=imgsz)
            if isinstance(info, dict):
                params_m = float(info.get("params", 0.0)) / 1e6
                gflops = float(info.get("GFLOPs", info.get("gflops", 0.0)))
            elif isinstance(info, (tuple, list)) and len(info) >= 2:
                params_m = float(info[0]) / 1e6
                gflops = float(info[1])
        except Exception:
            pass
        return params_m, gflops

    @staticmethod
    def _file_size_mb(path: str | Path) -> float:
        try:
            return Path(path).stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def capture_model_metrics(self, pose_model: YOLO, transformer: torch.nn.Module, pose_weights: str | Path, transformer_weights: str | Path) -> None:
        self.pose_params_m, self.pose_gflops = self._safe_model_info(pose_model)
        self.transformer_params_m, self.transformer_gflops = self._safe_model_info(transformer)
        self.pose_model_mb = self._file_size_mb(pose_weights)
        self.transformer_model_mb = self._file_size_mb(transformer_weights)

    def add_frame_times(
        self,
        read_ms: float,
        preprocess_ms: float,
        pose_ms: float,
        transformer_ms: float,
        post_ms: float,
    ) -> None:
        self.read_ms += read_ms
        self.preprocess_ms += preprocess_ms
        self.pose_ms += pose_ms
        self.transformer_ms += transformer_ms
        self.post_ms += post_ms
        self.total_ms += read_ms + preprocess_ms + pose_ms + transformer_ms + post_ms
        self.frames += 1

    def summary(self) -> dict[str, float]:
        n = max(self.frames, 1)
        total = max(self.total_ms, 1e-6)
        return {
            "frames": float(self.frames),
            "read_ms": self.read_ms / n,
            "preprocess_ms": self.preprocess_ms / n,
            "pose_ms": self.pose_ms / n,
            "transformer_ms": self.transformer_ms / n,
            "post_ms": self.post_ms / n,
            "total_ms": self.total_ms / n,
            "fps": 1000.0 / (total / n),
            "pose_params_m": self.pose_params_m,
            "pose_gflops": self.pose_gflops,
            "transformer_params_m": self.transformer_params_m,
            "transformer_gflops": self.transformer_gflops,
            "pose_model_mb": self.pose_model_mb,
            "transformer_model_mb": self.transformer_model_mb,
        }

# ---------- Runtime tuning ----------
MAX_MISSING_FRAMES = 15
INFER_STRIDE = 15
MIN_VALID_FRAMES_FOR_INFER = 8

MIN_PERSON_AREA_RATIO = 0.02
MIN_PERSON_HEIGHT_RATIO = 0.18
MIN_KEYPOINTS_CONFIDENT = 7
MAX_TRACK_CENTER_JUMP_RATIO = 0.20
TILT_LOW_THRESHOLD = 15.0
TILT_MEDIUM_THRESHOLD = 35.0
TILT_HIGH_THRESHOLD = 55.0
FALL_MIN_BOTTOM_RATIO = 0.65
SOFA_SIT_BOTTOM_RATIO_MAX = 0.83
DROP_LOOKBACK_FRAMES = 4
MIN_DROP_DELTA_CENTER_Y = 0.06
LOW_CONF_THRESHOLD = 0.10
HIGH_CONF_THRESHOLD = 0.25
ALERT_MIN_PROB = 0.18
ALERT_SUSTAIN_SEC = 0.0
ALERT_COOLDOWN_SEC = 10.0

STREAM_RECONNECT_DELAY_SEC = 2.5
STREAM_RECONNECT_BACKOFF_MAX = 8.0


class TelegramNotifier:
    """Send Telegram alerts asynchronously with cooldown protection."""

    def __init__(self, cooldown_sec: float = ALERT_COOLDOWN_SEC) -> None:
        self.cooldown_sec = cooldown_sec
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="telegram")
        self._pending: Future | None = None
        self._last_send_mono: float | None = None
        self.last_send_elapsed_ms: float | None = None

    def enabled(self) -> bool:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        return bool(token and chat_id)

    def _send_photo(self, image_path: Path, caption: str) -> bool:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return False

        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            t0 = time.perf_counter()
            with image_path.open("rb") as fp:
                files = {"photo": fp}
                data = {"chat_id": chat_id, "caption": caption[:1024]}
                resp = requests.post(url, files=files, data=data, timeout=30)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.last_send_elapsed_ms = elapsed_ms
            print(f"[telegram] send_photo={elapsed_ms:.2f}ms ok={resp.ok}")
            return resp.ok
        except (OSError, requests.RequestException):
            self.last_send_elapsed_ms = None
            return False

    def maybe_notify_async(self, image_path: Path, caption: str = "Fall detected") -> bool:
        now = time.monotonic()
        with self._lock:
            if self._last_send_mono is not None and now - self._last_send_mono < self.cooldown_sec:
                return False
            if self._pending is not None and not self._pending.done():
                return False
            self._last_send_mono = now
            self._pending = self._executor.submit(self._send_photo, image_path, caption)
            return True

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


def load_dotenv_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without overriding existing env."""
    if not env_path.is_file():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            os.environ[key] = value.strip().strip('"').strip("'")
    except OSError:
        pass


def _is_network_source(source: str | int) -> bool:
    return isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://"))


def _is_file_source(source: str | int) -> bool:
    return isinstance(source, str) and not _is_network_source(source)


def _is_valid_person_detection(
    keypoints_xyc: np.ndarray,
    box_xyxy: np.ndarray | None,
    frame_w: int,
    frame_h: int,
) -> bool:
    if keypoints_xyc.shape != (17, 3):
        return False

    conf = keypoints_xyc[:, 2]
    if int(np.sum(conf > 0.30)) < MIN_KEYPOINTS_CONFIDENT:
        return False

    if box_xyxy is None or len(box_xyxy.flatten()) < 4:
        return False

    x1, y1, x2, y2 = [float(x) for x in box_xyxy.flatten()[:4]]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 1.0 or bh < 1.0:
        return False

    area_ratio = (bw * bh) / float(max(frame_w * frame_h, 1))
    height_ratio = bh / float(max(frame_h, 1))
    if area_ratio < MIN_PERSON_AREA_RATIO or height_ratio < MIN_PERSON_HEIGHT_RATIO:
        return False

    return True


def _bbox_center_and_size(box_xyxy: np.ndarray | None) -> tuple[float, float, float, float] | None:
    if box_xyxy is None or len(box_xyxy.flatten()) < 4:
        return None
    x1, y1, x2, y2 = [float(x) for x in box_xyxy.flatten()[:4]]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    if bw < 1.0 or bh < 1.0:
        return None
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2), bw, bh


def _is_stable_track(
    curr_box_xyxy: np.ndarray | None,
    prev_box_xyxy: np.ndarray | None,
    frame_w: int,
    frame_h: int,
) -> bool:
    if prev_box_xyxy is None:
        return True
    curr = _bbox_center_and_size(curr_box_xyxy)
    prev = _bbox_center_and_size(prev_box_xyxy)
    if curr is None or prev is None:
        return False
    c_x, c_y, _, _ = curr
    p_x, p_y, _, _ = prev
    norm = float(max(min(frame_w, frame_h), 1))
    center_jump = float(np.hypot(c_x - p_x, c_y - p_y)) / norm
    return center_jump <= MAX_TRACK_CENTER_JUMP_RATIO


def _box_posture_features(
    box_xyxy: np.ndarray | None,
    frame_h: int,
) -> tuple[float, float, float, float] | None:
    stats = _bbox_center_and_size(box_xyxy)
    if stats is None:
        return None
    _, cy, bw, bh = stats
    x1, y1, x2, y2 = [float(x) for x in box_xyxy.flatten()[:4]]
    center_y_ratio = cy / float(max(frame_h, 1))
    height_ratio = bh / float(max(frame_h, 1))
    aspect_ratio = bw / max(bh, EPS)  # Width / Height
    bottom_ratio = y2 / float(max(frame_h, 1))
    return center_y_ratio, height_ratio, aspect_ratio, bottom_ratio


def _torso_angle_deg(keypoints_xy: np.ndarray | None) -> float | None:
    if keypoints_xy is None or keypoints_xy.shape[0] < 12:
        return None
    left_shoulder = keypoints_xy[5]
    right_shoulder = keypoints_xy[6]
    left_hip = keypoints_xy[11]
    right_hip = keypoints_xy[12]
    shoulder = 0.5 * (left_shoulder + right_shoulder)
    hip = 0.5 * (left_hip + right_hip)
    vec = shoulder - hip
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return None
    cos_theta = abs(vec[1]) / norm
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def _confidence_grade(prob: float) -> str:
    if prob >= HIGH_CONF_THRESHOLD:
        return "high"
    if prob >= LOW_CONF_THRESHOLD:
        return "medium"
    return "low"


class InferenceWorker(QThread):
    """Read frames, infer fall probability, and emit display frames."""

    frame_ready = pyqtSignal(QImage)
    log_message = pyqtSignal(str)
    fall_detected = pyqtSignal(float)

    def __init__(
        self,
        source: str | int,
        weights_pose: str,
        weights_cls: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.source = source
        self.weights_pose = weights_pose
        self.weights_cls = weights_cls
        self._running = True
        self.profiler = PerformanceProfiler()

    def stop(self) -> None:
        self._running = False

    @staticmethod
    def _open_capture(source: str | int) -> cv2.VideoCapture:
        if isinstance(source, int):
            return cv2.VideoCapture(source)
        return cv2.VideoCapture(str(source))

    def _draw_alert_frame(
        self,
        frame_bgr: np.ndarray,
        keypoints_xy: np.ndarray | None,
        keypoints_conf: np.ndarray | None,
        box_xyxy: np.ndarray | None,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        if keypoints_xy is not None and keypoints_conf is not None:
            pts = np.zeros((17, 2), dtype=np.int32)
            for i in range(17):
                pts[i, 0] = int(np.clip(keypoints_xy[i, 0], 0, w - 1))
                pts[i, 1] = int(np.clip(keypoints_xy[i, 1], 0, h - 1))
            for a, b in COCO_EDGES:
                if float(keypoints_conf[a]) < 0.1 or float(keypoints_conf[b]) < 0.1:
                    continue
                cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 180), 2, cv2.LINE_AA)
            for i in range(17):
                if float(keypoints_conf[i]) < 0.1:
                    continue
                cv2.circle(out, tuple(pts[i]), 4, (0, 128, 255), -1, cv2.LINE_AA)
        if box_xyxy is not None and len(box_xyxy) >= 4:
            x1, y1, x2, y2 = [int(x) for x in box_xyxy.flatten()[:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_AA)
        cv2.putText(
            out,
            "FALL ALERT",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return out

    def run(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose = YOLO(self.weights_pose)
        model = HybridFallTransformer().to(device)
        self.profiler.capture_model_metrics(pose, model, self.weights_pose, self.weights_cls)
        try:
            ckpt = torch.load(self.weights_cls, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(self.weights_cls, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        model.eval()
        infer_threshold = float(ckpt.get("best_threshold", ALERT_MIN_PROB))
        self.log_message.emit(f"Ngưỡng phát hiện fall={infer_threshold:.4f}")

        cap = self._open_capture(self.source)
        if not cap.isOpened():
            self.log_message.emit(f"Không mở được nguồn: {self.source}")
            return

        is_network_stream = _is_network_source(self.source)
        is_file_source = _is_file_source(self.source)

        self.log_message.emit(
            f"Source={'video-file' if is_file_source else 'live-stream'} | "
            f"stride={INFER_STRIDE}"
        )

        feat_buffer: deque[np.ndarray] = deque(maxlen=SEQ_LEN)
        y_buffer: deque[float] = deque(maxlen=SEQ_LEN)
        notifier = TelegramNotifier(cooldown_sec=ALERT_COOLDOWN_SEC)
        alert_path = _ROOT / "alert.jpg"

        frame_i = 0
        missing_frames = 0
        prev_valid_box: np.ndarray | None = None
        reconnect_tries = 0
        reconnect_at: float | None = None

        try:
            while self._running:
                t_loop_start = time.perf_counter()
                try:
                    t_read_start = time.perf_counter()
                    ok, frame_bgr = cap.read()
                    read_ms = (time.perf_counter() - t_read_start) * 1000.0
                except Exception:  # noqa: BLE001
                    ok, frame_bgr = False, None
                    read_ms = 0.0

                if not ok:
                    if is_network_stream and self._running:
                        reconnect_tries += 1
                        backoff = min(
                            STREAM_RECONNECT_DELAY_SEC * (1.25 ** max(reconnect_tries - 1, 0)),
                            STREAM_RECONNECT_BACKOFF_MAX,
                        )
                        reconnect_at = time.monotonic() + backoff
                        self.log_message.emit(
                            f"Mất luồng camera, thử kết nối lại (lần {reconnect_tries}, chờ {backoff:.1f}s)..."
                        )
                        cap.release()
                        while self._running and time.monotonic() < reconnect_at:
                            time.sleep(0.05)
                        cap = self._open_capture(self.source)
                        if cap.isOpened():
                            self.log_message.emit("Kết nối lại camera thành công.")
                            reconnect_tries = 0
                            continue
                        continue
                    break

                reconnect_tries = 0
                t_pre_start = time.perf_counter()
                frame_bgr = cv2.resize(frame_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
                h, w = frame_bgr.shape[:2]
                pre_ms = (time.perf_counter() - t_pre_start) * 1000.0

                t_pose_start = time.perf_counter()
                results = pose.predict(frame_bgr, imgsz=IMGSZ, verbose=False)
                pose_ms = (time.perf_counter() - t_pose_start) * 1000.0

                display = frame_bgr.copy()
                k_xy_draw: np.ndarray | None = None
                k_c_draw: np.ndarray | None = None
                box_draw: np.ndarray | None = None
                vec: np.ndarray | None = None
                current_mean_conf = 0.0
                current_confident_kpts = 0
                prev_box_for_stability = prev_valid_box.copy() if prev_valid_box is not None else None

                if results and results[0].keypoints is not None and results[0].keypoints.data is not None:
                    r0 = results[0]
                    kall = r0.keypoints.data.cpu().numpy()
                    if kall.size > 0:
                        best_i = -1
                        best_score = -1.0
                        best_box_xyxy: np.ndarray | None = None
                        for i, kp in enumerate(kall):
                            kpi = kp.astype(np.float32)
                            if r0.boxes is not None and len(r0.boxes) > i:
                                box_i = r0.boxes.xyxy[i].cpu().numpy()
                            else:
                                box_i = None
                            if not _is_valid_person_detection(kpi, box_i, w, h):
                                continue
                            score = float(kpi[:, 2].mean())
                            if score > best_score:
                                best_score = score
                                best_i = i
                                best_box_xyxy = box_i

                        if best_i >= 0 and best_score >= MIN_MEAN_CONF:
                            k = kall[best_i].astype(np.float32)
                            kn = k.copy()
                            kn[:, 0] /= float(w)
                            kn[:, 1] /= float(h)

                            if best_box_xyxy is not None:
                                box_xyxy = best_box_xyxy
                                x1, y1, x2, y2 = [float(x) for x in box_xyxy.flatten()[:4]]
                                bw, bh = max(x2 - x1, EPS), max(y2 - y1, EPS)
                                box_draw = box_xyxy
                            else:
                                bw, bh = float(w), float(h)

                            vec = frame_to_vector_60(kn)
                            missing_frames = 0
                            current_mean_conf = float(k[:, 2].mean())
                            current_confident_kpts = int(np.sum(k[:, 2] > 0.30))
                            k_xy_draw = k[:, :2].copy()
                            k_c_draw = k[:, 2].copy()
                            prev_valid_box = box_draw.copy() if box_draw is not None else None
                        else:
                            missing_frames += 1
                else:
                    missing_frames += 1

                if vec is not None:
                    feat_buffer.append(vec.astype(np.float32))
                    if box_draw is not None:
                        _, cy_now, _, _ = _bbox_center_and_size(box_draw) or (0.0, 0.0, 0.0, 0.0)
                        y_buffer.append(cy_now / float(max(h, 1)))
                else:
                    prev_valid_box = None
                    if missing_frames >= MAX_MISSING_FRAMES:
                        feat_buffer.clear()
                        y_buffer.clear()
                        self.log_message.emit("Mất tracking quá lâu -> clear buffer.")

                if k_xy_draw is not None and k_c_draw is not None:
                    pts = np.zeros((17, 2), dtype=np.int32)
                    for i in range(17):
                        pts[i, 0] = int(np.clip(k_xy_draw[i, 0], 0, w - 1))
                        pts[i, 1] = int(np.clip(k_xy_draw[i, 1], 0, h - 1))
                    for a, b in COCO_EDGES:
                        if float(k_c_draw[a]) < 0.1 or float(k_c_draw[b]) < 0.1:
                            continue
                        cv2.line(display, tuple(pts[a]), tuple(pts[b]), (0, 255, 180), 2, cv2.LINE_AA)
                    for i in range(17):
                        if float(k_c_draw[i]) < 0.1:
                            continue
                        cv2.circle(display, tuple(pts[i]), 4, (0, 128, 255), -1, cv2.LINE_AA)
                    label = f"kpts={int(np.sum(k_c_draw > 0.30))}/17 conf={float(np.mean(k_c_draw)):.2f}"
                    cv2.putText(
                        display,
                        label,
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    if box_draw is not None and len(box_draw.flatten()) >= 4:
                        x1, y1, x2, y2 = [int(x) for x in box_draw.flatten()[:4]]
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)

                person_visible_now = vec is not None
                stable_track_now = _is_stable_track(box_draw, prev_box_for_stability, w, h)

                fall_like_now = False
                sofa_like_now = False
                torso_angle_deg = _torso_angle_deg(k_xy_draw)
                if box_draw is not None:
                    posture_feats = _box_posture_features(box_draw, h)
                    if posture_feats is not None:
                        _, _, bbox_ratio, bottom_ratio = posture_feats
                        if len(y_buffer) >= DROP_LOOKBACK_FRAMES + 1:
                            current_y = y_buffer[-1]
                            lookback_y = y_buffer[-(DROP_LOOKBACK_FRAMES + 1)]
                            centroid_drop_now = (current_y - lookback_y) > MIN_DROP_DELTA_CENTER_Y
                        else:
                            centroid_drop_now = False

                        tilt_score = torso_angle_deg if torso_angle_deg is not None else 0.0
                        vertical_span = 1.0 / max(bbox_ratio, EPS)
                        strong_fall_shape = bbox_ratio >= 1.15 and bottom_ratio >= FALL_MIN_BOTTOM_RATIO
                        moderate_fall_shape = bbox_ratio >= 0.85 and bottom_ratio >= FALL_MIN_BOTTOM_RATIO and centroid_drop_now
                        fall_like_now = strong_fall_shape or moderate_fall_shape or tilt_score >= TILT_HIGH_THRESHOLD
                        sofa_like_now = (
                            bbox_ratio < 0.75
                            and bottom_ratio < SOFA_SIT_BOTTOM_RATIO_MAX
                            and tilt_score < TILT_LOW_THRESHOLD
                        )

                infer_now = (
                    len(feat_buffer) >= MIN_VALID_FRAMES_FOR_INFER
                    and person_visible_now
                    and (frame_i % INFER_STRIDE == 0 or len(feat_buffer) == SEQ_LEN)
                )

                if infer_now:
                    seq = np.stack(list(feat_buffer), axis=0)
                    seq_fixed = resample_to_length(seq, SEQ_LEN)
                    x = torch.from_numpy(seq_fixed).float().unsqueeze(0).to(device)
                    t_tfm_start = time.perf_counter()
                    with torch.no_grad():
                        logit = model(x)
                        prob = float(torch.sigmoid(logit).cpu().item())
                    transformer_ms = (time.perf_counter() - t_tfm_start) * 1000.0
                    grade = _confidence_grade(prob)
                    if frame_i % 15 == 0:
                        self.log_message.emit(f"p(fall)={prob:.4f} grade={grade}")

                    prob_color = (0, 0, 255) if grade == "high" else ((0, 165, 255) if grade == "medium" else (0, 255, 0))
                    cv2.putText(
                        display,
                        f"p(fall)={prob:.3f} [{grade}]",
                        (10, 88),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        prob_color,
                        2,
                        cv2.LINE_AA,
                    )

                    alert_prob_threshold = min(infer_threshold, 0.18)
                    fall_score_now = prob >= alert_prob_threshold and person_visible_now and fall_like_now and not sofa_like_now

                    if fall_score_now:
                        detect_mono = time.perf_counter()
                        self.log_message.emit(f"[fall] detected_at={detect_mono:.6f}")
                        self.fall_detected.emit(prob)
                        snap = self._draw_alert_frame(display, k_xy_draw, k_c_draw, box_draw)
                        cv2.imwrite(str(alert_path), snap)
                        submitted = notifier.maybe_notify_async(alert_path, caption=f"Fall p={prob:.3f}")
                        if submitted:
                            submit_ms = (time.perf_counter() - detect_mono) * 1000.0
                            send_ms = notifier.last_send_elapsed_ms
                            if send_ms is not None:
                                total_ms = submit_ms + send_ms
                                self.log_message.emit(f"Alert latency = {total_ms:.1f} ms (submit={submit_ms:.1f} ms, send={send_ms:.1f} ms)")
                            else:
                                self.log_message.emit(f"Đã gửi Telegram cảnh báo (background thread), submit={submit_ms:.2f}ms.")
                        elif not notifier.enabled():
                            self.log_message.emit("Thiếu TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.")
                        y_buffer.clear()

                total_ms = (time.perf_counter() - t_loop_start) * 1000.0
                post_ms = max(total_ms - read_ms - pre_ms - pose_ms - (transformer_ms if 'transformer_ms' in locals() else 0.0), 0.0)
                self.profiler.add_frame_times(
                    read_ms,
                    pre_ms,
                    pose_ms,
                    transformer_ms if 'transformer_ms' in locals() else 0.0,
                    post_ms,
                )

                if frame_i % 100 == 0 and self.profiler.frames > 0:
                    s = self.profiler.summary()
                    self.log_message.emit(
                        "[profile] "
                        f"read={s['read_ms']:.2f}ms "
                        f"pre={s['preprocess_ms']:.2f}ms "
                        f"pose={s['pose_ms']:.2f}ms "
                        f"tfm={s['transformer_ms']:.2f}ms "
                        f"post={s['post_ms']:.2f}ms "
                        f"fps={s['fps']:.2f}"
                    )

                frame_i += 1

                rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
                ch, cw = rgb.shape[:2]
                qimg = QImage(rgb.data, cw, ch, rgb.strides[0], QImage.Format_RGB888).copy()
                self.frame_ready.emit(qimg)

        finally:
            cap.release()
            notifier.shutdown()
            if self.profiler.frames > 0:
                s = self.profiler.summary()
                self.log_message.emit(
                    "[profile-summary] "
                    f"frames={int(s['frames'])} "
                    f"read={s['read_ms']:.2f}ms "
                    f"pre={s['preprocess_ms']:.2f}ms "
                    f"pose={s['pose_ms']:.2f}ms "
                    f"tfm={s['transformer_ms']:.2f}ms "
                    f"post={s['post_ms']:.2f}ms "
                    f"fps={s['fps']:.2f} "
                    f"pose_params={s['pose_params_m']:.2f}M "
                    f"pose_gflops={s['pose_gflops']:.2f} "
                    f"tfm_params={s['transformer_params_m']:.2f}M "
                    f"tfm_gflops={s['transformer_gflops']:.2f} "
                    f"pose_mb={s['pose_model_mb']:.2f} "
                    f"tfm_mb={s['transformer_model_mb']:.2f}"
                )
            self.log_message.emit("Luồng kết thúc.")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hybrid Pose-Informed Transformer — Fall Detection")
        self._worker: InferenceWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.video_label = QLabel("Video")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; color: #888;")
        layout.addWidget(self.video_label)

        row = QHBoxLayout()
        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_load = QPushButton("Load Video")
        self.btn_rtsp = QPushButton("Connect RTSP")
        row.addWidget(self.btn_webcam)
        row.addWidget(self.btn_load)
        row.addWidget(self.btn_rtsp)
        layout.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)

        self.btn_webcam.clicked.connect(self._start_webcam)
        self.btn_load.clicked.connect(self._load_video)
        self.btn_rtsp.clicked.connect(self._connect_rtsp)

    def _append_log(self, s: str) -> None:
        self.log.append(s)

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(4000)
            self._worker = None

    def _start_inference(self, source: str | int) -> None:
        self._stop_worker()
        wpath = _ROOT / "best_hybrid_transformer.pth"
        if not wpath.is_file():
            QMessageBox.warning(self, "Thiếu model", f"Không thấy {wpath}")
            return
        self._worker = InferenceWorker(source, "yolo11n-pose.pt", wpath)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.log_message.connect(self._append_log)
        self._worker.fall_detected.connect(lambda p: self._append_log(f"[FALL] p={p:.4f}"))
        self._worker.start()

    def _start_webcam(self) -> None:
        self._start_inference(0)
        self._append_log("Webcam đã bật.")

    def _load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            str(_ROOT),
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All (*)",
        )
        if not path:
            return
        self._start_inference(path)
        self._append_log(f"Đang phát: {path}")

    def _connect_rtsp(self) -> None:
        url, ok = QInputDialog.getText(
            self,
            "Kết nối camera RTSP/HTTP",
            "Nhập URL stream (ví dụ rtsp://user:pass@ip:554/stream1):",
        )
        if not ok:
            return
        stream_url = url.strip()
        if not stream_url:
            QMessageBox.information(self, "Thiếu URL", "Bạn chưa nhập URL camera.")
            return
        if not stream_url.startswith(("rtsp://", "http://", "https://")):
            QMessageBox.warning(self, "URL không hợp lệ", "URL phải bắt đầu bằng rtsp://, http:// hoặc https://")
            return
        self._start_inference(stream_url)
        self._append_log(f"Đang kết nối stream: {stream_url}")

    def _show_frame(self, img: QImage) -> None:
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._stop_worker()
        event.accept()


def main() -> None:
    load_dotenv_file(_ROOT / ".env")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        win.close()
        app.quit()


if __name__ == "__main__":
    main()
