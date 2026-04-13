#!/usr/bin/env python3
"""
app_inference.py — PyQt5: webcam/video + HybridFallTransformer + cảnh báo Telegram.

Luồng: QThread đọc khung → YOLOv11-Pose → vector 60-D (giống data_extractor) → buffer 60 frame
→ suy luận → sigmoid > ngưỡng (best_threshold trong checkpoint, mặc định 0.5) = Fall.
Cooldown Telegram 10s; lưu alert.jpg (khung xương + bbox).

Biến môi trường: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Chạy:
  python app_inference.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

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
    IMGSZ,
    MIN_MEAN_CONF,
    SEQ_LEN,
    frame_to_vector_60,
    resample_to_length,
)
from src.viz import COCO_EDGES
from ultralytics import YOLO


def draw_alert_frame(
    frame_bgr: np.ndarray,
    keypoints_xy: np.ndarray | None,
    keypoints_conf: np.ndarray | None,
    box_xyxy: np.ndarray | None,
) -> np.ndarray:
    """Vẽ bbox + skeleton lên bản sao khung BGR."""
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
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
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


class TelegramNotifier:
    """Gửi ảnh lên Telegram; cooldown 10 giây giữa các lần gửi."""

    def __init__(self, cooldown_sec: float = 10.0) -> None:
        self.cooldown_sec = cooldown_sec
        self._last_send_mono: float | None = None

    def send_photo(self, image_path: Path, caption: str = "Fall detected") -> bool:
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            return False
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with image_path.open("rb") as fp:
                files = {"photo": fp}
                data = {"chat_id": chat_id, "caption": caption[:1024]}
                r = requests.post(url, files=files, data=data, timeout=30)
            return r.status_code == 200
        except (OSError, requests.RequestException, ValueError):
            return False

    def maybe_notify(self, image_path: Path, caption: str = "Fall detected") -> bool:
        now = time.monotonic()
        if self._last_send_mono is not None:
            if now - self._last_send_mono < self.cooldown_sec:
                return False
        ok = self.send_photo(image_path, caption=caption)
        if ok:
            self._last_send_mono = now
        return ok


class InferenceWorker(QThread):
    """Đọc video/webcam, suy luận, phát tín hiệu khung + log + fall."""

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

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pose = YOLO(self.weights_pose)
        model = HybridFallTransformer().to(device)
        try:
            ck = torch.load(self.weights_cls, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(self.weights_cls, map_location=device)
        state = ck.get("model_state_dict", ck)
        model.load_state_dict(state, strict=True)
        model.eval()
        infer_threshold = float(ck.get("best_threshold", 0.5))

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.log_message.emit(f"Không mở được nguồn: {self.source}")
            return

        feat_buffer: list[np.ndarray] = []
        last_vec: np.ndarray | None = None
        notifier = TelegramNotifier(cooldown_sec=10.0)
        alert_path = _ROOT / "alert.jpg"
        frame_i = 0

        while self._running:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_bgr = cv2.resize(frame_bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
            h, w = frame_bgr.shape[:2]
            results = pose.predict(frame_bgr, imgsz=IMGSZ, verbose=False)

            display = frame_bgr.copy()
            k_xy_draw: np.ndarray | None = None
            k_c_draw: np.ndarray | None = None
            box_draw: np.ndarray | None = None

            if (
                results
                and results[0].keypoints is not None
                and results[0].keypoints.data is not None
            ):
                r0 = results[0]
                kall = r0.keypoints.data.cpu().numpy()
                if kall.size > 0:
                    best_i = int(np.argmax([float(k[:, 2].mean()) for k in kall]))
                    k = kall[best_i].astype(np.float32)
                    mean_c = float(k[:, 2].mean())

                    if mean_c >= MIN_MEAN_CONF:
                        kn = k.copy()
                        kn[:, 0] /= float(w)
                        kn[:, 1] /= float(h)

                        if r0.boxes is not None and len(r0.boxes) > best_i:
                            box_xyxy = r0.boxes.xyxy[best_i].cpu().numpy()
                            x1, y1, x2, y2 = [float(x) for x in box_xyxy.flatten()[:4]]
                            bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                            box_draw = box_xyxy
                        else:
                            bw, bh = float(w), float(h)

                        vec = frame_to_vector_60(kn, (bw, bh))
                        last_vec = vec.copy()
                    else:
                        vec = last_vec.copy() if last_vec is not None else None
                else:
                    vec = last_vec.copy() if last_vec is not None else None
            else:
                vec = last_vec.copy() if last_vec is not None else None

            if vec is not None:
                feat_buffer.append(vec.astype(np.float32))
                if len(feat_buffer) > SEQ_LEN:
                    feat_buffer.pop(0)

            # Hiển thị overlay nếu có detection gần nhất
            if results and results[0].keypoints is not None and results[0].keypoints.data is not None:
                r0 = results[0]
                kall = r0.keypoints.data.cpu().numpy()
                if kall.size > 0:
                    bi = int(np.argmax([float(x[:, 2].mean()) for x in kall]))
                    kk = kall[bi]
                    k_xy_draw = kk[:, :2].copy()
                    k_c_draw = kk[:, 2].copy()
                    if r0.boxes is not None and len(r0.boxes) > bi:
                        box_draw = r0.boxes.xyxy[bi].cpu().numpy()

            if len(feat_buffer) > 0:
                seq = np.stack(feat_buffer, axis=0)
                seq_fixed = resample_to_length(seq, SEQ_LEN)
                x = torch.from_numpy(seq_fixed).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logit = model(x)
                    prob = float(torch.sigmoid(logit).cpu().item())
                if frame_i % 15 == 0:
                    self.log_message.emit(f"p(fall)={prob:.4f}")

                if prob > infer_threshold:
                    self.fall_detected.emit(prob)
                    snap = draw_alert_frame(display, k_xy_draw, k_c_draw, box_draw)
                    cv2.imwrite(str(alert_path), snap)
                    sent = notifier.maybe_notify(alert_path, caption=f"Fall p={prob:.3f}")
                    if sent:
                        self.log_message.emit("Đã gửi cảnh báo Telegram (sau cooldown).")
                    elif not os.environ.get("TELEGRAM_BOT_TOKEN", "").strip():
                        self.log_message.emit("Thiếu TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.")
            frame_i += 1

            rgb = np.ascontiguousarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            ch, cw = rgb.shape[:2]
            qimg = QImage(
                rgb.data,
                cw,
                ch,
                rgb.strides[0],
                QImage.Format_RGB888,
            ).copy()
            self.frame_ready.emit(qimg)

        cap.release()
        self.log_message.emit("Luồng kết thúc.")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Hybrid Pose-Informed Transformer — Fall Detection")
        self._worker: InferenceWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #222; color: #888;")
        self.video_label.setText("Video")
        layout.addWidget(self.video_label)

        row = QHBoxLayout()
        self.btn_webcam = QPushButton("Start Webcam")
        self.btn_load = QPushButton("Load Video")
        row.addWidget(self.btn_webcam)
        row.addWidget(self.btn_load)
        layout.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)

        self.btn_webcam.clicked.connect(self._start_webcam)
        self.btn_load.clicked.connect(self._load_video)

    def _append_log(self, s: str) -> None:
        self.log.append(s)

    def _stop_worker(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
            self._worker = None

    def _start_webcam(self) -> None:
        self._stop_worker()
        wpath = _ROOT / "best_hybrid_transformer.pth"
        if not wpath.is_file():
            QMessageBox.warning(self, "Thiếu model", f"Không thấy {wpath}")
            return
        self._worker = InferenceWorker(0, "yolo11n-pose.pt", wpath)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.log_message.connect(self._append_log)
        self._worker.fall_detected.connect(lambda p: self._append_log(f"[FALL] p={p:.4f}"))
        self._worker.start()
        self._append_log("Webcam đã bật.")

    def _load_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn video",
            str(_ROOT),
            "Video (*.mp4 *.avi *.mov *.mkv);;All (*)",
        )
        if not path:
            return
        self._stop_worker()
        wpath = _ROOT / "best_hybrid_transformer.pth"
        if not wpath.is_file():
            QMessageBox.warning(self, "Thiếu model", f"Không thấy {wpath}")
            return
        self._worker = InferenceWorker(path, "yolo11n-pose.pt", wpath)
        self._worker.frame_ready.connect(self._show_frame)
        self._worker.log_message.connect(self._append_log)
        self._worker.fall_detected.connect(lambda p: self._append_log(f"[FALL] p={p:.4f}"))
        self._worker.start()
        self._append_log(f"Đang phát: {path}")

    def _show_frame(self, img: QImage) -> None:
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(
            pix.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._stop_worker()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(900, 700)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
