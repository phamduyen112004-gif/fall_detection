"""
Giao diện Tkinter cho Hybrid Fall Detection + Telegram Alert.
Ba nút theo tài liệu: Webcam trực tiếp, Tải video, Cài đặt góc ngã.

Chạy: python gui_app.py
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from src.config import PipelineConfig
from src.pipeline import FrameDiag, HybridFallPipeline
from src.stage4_alert import TelegramAlerter


@dataclass
class GuiSettings:
    """Tham số có thể chỉnh từ cửa sổ Settings."""

    laydown_torso_angle_deg: float = 55.0
    laydown_nose_ankle_angle_deg: float = 50.0
    fall_min_frames: int = 60
    fall_min_seconds: float | None = 10.0
    min_mean_keypoint_conf: float = 0.2
    mirror_webcam: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    def to_pipeline_config(self) -> PipelineConfig:
        tok = self.telegram_bot_token.strip() or None
        cid = self.telegram_chat_id.strip() or None
        return PipelineConfig(
            laydown_torso_angle_deg=float(self.laydown_torso_angle_deg),
            laydown_nose_ankle_angle_deg=float(self.laydown_nose_ankle_angle_deg),
            fall_min_frames=int(self.fall_min_frames),
            fall_min_seconds=self.fall_min_seconds,
            min_mean_keypoint_conf=float(self.min_mean_keypoint_conf),
            telegram_bot_token=tok,
            telegram_chat_id=cid,
        )


def bgr_to_photoimage(master: tk.Misc, frame_bgr: np.ndarray) -> tk.PhotoImage:
    """BGR → PhotoImage (PPM trong bộ nhớ), không cần Pillow."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    header = f"P6 {w} {h} 255\n".encode("ascii")
    return tk.PhotoImage(master=master, data=header + rgb.tobytes(), format="ppm")


class FallDetectionApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Hệ thống phát hiện ngã — Hybrid Fall Detection + Telegram")
        self.minsize(720, 560)
        self.settings = GuiSettings()

        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._frame_queue: queue.Queue[tuple[np.ndarray, FrameDiag] | None] = queue.Queue(
            maxsize=2
        )

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(16, self._pump_frame_queue)

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 6}

        title = ttk.Label(
            self,
            text="Phát hiện ngã lai (YOLOv11-pose + hình học + bộ lọc thời gian)",
            font=("Segoe UI", 11, "bold"),
        )
        title.pack(**pad)

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, **pad)

        self.btn_webcam = ttk.Button(
            btn_frame,
            text="Live Webcam Inference\n(Chạy Webcam)",
            command=self._start_webcam,
        )
        self.btn_webcam.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self.btn_upload = ttk.Button(
            btn_frame,
            text="Upload Video Inference\n(Tải video lên)",
            command=self._start_upload,
        )
        self.btn_upload.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        self.btn_settings = ttk.Button(
            btn_frame,
            text="Settings\n(Cài đặt góc ngã)",
            command=self._open_settings,
        )
        self.btn_settings.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        ctrl = ttk.Frame(self)
        ctrl.pack(fill=tk.X, **pad)
        self.btn_stop = ttk.Button(
            ctrl, text="Dừng xử lý", command=self._request_stop, state=tk.DISABLED
        )
        self.btn_stop.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Sẵn sàng. Chọn Webcam hoặc Tải video.")
        ttk.Label(ctrl, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        self.alert_hint = ttk.Label(
            self,
            text="Telegram: điền Bot Token + Chat ID trong Settings hoặc biến môi trường.",
            foreground="#444",
        )
        self.alert_hint.pack(**pad)

        preview = ttk.LabelFrame(self, text="Luồng xử lý (640×640)")
        preview.pack(fill=tk.BOTH, expand=True, **pad)
        self.preview_label = ttk.Label(preview, anchor=tk.CENTER)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        self._preview_photo: tk.PhotoImage | None = None

    def _set_running_ui(self, running: bool) -> None:
        state = tk.DISABLED if running else tk.NORMAL
        for b in (self.btn_webcam, self.btn_upload, self.btn_settings):
            b.configure(state=state)
        self.btn_stop.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _request_stop(self) -> None:
        self._stop.set()
        self.status_var.set("Đang dừng…")

    def _on_close(self) -> None:
        self._request_stop()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
        self.destroy()

    def _pump_frame_queue(self) -> None:
        try:
            while True:
                item = self._frame_queue.get_nowait()
                if item is None:
                    break
                disp, diag = item
                photo = bgr_to_photoimage(self, disp)
                self._preview_photo = photo
                self.preview_label.configure(image=photo)
                mc = (
                    f"{diag.mean_kpt_conf:.2f}"
                    if diag.mean_kpt_conf is not None
                    else "—"
                )
                self.status_var.set(
                    f"Tư thế: {diag.posture} | conf_tb={mc} | "
                    f"góc_thân={diag.torso_deg} | góc_mũi_cổ_chân={diag.nose_ankle_deg}"
                )
        except queue.Empty:
            pass
        self.after(16, self._pump_frame_queue)

    def _notify_fall(self, diag: FrameDiag) -> None:
        self.status_var.set(
            "⚠ NGÃ ĐÃ XÁC NHẬN — đã gửi cảnh báo (nếu cấu hình Telegram)."
        )
        messagebox.showwarning(
            "Phát hiện ngã",
            "Hệ thống đã xác nhận sự kiện NGÃ.\n"
            "Ảnh snapshot đã được gửi qua Telegram (nếu token/chat hợp lệ).",
            parent=self,
        )

    def _start_webcam(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._drain_queue()
        self._set_running_ui(True)
        self.status_var.set("Đang khởi động webcam…")
        cfg = self.settings.to_pipeline_config()
        mirror = self.settings.mirror_webcam
        self._worker = threading.Thread(
            target=self._run_capture_loop,
            args=(0, cfg, mirror),
            daemon=True,
        )
        self._worker.start()

    def _start_upload(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Chọn file video",
            filetypes=[
                ("Video", "*.mp4 *.avi *.mov *.mkv *.webm"),
                ("Tất cả", "*.*"),
            ],
        )
        if not path:
            return
        self._stop.clear()
        self._drain_queue()
        self._set_running_ui(True)
        self.status_var.set(f"Đang xử lý: {path}")
        cfg = self.settings.to_pipeline_config()
        self._worker = threading.Thread(
            target=self._run_capture_loop,
            args=(path, cfg, False),
            daemon=True,
        )
        self._worker.start()

    def _drain_queue(self) -> None:
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

    def _run_capture_loop(
        self,
        source: int | str,
        cfg: PipelineConfig,
        mirror: bool,
    ) -> None:
        cap: cv2.VideoCapture | None = None
        try:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.after(
                    0,
                    lambda: messagebox.showerror(
                        "Lỗi",
                        f"Không mở được nguồn: {source}",
                        parent=self,
                    ),
                )
                return

            pipe = HybridFallPipeline(cfg)
            alerter = TelegramAlerter(cfg)

            def on_fall(vis: np.ndarray, diag: FrameDiag) -> None:
                if alerter.enabled():
                    try:
                        alerter.send_fall_alert(
                            vis,
                            extra_text=(
                                f"conf_tb={diag.mean_kpt_conf}, "
                                f"góc_thân={diag.torso_deg}, "
                                f"góc_mũi_cổ_chân={diag.nose_ankle_deg}"
                            ),
                        )
                    except Exception as e:
                        self.after(
                            0,
                            lambda err=str(e): messagebox.showerror(
                                "Telegram",
                                f"Gửi cảnh báo thất bại:\n{err}",
                                parent=self,
                            ),
                        )
                self.after(0, lambda d=diag: self._notify_fall(d))

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)
                diag, display = pipe.process_frame_with_display(frame, on_fall=on_fall)
                try:
                    self._frame_queue.put_nowait((display, diag))
                except queue.Full:
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait((display, diag))
                    except queue.Full:
                        pass
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            self._frame_queue.put(None)
            self.after(0, self._on_worker_done)

    def _on_worker_done(self) -> None:
        try:
            self._frame_queue.get_nowait()
        except queue.Empty:
            pass
        self._set_running_ui(False)
        self.status_var.set("Đã dừng. Chọn Webcam hoặc Tải video.")
        self._worker = None

    def _open_settings(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo(
                "Settings",
                "Vui lòng dừng xử lý trước khi đổi cài đặt.",
                parent=self,
            )
            return
        win = tk.Toplevel(self)
        win.title("Settings — Cài đặt góc ngã & Telegram")
        win.minsize(420, 420)
        win.transient(self)

        r = 0

        def add_row(label: str, widget: tk.Widget) -> None:
            nonlocal r
            ttk.Label(win, text=label).grid(row=r, column=0, sticky=tk.W, padx=8, pady=4)
            widget.grid(row=r, column=1, sticky=tk.EW, padx=8, pady=4)
            r += 1

        win.columnconfigure(1, weight=1)

        s = self.settings
        v_torso = tk.DoubleVar(value=s.laydown_torso_angle_deg)
        v_nose = tk.DoubleVar(value=s.laydown_nose_ankle_angle_deg)
        v_frames = tk.IntVar(value=s.fall_min_frames)
        v_conf = tk.DoubleVar(value=s.min_mean_keypoint_conf)
        v_secs = tk.StringVar(
            value="" if s.fall_min_seconds is None else str(s.fall_min_seconds)
        )
        v_mirror = tk.BooleanVar(value=s.mirror_webcam)
        v_tok = tk.StringVar(value=s.telegram_bot_token)
        v_chat = tk.StringVar(value=s.telegram_chat_id)

        add_row("Góc thân (Torso) — ngưỡng laydown (°)", ttk.Scale(win, from_=30, to=90, variable=v_torso))
        add_row("", ttk.Label(win, textvariable=v_torso))
        add_row("Góc mũi–cổ chân — ngưỡng laydown (°)", ttk.Scale(win, from_=30, to=90, variable=v_nose))
        add_row("", ttk.Label(win, textvariable=v_nose))
        add_row("Số frame tối thiểu (laydown liên tục)", ttk.Spinbox(win, from_=1, to=600, textvariable=v_frames))
        add_row(
            "Thời gian tối thiểu (giây, để trống = tắt)",
            ttk.Entry(win, textvariable=v_secs),
        )
        add_row("Confidence TB tối thiểu (0–1)", ttk.Spinbox(win, from_=0.05, to=0.9, increment=0.05, textvariable=v_conf))
        add_row("Lật ngang webcam (mirror)", ttk.Checkbutton(win, variable=v_mirror))

        ttk.Separator(win, orient=tk.HORIZONTAL).grid(row=r, column=0, columnspan=2, sticky=tk.EW, pady=8)
        r += 1
        ttk.Label(win, text="Telegram (tùy chọn)", font=("Segoe UI", 9, "bold")).grid(
            row=r, column=0, columnspan=2, sticky=tk.W, padx=8
        )
        r += 1
        add_row("Bot Token", ttk.Entry(win, textvariable=v_tok, width=40, show="•"))
        add_row("Chat ID", ttk.Entry(win, textvariable=v_chat, width=40))

        def apply_settings() -> None:
            secs_raw = v_secs.get().strip()
            secs: float | None
            if not secs_raw:
                secs = None
            else:
                try:
                    secs = float(secs_raw)
                except ValueError:
                    messagebox.showerror("Lỗi", "Thời gian (giây) không hợp lệ.", parent=win)
                    return
            self.settings = GuiSettings(
                laydown_torso_angle_deg=float(v_torso.get()),
                laydown_nose_ankle_angle_deg=float(v_nose.get()),
                fall_min_frames=int(v_frames.get()),
                fall_min_seconds=secs,
                min_mean_keypoint_conf=float(v_conf.get()),
                mirror_webcam=bool(v_mirror.get()),
                telegram_bot_token=v_tok.get().strip(),
                telegram_chat_id=v_chat.get().strip(),
            )
            messagebox.showinfo("Settings", "Đã lưu cài đặt.", parent=win)
            win.destroy()

        bf = ttk.Frame(win)
        bf.grid(row=r, column=0, columnspan=2, pady=12)
        ttk.Button(bf, text="Áp dụng & Đóng", command=apply_settings).pack(side=tk.RIGHT, padx=6)
        ttk.Button(bf, text="Hủy", command=win.destroy).pack(side=tk.RIGHT)


def main() -> None:
    app = FallDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
