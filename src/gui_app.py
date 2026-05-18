#!/usr/bin/env python
"""
Real-Time Fall Detection GUI Application

A production-ready PyQt5 application for live fall detection using:
- YOLOv11-Pose for keypoint extraction
- HybridFallTransformer for temporal classification
- Telegram Bot alerts for emergency notifications
"""

import os
import gc
import time
import logging
import collections
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QGroupBox, QMessageBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

from src.hybrid_transformer import HybridFallTransformer
from src.pifr_features import extract_keypoints, compute_pifr
from src.config import (
    PipelineConfig,
    DEFAULT_CONFIG,
    TRAINING_CONFIG,
    YOLO_MODEL,
    MODEL_SAVE_DIR,
    LOG_DIR,
    ALERT_COOLDOWN_SEC,
    ALERT_MIN_PROB,
)


# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

# Inference thresholds
_KEYPOINT_CONFIDENCE_THRESHOLD: float = 0.3
"""Minimum confidence score for a keypoint to be rendered (0.0-1.0)."""

_FPS_UPDATE_INTERVAL_SEC: float = 1.0
"""Elapsed time in seconds before FPS recalculation."""

# Video display dimensions
_VIDEO_LABEL_WIDTH: int = 800
"""Minimum width of the video display label in pixels."""
_VIDEO_LABEL_HEIGHT: int = 500
"""Minimum height of the video display label in pixels."""

# Window geometry
_WINDOW_X: int = 100
_WINDOW_Y: int = 100
_WINDOW_WIDTH: int = 1200
_WINDOW_HEIGHT: int = 800

# Layout spacing
_MAIN_LAYOUT_SPACING: int = 20
_CONTROL_LAYOUT_SPACING: int = 15

# Alert overlay
_ALERT_BORDER_WIDTH: int = 10
"""Width of the red border around frame when fall is detected (in pixels)."""
_ALERT_TEXT_X: int = 20
_ALERT_TEXT_Y: int = 50
_ALERT_FONT_SCALE: float = 1.2
_ALERT_TEXT_THICKNESS: int = 3

# Telegram cooldown
_DEFAULT_TELEGRAM_COOLDOWN: int = 15
"""Default cooldown between Telegram alerts in seconds."""


class Config:
    """Application configuration - wraps PipelineConfig and TrainingConfig."""
    
    def __init__(
        self,
        pipeline_config: Optional[PipelineConfig] = None,
        train_config: Optional[object] = None
    ) -> None:
        """
        Initialize application configuration.
        
        Args:
            pipeline_config: Optional PipelineConfig instance. Uses DEFAULT_CONFIG if None.
            train_config: Optional TrainingConfig instance. Uses TRAINING_CONFIG if None.
        """
        self._pipeline = pipeline_config or DEFAULT_CONFIG
        self._train = train_config or TRAINING_CONFIG
    
    # Model paths
    @property
    def YOLO_MODEL(self) -> str:
        """Path to the YOLO pose estimation model."""
        return self._pipeline.pose_model
    
    @property
    def TRANSFORMER_MODEL(self) -> str:
        """Path to the trained Transformer model weights."""
        return str(MODEL_SAVE_DIR / "best_model.pth")
    
    # Transformer config (from TrainingConfig)
    @property
    def INPUT_DIM(self) -> int:
        """Input feature dimension for the Transformer."""
        return self._train.INPUT_DIM
    
    @property
    def NUM_FRAMES(self) -> int:
        """Number of frames in the temporal window."""
        return self._train.NUM_FRAMES
    
    @property
    def D_MODEL(self) -> int:
        """Model dimension for Transformer embeddings."""
        return self._train.D_MODEL
    
    @property
    def NHEAD(self) -> int:
        """Number of attention heads in Transformer."""
        return self._train.NHEAD
    
    @property
    def NUM_LAYERS(self) -> int:
        """Number of Transformer encoder layers."""
        return self._train.NUM_LAYERS
    
    @property
    def DROPOUT(self) -> float:
        """Dropout probability for Transformer."""
        return self._train.DROPOUT
    
    # Inference
    FALL_THRESHOLD: float = 0.6
    """Display threshold for fall detection (probability > threshold = fall)."""
    
    WINDOW_SIZE: int = 60
    """Sliding window length for temporal features."""
    
    STRIDE: int = 15
    """SOTA: Run inference every N frames to improve FPS."""
    
    FPS: int = 30
    """Assumed frames per second for keypoint extraction."""
    
    # Telegram (from PipelineConfig)
    @property
    def TELEGRAM_BOT_TOKEN(self) -> str:
        """Telegram bot token for alert notifications."""
        return self._pipeline.telegram_bot_token or ""
    
    @property
    def TELEGRAM_CHAT_ID(self) -> str:
        """Telegram chat ID for alert destination."""
        return self._pipeline.telegram_chat_id or ""
    
    @property
    def ALERT_COOLDOWN(self) -> int:
        """Cooldown between Telegram alerts in seconds."""
        return ALERT_COOLDOWN_SEC
    
    @property
    def ALERT_MIN_PROB(self) -> float:
        """Minimum probability to trigger Telegram alert."""
        return ALERT_MIN_PROB
    
    # UI Colors
    COLOR_NORMAL: str = "#2ECC71"
    COLOR_FALL: str = "#E74C3C"
    COLOR_WARNING: str = "#F39C12"
    COLOR_BG: str = "#1A1A2E"
    COLOR_PANEL: str = "#16213E"
    COLOR_TEXT: str = "#FFFFFF"
    COLOR_ACCENT: str = "#0F3460"


# COCO skeleton connections for keypoint visualization
_COCO_CONNECTIONS: list[Tuple[int, int]] = [
    (5, 6),   # shoulders
    (5, 7),   # left arm
    (6, 8),   # right arm
    (7, 9),   # left forearm
    (8, 10),  # right forearm
    (5, 11),  # left torso
    (6, 12),  # right torso
    (11, 12), # hip
    (11, 13), # left thigh
    (12, 14), # right thigh
    (13, 15), # left shin
    (14, 16), # right shin
]
"""COCO keypoint connection pairs for skeleton visualization."""


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> logging.Logger:
    """
    Configure application logging with file and console handlers.
    
    Returns:
        Configured logger instance for the gui_app module.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "gui_app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("gui_app")


# ============================================================
# TELEGRAM ALERTS
# ============================================================

class TelegramAlert:
    """
    Telegram Bot alert system with cooldown to prevent notification spam.
    
    Attributes:
        bot_token: Telegram bot authentication token.
        chat_id: Target chat ID for alert messages.
        cooldown: Minimum seconds between consecutive alerts.
        last_alert_time: Timestamp of the last sent alert.
    
    Example:
        >>> alert = TelegramAlert("token", "chat_id", cooldown=15)
        >>> alert.send_alert(frame, confidence=0.85)
    """
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        cooldown: int = _DEFAULT_TELEGRAM_COOLDOWN
    ) -> None:
        """
        Initialize Telegram alert handler.
        
        Args:
            bot_token: Telegram bot token from BotFather.
            chat_id: Target chat ID for alerts.
            cooldown: Minimum seconds between alerts (default: 15).
        """
        self.bot_token: str = bot_token
        self.chat_id: str = chat_id
        self.cooldown: int = cooldown
        self.last_alert_time: float = 0.0
        self.logger: logging.Logger = logging.getLogger("gui_app.telegram")
    
    def _get_snapshot_path(self) -> Path:
        """Get the filesystem path for fall snapshot images."""
        return Path("fall_snapshot.jpg")
    
    def send_alert(
        self,
        frame: np.ndarray,
        confidence: float = 0.0,
        message: str = "EMERGENCY: Fall Detected!"
    ) -> bool:
        """
        Send Telegram alert with fall snapshot if cooldown has elapsed.
        
        Args:
            frame: Video frame to attach as snapshot.
            confidence: Fall probability (0.0-1.0) shown in caption.
            message: Custom alert message text.
        
        Returns:
            True if alert was sent successfully, False otherwise.
        """
        current_time: float = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.cooldown:
            self.logger.debug(f"Alert skipped (cooldown: {self.cooldown}s)")
            return False
        
        try:
            snapshot_path: str = str(self._get_snapshot_path())
            cv2.imwrite(snapshot_path, frame)
            
            url: str = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            with open(snapshot_path, "rb") as photo:
                files = {"photo": photo}
                data = {
                    "chat_id": self.chat_id,
                    "caption": f"FALL DETECTED!\nConfidence: {confidence:.1%}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
                response = requests.post(url, files=files, data=data, timeout=10)
            
            if response.status_code == 200:
                self.last_alert_time = current_time
                self.logger.info("Telegram alert sent successfully")
                return True
            else:
                self.logger.error(f"Telegram API error: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send Telegram alert: {e}")
            return False


# ============================================================
# VIDEO INFERENCE THREAD
# ============================================================

class VideoInferenceThread(QThread):
    """
    Dedicated QThread for video capture and AI inference pipeline.
    
    Runs entirely on a separate thread to prevent GUI freezing.
    Emits signals for frame updates, detection status, and errors.
    
    Signals:
        frame_ready: Emitted with processed frame (numpy array).
        status_update: Emitted with (status_text, status_color).
        fps_update: Emitted with current FPS (float).
        fall_detected: Emitted with detection state (bool).
        buffer_update: Emitted with (current_size, max_size).
        confidence_update: Emitted with fall probability (float).
        error_occurred: Emitted with error message (str).
    
    Example:
        >>> thread = VideoInferenceThread(config)
        >>> thread.set_source(0, 'webcam')
        >>> thread.start()
    """
    
    # PyQt signals for thread-safe GUI communication
    frame_ready = pyqtSignal(np.ndarray)
    status_update = pyqtSignal(str, str)
    fps_update = pyqtSignal(float)
    fall_detected = pyqtSignal(bool)
    buffer_update = pyqtSignal(int, int)
    confidence_update = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Config) -> None:
        """
        Initialize the inference thread.
        
        Args:
            config: Application Config instance with model paths and thresholds.
        """
        super().__init__()
        self.config: Config = config
        self.logger: logging.Logger = logging.getLogger("gui_app.inference")
        
        # Thread state
        self.running: bool = False
        self.source: Optional[int | str] = None
        self.source_type: Optional[str] = None
        
        # Model references
        self.yolo_model: Optional[object] = None
        self.transformer_model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None
        
        # Sliding window for temporal features
        self.feature_window: Deque[np.ndarray] = deque(maxlen=config.WINDOW_SIZE)
        
        # Stride-based inference optimization
        self.stride: int = config.STRIDE
        self.frame_counter: int = 0
        
        # Alert system
        self.telegram: TelegramAlert = TelegramAlert(
            config.TELEGRAM_BOT_TOKEN,
            config.TELEGRAM_CHAT_ID,
            config.ALERT_COOLDOWN
        )
        
        # Previous keypoints for padding when no person detected
        self.prev_keypoints: np.ndarray = np.zeros((17, 3), dtype=np.float32)
        
        # FPS tracking
        self.fps: float = 0.0
        self.frame_count: int = 0
        self.fps_start_time: float = time.time()
        
        # Cached probability for display
        self.current_probability: float = 0.0
    
    def load_models(self) -> bool:
        """
        Load YOLO pose model and Transformer classification model.
        
        Returns:
            True if both models loaded successfully, False otherwise.
        """
        try:
            from ultralytics import YOLO
            
            self.logger.info("Loading YOLO model...")
            self.yolo_model = YOLO(self.config.YOLO_MODEL)
            
            self.logger.info("Loading Transformer model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            self.transformer_model = HybridFallTransformer(
                input_dim=self.config.INPUT_DIM,
                num_frames=self.config.NUM_FRAMES,
                d_model=self.config.D_MODEL,
                nhead=self.config.NHEAD,
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT
            ).to(self.device)
            
            if os.path.exists(self.config.TRANSFORMER_MODEL):
                self.transformer_model.load_state_dict(
                    torch.load(
                        self.config.TRANSFORMER_MODEL,
                        map_location=self.device,
                        weights_only=True
                    )
                )
                self.logger.info("Transformer weights loaded successfully")
            else:
                self.logger.warning(
                    f"Transformer model not found: {self.config.TRANSFORMER_MODEL}"
                )
            
            self.transformer_model.eval()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False
    
    def set_source(self, source: int | str, source_type: str) -> None:
        """
        Set the video source for inference.
        
        Args:
            source: Video source (0 for webcam, file path, or RTSP URL).
            source_type: Type of source ('webcam', 'video', or 'rtsp').
        """
        self.source = source
        self.source_type = source_type
        self.logger.info(f"Source set: {source} (type: {source_type})")
    
    def run(self) -> None:
        """
        Main inference loop running in the separate QThread.
        
        Opens video capture, processes frames, and emits signals.
        Ensures resources are released in a finally block regardless of errors.
        """
        cap: Optional[cv2.VideoCapture] = None
        
        try:
            if not self.load_models():
                self.error_occurred.emit("Failed to load AI models")
                return
            
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error_occurred.emit(f"Failed to open video source: {self.source}")
                return
            
            self.running = True
            self.logger.info("Inference started")
            
            while self.running:
                ret: bool
                frame: np.ndarray
                ret, frame = cap.read()
                
                if not ret:
                    if self.source_type == 'video':
                        self.stop()
                        break
                    else:
                        time.sleep(0.1)
                        continue
                
                # Process frame through AI pipeline
                processed_frame, fall_detected = self.process_frame(frame)
                
                # Calculate and emit FPS
                self.frame_count += 1
                elapsed: float = time.time() - self.fps_start_time
                if elapsed >= _FPS_UPDATE_INTERVAL_SEC:
                    self.fps = self.frame_count / elapsed
                    self.fps_update.emit(self.fps)
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                
                # Emit signals for GUI updates
                self.frame_ready.emit(processed_frame)
                
                if fall_detected:
                    self.status_update.emit("FALL DETECTED", self.config.COLOR_FALL)
                    self.fall_detected.emit(True)
                else:
                    self.status_update.emit("NORMAL", self.config.COLOR_NORMAL)
                    self.fall_detected.emit(False)
        
        finally:
            # CRITICAL: Ensure resources are freed even on crash
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.logger.info("Inference thread stopped and resources released")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a single frame through the full AI inference pipeline.
        
        Pipeline steps:
            1. Extract keypoints with YOLO-Pose
            2. Compute 60D PIFR features
            3. Run stride-based Transformer inference
            4. Draw skeleton visualization
        
        Args:
            frame: Input video frame (BGR format).
        
        Returns:
            Tuple of (annotated_frame, fall_detected) where annotated_frame
            includes keypoint visualization and fall_detected is a bool flag.
        """
        # Step 1: Extract keypoints with YOLO
        keypoints: Optional[np.ndarray] = extract_keypoints(
            frame, self.yolo_model, self.config.FPS
        )
        
        if keypoints is None:
            keypoints = self.prev_keypoints.copy()
        else:
            self.prev_keypoints = keypoints.copy()
        
        # Step 2: Compute PIFR features (60D)
        h: int
        w: int
        h, w = frame.shape[:2]
        pifr_features: np.ndarray = compute_pifr(keypoints, w, h)
        
        # Add to sliding window
        self.feature_window.append(pifr_features)
        
        # Emit buffer update
        self.buffer_update.emit(len(self.feature_window), self.config.WINDOW_SIZE)
        
        # Step 3: Draw keypoints on frame
        annotated_frame: np.ndarray = self.draw_keypoints(frame.copy(), keypoints)
        
        # Step 4: SOTA Stride-based Inference
        self.frame_counter += 1
        fall_detected: bool = False
        
        if (self.frame_counter % self.stride == 0 and 
            len(self.feature_window) >= self.config.WINDOW_SIZE):
            fall_detected = self.check_fall_detection(self.config.FALL_THRESHOLD)
            self.confidence_update.emit(self.current_probability)
            
            if self.current_probability > self.config.ALERT_MIN_PROB:
                self.telegram.send_alert(frame, self.current_probability)
                annotated_frame = self.draw_fall_alert(annotated_frame)
        
        elif len(self.feature_window) >= self.config.WINDOW_SIZE:
            fall_detected = self.current_probability > self.config.FALL_THRESHOLD
        
        return annotated_frame, fall_detected
    
    def check_fall_detection(self, threshold: float = 0.5) -> bool:
        """
        Run inference on accumulated temporal features.
        
        Args:
            threshold: Probability threshold for fall detection.
                Display uses FALL_THRESHOLD (0.6), alert uses ALERT_MIN_PROB (0.18).
        
        Returns:
            True if fall probability exceeds threshold, False otherwise.
        """
        if self.transformer_model is None:
            return False
        
        # Prepare input tensor (batch=1, frames, features)
        features: np.ndarray = np.array(
            list(self.feature_window), dtype=np.float32
        )
        features = np.expand_dims(features, axis=0)
        tensor: torch.Tensor = torch.FloatTensor(features).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output: torch.Tensor = self.transformer_model(tensor)
            probability: float = torch.sigmoid(output).item()
        
        self.current_probability = probability
        return probability > threshold
    
    def get_current_probability(self) -> float:
        """
        Get the current fall probability from last inference.
        
        Returns:
            Fall probability value between 0.0 and 1.0.
        """
        return self.current_probability
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """
        Draw skeleton keypoints and connections on the frame.
        
        Args:
            frame: Input frame to draw on.
            keypoints: Array of shape (17, 3) with [x, y, confidence].
        
        Returns:
            Frame with skeleton visualization.
        """
        if keypoints is None or len(keypoints) < 17:
            return frame
        
        h: int
        w: int
        h, w = frame.shape[:2]
        
        # Draw skeleton connections
        for joint1: int, joint2: int in _COCO_CONNECTIONS:
            if joint1 < len(keypoints) and joint2 < len(keypoints):
                pt1: Tuple[int, int] = (
                    int(keypoints[joint1, 0] * w),
                    int(keypoints[joint1, 1] * h)
                )
                pt2: Tuple[int, int] = (
                    int(keypoints[joint2, 0] * w),
                    int(keypoints[joint2, 1] * h)
                )
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoint circles
        for i: int, kp: np.ndarray in enumerate(keypoints):
            if kp[2] > _KEYPOINT_CONFIDENCE_THRESHOLD:
                x: int = int(kp[0] * w)
                y: int = int(kp[1] * h)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        return frame
    
    def draw_fall_alert(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw red border and FALL DETECTED text on the frame.
        
        Args:
            frame: Input frame to add alert overlay.
        
        Returns:
            Frame with red border and alert text.
        """
        # Add red border
        frame = cv2.copyMakeBorder(
            frame,
            _ALERT_BORDER_WIDTH, _ALERT_BORDER_WIDTH,
            _ALERT_BORDER_WIDTH, _ALERT_BORDER_WIDTH,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 255)
        )
        
        # Add alert text
        cv2.putText(
            frame,
            "FALL DETECTED!",
            (_ALERT_TEXT_X, _ALERT_TEXT_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            _ALERT_FONT_SCALE,
            (0, 0, 255),
            _ALERT_TEXT_THICKNESS
        )
        
        return frame
    
    @pyqtSlot()
    def stop(self) -> None:
        """
        Stop the inference thread and release all resources.
        
        Sets running flag to False, clears model references,
        and releases GPU memory if CUDA is in use.
        """
        self.running = False
        self.logger.info("Stopping inference thread...")
        
        # Release model references
        self.yolo_model = None
        self.transformer_model = None
        
        # Clear GPU cache
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        gc.collect()


# ============================================================
# GUI APPLICATION
# ============================================================

class FallDetectionGUI(QMainWindow):
    """
    Main GUI Application for Real-Time Fall Detection.
    
    Features:
        - Real-time video display with keypoint visualization
        - Webcam, video file, and RTSP stream support
        - Color-coded fall detection status indicators
        - Telegram Bot emergency alert integration
    
    Attributes:
        config: Application configuration instance.
        inference_thread: Active VideoInferenceThread or None.
        is_running: Boolean indicating if inference is currently active.
    
    Example:
        >>> app = QApplication([])
        >>> window = FallDetectionGUI()
        >>> window.show()
        >>> app.exec()
    """
    
    def __init__(self) -> None:
        """Initialize the GUI application with default configuration."""
        super().__init__()
        self.config: Config = Config()
        self.logger: logging.Logger = setup_logging()
        
        # Thread management
        self.inference_thread: Optional[VideoInferenceThread] = None
        
        # UI state
        self.is_running: bool = False
        
        # Initialize UI components
        self.init_ui()
    
    def init_ui(self) -> None:
        """Initialize and layout all UI components."""
        self.setWindowTitle("Hybrid Fall Detection System")
        self.setGeometry(
            _WINDOW_X, _WINDOW_Y,
            _WINDOW_WIDTH, _WINDOW_HEIGHT
        )
        self.setStyleSheet(f"background-color: {self.config.COLOR_BG};")
        
        # Central widget
        central_widget: QWidget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout: QHBoxLayout = QHBoxLayout(central_widget)
        main_layout.setSpacing(_MAIN_LAYOUT_SPACING)
        
        # Left panel - Video display
        left_panel: QFrame = self.create_video_panel()
        main_layout.addWidget(left_panel, stretch=3)
        
        # Right panel - Controls and status
        right_panel: QFrame = self.create_control_panel()
        main_layout.addWidget(right_panel, stretch=1)
    
    def create_video_panel(self) -> QFrame:
        """
        Create the video display panel with FPS and frame counters.
        
        Returns:
            QFrame widget containing video label and info bar.
        """
        panel: QFrame = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.config.COLOR_PANEL};
                border-radius: 10px;
                border: 2px solid {self.config.COLOR_ACCENT};
            }}
        """)
        
        layout: QVBoxLayout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title: QLabel = QLabel("Live Video Feed")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(f"color: {self.config.COLOR_TEXT};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Video display label
        self.video_label: QLabel = QLabel()
        self.video_label.setMinimumSize(_VIDEO_LABEL_WIDTH, _VIDEO_LABEL_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #000;
                border-radius: 5px;
                color: #666;
            }}
        """)
        self.video_label.setText("No video source\n\nClick a button to start")
        layout.addWidget(self.video_label)
        
        # Info bar with FPS and frame count
        info_layout: QHBoxLayout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Arial", 10))
        self.fps_label.setStyleSheet(f"color: {self.config.COLOR_TEXT};")
        info_layout.addWidget(self.fps_label)
        
        info_layout.addStretch()
        
        self.frame_count_label = QLabel("Frames: 0")
        self.frame_count_label.setFont(QFont("Arial", 10))
        self.frame_count_label.setStyleSheet(f"color: {self.config.COLOR_TEXT};")
        info_layout.addWidget(self.frame_count_label)
        
        layout.addLayout(info_layout)
        
        return panel
    
    def create_control_panel(self) -> QFrame:
        """
        Create the control panel with buttons and status display.
        
        Returns:
            QFrame widget containing status, buttons, and settings groups.
        """
        panel: QFrame = QFrame()
        panel.setStyleSheet(f"""
            QFrame {{
                background-color: {self.config.COLOR_PANEL};
                border-radius: 10px;
                border: 2px solid {self.config.COLOR_ACCENT};
            }}
        """)
        
        layout: QVBoxLayout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(_CONTROL_LAYOUT_SPACING)
        
        # Title
        title: QLabel = QLabel("Controls")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet(f"color: {self.config.COLOR_TEXT};")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Status display
        status_group: QGroupBox = self.create_status_group()
        layout.addWidget(status_group)
        
        # Control buttons
        buttons_group: QGroupBox = self.create_buttons_group()
        layout.addWidget(buttons_group)
        
        # Settings info
        settings_group: QGroupBox = self.create_settings_group()
        layout.addWidget(settings_group)
        
        layout.addStretch()
        
        return panel
    
    def create_status_group(self) -> QGroupBox:
        """
        Create the detection status display group.
        
        Returns:
            QGroupBox with status label, confidence, and buffer info.
        """
        group: QGroupBox = QGroupBox("Detection Status")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {self.config.COLOR_TEXT};
                border: 1px solid {self.config.COLOR_ACCENT};
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """)
        
        layout: QVBoxLayout = QVBoxLayout()
        
        # Large status label
        self.status_label: QLabel = QLabel("STOPPED")
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {self.config.COLOR_TEXT};
                background-color: {self.config.COLOR_ACCENT};
                border-radius: 10px;
                padding: 20px;
            }}
        """)
        layout.addWidget(self.status_label)
        
        # Confidence display
        self.confidence_label: QLabel = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 12))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet(f"color: {self.config.COLOR_TEXT};")
        layout.addWidget(self.confidence_label)
        
        # Buffer progress
        self.window_label: QLabel = QLabel(f"Buffer: 0/{self.config.WINDOW_SIZE} frames")
        self.window_label.setFont(QFont("Arial", 10))
        self.window_label.setAlignment(Qt.AlignCenter)
        self.window_label.setStyleSheet(f"color: #888;")
        layout.addWidget(self.window_label)
        
        group.setLayout(layout)
        return group
    
    def create_buttons_group(self) -> QGroupBox:
        """
        Create the source control buttons group.
        
        Returns:
            QGroupBox with webcam, video, RTSP, and stop buttons.
        """
        group: QGroupBox = QGroupBox("Source Controls")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {self.config.COLOR_TEXT};
                border: 1px solid {self.config.COLOR_ACCENT};
                border-radius: 5px;
                margin-top: 10px;
            }}
        """)
        
        layout: QVBoxLayout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Start Webcam button
        self.btn_webcam: QPushButton = QPushButton("Start Webcam")
        self.btn_webcam.setFont(QFont("Arial", 11))
        self.btn_webcam.setMinimumHeight(45)
        self.btn_webcam.clicked.connect(self.start_webcam)
        self.style_button(self.btn_webcam, "#3498DB")
        layout.addWidget(self.btn_webcam)
        
        # Load Video button
        self.btn_video: QPushButton = QPushButton("Load Video File")
        self.btn_video.setFont(QFont("Arial", 11))
        self.btn_video.setMinimumHeight(45)
        self.btn_video.clicked.connect(self.load_video)
        self.style_button(self.btn_video, "#9B59B6")
        layout.addWidget(self.btn_video)
        
        # RTSP Stream button
        self.btn_rtsp: QPushButton = QPushButton("Start RTSP Stream")
        self.btn_rtsp.setFont(QFont("Arial", 11))
        self.btn_rtsp.setMinimumHeight(45)
        self.btn_rtsp.clicked.connect(self.start_rtsp)
        self.style_button(self.btn_rtsp, "#1ABC9C")
        layout.addWidget(self.btn_rtsp)
        
        # Stop button
        self.btn_stop: QPushButton = QPushButton("Stop")
        self.btn_stop.setFont(QFont("Arial", 11, QFont.Bold))
        self.btn_stop.setMinimumHeight(45)
        self.btn_stop.clicked.connect(self.stop_inference)
        self.btn_stop.setEnabled(False)
        self.style_button(self.btn_stop, "#E74C3C")
        layout.addWidget(self.btn_stop)
        
        group.setLayout(layout)
        return group
    
    def create_settings_group(self) -> QGroupBox:
        """
        Create the settings/info display group.
        
        Returns:
            QGroupBox showing model paths and configuration parameters.
        """
        group: QGroupBox = QGroupBox("Settings")
        group.setStyleSheet(f"""
            QGroupBox {{
                color: {self.config.COLOR_TEXT};
                border: 1px solid {self.config.COLOR_ACCENT};
                border-radius: 5px;
                margin-top: 10px;
            }}
        """)
        
        layout: QVBoxLayout = QVBoxLayout()
        
        info: QLabel = QLabel(
            f"<b>Models:</b><br>"
            f"• YOLO: {self.config.YOLO_MODEL}<br>"
            f"• Transformer: {self.config.TRANSFORMER_MODEL}<br><br>"
            f"<b>Parameters:</b><br>"
            f"• Window: {self.config.WINDOW_SIZE} frames<br>"
            f"• Threshold: {self.config.FALL_THRESHOLD}<br>"
            f"• Cooldown: {self.config.ALERT_COOLDOWN}s"
        )
        info.setFont(QFont("Arial", 9))
        info.setStyleSheet(f"color: #AAA;")
        layout.addWidget(info)
        
        group.setLayout(layout)
        return group
    
    def style_button(self, button: QPushButton, color: str) -> None:
        """
        Apply styled appearance to a button with hover and pressed states.
        
        Args:
            button: QPushButton widget to style.
            color: Base hex color string (e.g., "#3498DB").
        """
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {self.lighten_color(color)};
                border: 2px solid white;
            }}
            QPushButton:pressed {{
                background-color: {self.darken_color(color)};
            }}
            QPushButton:disabled {{
                background-color: #555;
                color: #888;
            }}
        """)
    
    def lighten_color(self, hex_color: str) -> str:
        """
        Lighten a hex color by adding RGB offset.
        
        Args:
            hex_color: Color string in format "#RRGGBB".
        
        Returns:
            Lightened color string in format "#RRGGBB".
        """
        r: int = int(hex_color[1:3], 16)
        g: int = int(hex_color[3:5], 16)
        b: int = int(hex_color[5:7], 16)
        r = min(255, r + 40)
        g = min(255, g + 40)
        b = min(255, b + 40)
        return f"#{r:02X}{g:02X}{b:02X}"
    
    def darken_color(self, hex_color: str) -> str:
        """
        Darken a hex color by subtracting RGB offset.
        
        Args:
            hex_color: Color string in format "#RRGGBB".
        
        Returns:
            Darkened color string in format "#RRGGBB".
        """
        r: int = int(hex_color[1:3], 16)
        g: int = int(hex_color[3:5], 16)
        b: int = int(hex_color[5:7], 16)
        r = max(0, r - 40)
        g = max(0, g - 40)
        b = max(0, b - 40)
        return f"#{r:02X}{g:02X}{b:02X}"
    
    # ============================================================
    # BUTTON HANDLERS
    # ============================================================
    
    def start_webcam(self) -> None:
        """Start webcam capture and inference."""
        self.stop_inference()
        self.inference_thread = VideoInferenceThread(self.config)
        self.inference_thread.set_source(0, 'webcam')
        self.connect_signals()
        self.inference_thread.start()
        self.update_ui_state(True)
        self.logger.info("Webcam started")
    
    def load_video(self) -> None:
        """Open file dialog to select video file for inference."""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path: str
        _: str
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.stop_inference()
            self.inference_thread = VideoInferenceThread(self.config)
            self.inference_thread.set_source(file_path, 'video')
            self.connect_signals()
            self.inference_thread.start()
            self.update_ui_state(True)
            self.logger.info(f"Video loaded: {file_path}")
    
    def start_rtsp(self) -> None:
        """Start RTSP stream with URL input dialog."""
        from PyQt5.QtWidgets import QInputDialog
        
        rtsp_url: str
        ok: bool
        rtsp_url, ok = QInputDialog.getText(
            self, "RTSP Stream", "Enter RTSP URL:")
        
        if ok and rtsp_url:
            self.stop_inference()
            self.inference_thread = VideoInferenceThread(self.config)
            self.inference_thread.set_source(rtsp_url, 'rtsp')
            self.connect_signals()
            self.inference_thread.start()
            self.update_ui_state(True)
            self.logger.info(f"RTSP stream started: {rtsp_url}")
    
    def stop_inference(self) -> None:
        """Stop the current inference thread if running."""
        if self.inference_thread and self.inference_thread.isRunning():
            self.inference_thread.stop()
            self.inference_thread.wait(2000)  # Wait max 2 seconds
        
        self.update_ui_state(False)
        self.status_label.setText("STOPPED")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {self.config.COLOR_TEXT};
                background-color: {self.config.COLOR_ACCENT};
                border-radius: 10px;
                padding: 20px;
            }}
        """)
        self.logger.info("Inference stopped")
    
    # ============================================================
    # SIGNAL HANDLERS
    # ============================================================
    
    def connect_signals(self) -> None:
        """Connect inference thread signals to GUI slot handlers."""
        self.inference_thread.frame_ready.connect(self.update_frame)
        self.inference_thread.status_update.connect(self.update_status)
        self.inference_thread.fps_update.connect(self.update_fps)
        self.inference_thread.fall_detected.connect(self.on_fall_detected)
        self.inference_thread.buffer_update.connect(self.update_buffer)
        self.inference_thread.confidence_update.connect(self.update_confidence)
        self.inference_thread.error_occurred.connect(self.on_error)
        self.inference_thread.finished.connect(self.on_thread_finished)
    
    @pyqtSlot(np.ndarray)
    def update_frame(self, frame: np.ndarray) -> None:
        """
        Update video display with processed frame.
        
        Args:
            frame: Processed frame with keypoint visualization (BGR).
        """
        rgb_frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h: int
        w: int
        ch: int
        h, w, ch = rgb_frame.shape
        bytes_per_line: int = ch * w
        qt_image: QImage = QImage(
            rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        pixmap: QPixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap: QPixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    @pyqtSlot(str, str)
    def update_status(self, text: str, color: str) -> None:
        """
        Update the status label with new text and color.
        
        Args:
            text: Status text (e.g., "NORMAL", "FALL DETECTED").
            color: Hex color string for background.
        """
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: white;
                background-color: {color};
                border-radius: 10px;
                padding: 20px;
                font-weight: bold;
            }}
        """)
    
    @pyqtSlot(float)
    def update_fps(self, fps: float) -> None:
        """
        Update the FPS display label.
        
        Args:
            fps: Current frames per second value.
        """
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    @pyqtSlot(int, int)
    def update_buffer(self, current: int, maximum: int) -> None:
        """
        Update the buffer progress display.
        
        Args:
            current: Current number of frames in buffer.
            maximum: Maximum buffer capacity.
        """
        self.window_label.setText(f"Buffer: {current}/{maximum} frames")
    
    @pyqtSlot(float)
    def update_confidence(self, probability: float) -> None:
        """
        Update the confidence display label.
        
        Args:
            probability: Fall probability (0.0-1.0).
        """
        self.confidence_label.setText(f"Confidence: {probability:.2%}")
    
    @pyqtSlot(bool)
    def on_fall_detected(self, detected: bool) -> None:
        """
        Handle fall detection state change event.
        
        Args:
            detected: True if fall is currently detected.
        """
        if detected:
            self.logger.warning("FALL DETECTED!")
    
    @pyqtSlot(str)
    def on_error(self, message: str) -> None:
        """
        Handle error from inference thread.
        
        Args:
            message: Error description string.
        """
        self.logger.error(f"Inference error: {message}")
        QMessageBox.critical(self, "Error", message)
        self.stop_inference()
    
    @pyqtSlot()
    def on_thread_finished(self) -> None:
        """Handle inference thread completion."""
        self.update_ui_state(False)
        self.logger.info("Inference thread finished")
    
    def update_ui_state(self, running: bool) -> None:
        """
        Update button enabled/disabled states based on running status.
        
        Args:
            running: True if inference is active, False otherwise.
        """
        self.is_running = running
        self.btn_webcam.setEnabled(not running)
        self.btn_video.setEnabled(not running)
        self.btn_rtsp.setEnabled(not running)
        self.btn_stop.setEnabled(running)
    
    # ============================================================
    # WINDOW EVENTS
    # ============================================================
    
    def closeEvent(self, event) -> None:
        """
        Handle window close event to ensure cleanup.
        
        Args:
            event: Close event from Qt.
        """
        self.stop_inference()
        event.accept()


# ============================================================
# ENTRY POINT
# ============================================================

def run_app() -> int:
    """
    Entry point for GUI application - compatible with main.py.
    
    Launches the PyQt5 real-time fall detection GUI with full
    YOLOv11-Pose + Transformer inference pipeline.
    
    Returns:
        Application exit code from Qt event loop.
    """
    app: QApplication = QApplication([])
    app.setApplicationName("Fall Detection System")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window: FallDetectionGUI = FallDetectionGUI()
    window.show()
    
    # Run Qt event loop
    exit_code: int = app.exec()
    
    # Cleanup after exit
    gc.collect()
    
    return exit_code


def main() -> int:
    """
    Main entry point for standalone script execution.
    
    Returns:
        Exit code from run_app().
    """
    return run_app()


if __name__ == "__main__":
    exit(main())
