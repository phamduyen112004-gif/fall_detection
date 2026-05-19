"""
Fall Detection Application - Real-time GUI
Using PyQt5, OpenCV, and YOLOv11-Pose
"""

import sys
import os
import cv2
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QMessageBox, QInputDialog, QFrame, QTextEdit
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# YOLOv11 imports
from ultralytics import YOLO

# Try to import torch for the transformer's P_fall calculation
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PerformanceMetrics:
    """Track and calculate real-time performance metrics for evaluation."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        
        # FPS tracking
        self.fps_history = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        
        # Processing time breakdown
        self.inference_times = deque(maxlen=window_size)
        self.preprocessing_times = deque(maxlen=window_size)
        self.postprocessing_times = deque(maxlen=window_size)
        self.drawing_times = deque(maxlen=window_size)
        
        # Overall statistics
        self.total_frames = 0
        self.total_inference_time = 0
        self.start_time = None
        self.end_time = None
        
        # Fall detection stats
        self.total_falls_detected = 0
        self.fall_detection_times = []
    
    def start(self):
        """Start tracking."""
        import time
        self.start_time = time.time()
        self.total_frames = 0
        self.total_inference_time = 0
    
    def stop(self):
        """Stop tracking."""
        import time
        self.end_time = time.time()
    
    def add_frame(self, fps, inference_time=0, preprocessing_time=0, 
                  postprocessing_time=0, drawing_time=0):
        """Add metrics for a single frame."""
        self.fps_history.append(fps)
        self.frame_times.append(1.0 / fps if fps > 0 else 0)
        self.inference_times.append(inference_time)
        self.preprocessing_times.append(preprocessing_time)
        self.postprocessing_times.append(postprocessing_time)
        self.drawing_times.append(drawing_time)
        self.total_frames += 1
        self.total_inference_time += inference_time
    
    def record_fall(self, p_fall, timestamp):
        """Record a fall detection event."""
        self.total_falls_detected += 1
        self.fall_detection_times.append((timestamp, p_fall))
    
    def get_current_fps(self):
        """Get current FPS."""
        return self.fps_history[-1] if self.fps_history else 0
    
    def get_average_fps(self):
        """Get average FPS over window."""
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def get_min_fps(self):
        """Get minimum FPS."""
        return np.min(self.fps_history) if self.fps_history else 0
    
    def get_max_fps(self):
        """Get maximum FPS."""
        return np.max(self.fps_history) if self.fps_history else 0
    
    def get_fps_std(self):
        """Get FPS standard deviation."""
        return np.std(self.fps_history) if len(self.fps_history) > 1 else 0
    
    def get_average_latency(self):
        """Get average frame latency (ms)."""
        return np.mean(self.frame_times) * 1000 if self.frame_times else 0
    
    def get_average_inference_time(self):
        """Get average inference time (ms)."""
        return np.mean(self.inference_times) * 1000 if self.inference_times else 0
    
    def get_total_processing_breakdown(self):
        """Get total time spent in each processing stage."""
        total_inference = sum(self.inference_times) * 1000
        total_preprocessing = sum(self.preprocessing_times) * 1000
        total_postprocessing = sum(self.postprocessing_times) * 1000
        total_drawing = sum(self.drawing_times) * 1000
        
        total = total_inference + total_preprocessing + total_postprocessing + total_drawing
        
        return {
            'inference': total_inference,
            'preprocessing': total_preprocessing,
            'postprocessing': total_postprocessing,
            'drawing': total_drawing,
            'total': total,
            'fps_breakdown': {
                'inference_pct': (total_inference / total * 100) if total > 0 else 0,
                'preprocessing_pct': (total_preprocessing / total * 100) if total > 0 else 0,
                'postprocessing_pct': (total_postprocessing / total * 100) if total > 0 else 0,
                'drawing_pct': (total_drawing / total * 100) if total > 0 else 0,
            }
        }
    
    def get_real_time_factor(self, target_fps=30):
        """Calculate real-time factor (RTF). RTF > 1 means real-time capable."""
        avg_fps = self.get_average_fps()
        return avg_fps / target_fps if target_fps > 0 else 0
    
    def get_summary(self):
        """Get complete performance summary for report."""
        duration = (self.end_time - self.start_time) if (self.end_time and self.start_time) else 0
        
        return {
            'total_frames': self.total_frames,
            'duration_seconds': duration,
            'average_fps': self.get_average_fps(),
            'min_fps': self.get_min_fps(),
            'max_fps': self.get_max_fps(),
            'fps_std': self.get_fps_std(),
            'average_latency_ms': self.get_average_latency(),
            'average_inference_time_ms': self.get_average_inference_time(),
            'real_time_factor': self.get_real_time_factor(),
            'total_falls_detected': self.total_falls_detected,
            'throughput_fps': self.total_frames / duration if duration > 0 else 0,
        }
    
    def generate_report(self):
        """Generate a formatted report string for the thesis."""
        summary = self.get_summary()
        breakdown = self.get_total_processing_breakdown()
        
        report = f"""
================================================================================
                    REAL-TIME PERFORMANCE EVALUATION REPORT
================================================================================

1. OVERALL STATISTICS
--------------------------------------------------------------------------------
   Total Frames Processed:     {summary['total_frames']}
   Total Duration:            {summary['duration_seconds']:.2f} seconds
   Average FPS:               {summary['average_fps']:.2f} fps
   Min FPS:                   {summary['min_fps']:.2f} fps
   Max FPS:                   {summary['max_fps']:.2f} fps
   FPS Standard Deviation:    {summary['fps_std']:.2f}
   Throughput:                {summary['throughput_fps']:.2f} frames/second

2. LATENCY ANALYSIS
--------------------------------------------------------------------------------
   Average Frame Latency:     {summary['average_latency_ms']:.2f} ms
   Average Inference Time:    {summary['average_inference_time_ms']:.2f} ms

3. REAL-TIME CAPABILITY
--------------------------------------------------------------------------------
   Real-Time Factor (RTF):    {summary['real_time_factor']:.3f}
   Target FPS:                30 fps
   Status:                    {"REAL-TIME CAPABLE" if summary['real_time_factor'] >= 1.0 else "NOT REAL-TIME CAPABLE"}

4. PROCESSING TIME BREAKDOWN
--------------------------------------------------------------------------------
   Inference:                 {breakdown['inference']:.2f} ms total
   Preprocessing:             {breakdown['preprocessing']:.2f} ms total
   Postprocessing:            {breakdown['postprocessing']:.2f} ms total
   Drawing/Overlay:            {breakdown['drawing']:.2f} ms total

   Percentage Breakdown:
     - Inference:             {breakdown['fps_breakdown']['inference_pct']:.1f}%
     - Preprocessing:          {breakdown['fps_breakdown']['preprocessing_pct']:.1f}%
     - Postprocessing:         {breakdown['fps_breakdown']['postprocessing_pct']:.1f}%
     - Drawing/Overlay:        {breakdown['fps_breakdown']['drawing_pct']:.1f}%

5. FALL DETECTION STATISTICS
--------------------------------------------------------------------------------
   Total Falls Detected:      {summary['total_falls_detected']}

================================================================================
"""
        return report


class VideoThread(QThread):
    """Thread for video capture and YOLO inference."""
    
    # Signals to communicate with main GUI thread
    frame_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)
    pfall_signal = pyqtSignal(float)
    keypoints_signal = pyqtSignal(int)
    metrics_signal = pyqtSignal(dict)
    report_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.source = 0  # 0 = webcam, or video file path, or rtsp URL
        self.model = None
        self.fps = 0.0
        self.p_fall = 0.0
        self.keypoints_count = 0
        
        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.track_detailed_metrics = True
        self.show_fps_overlay = True
        
        # VRAM tracking
        self.peak_vram_mb = 0.0
        
        # Calculate static model profile (GFLOPs & Params) on init
        self.model_params = 0
        self.model_gflops = 0
        self.calculate_model_profile()
    
    def calculate_model_profile(self):
        """
        Calculate static model profile using thop library.
        Measures Parameters and GFLOPs of the placeholder Transformer model.
        """
        if not TORCH_AVAILABLE:
            print("[INFO] Torch not available, skipping model profile calculation")
            return
        
        try:
            from thop import profile
            print("\n" + "="*60)
            print("  CALCULATING MODEL PROFILE (GFLOPs & Params)")
            print("="*60)
            
            # Create a dummy placeholder Transformer model for profiling
            class DummyTransformer(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    # Simulate a small Transformer for fall detection
                    self.embedding = torch.nn.Linear(60, 128)
                    encoder_layer = torch.nn.TransformerEncoderLayer(
                        d_model=128, nhead=4, batch_first=True, dim_feedforward=256
                    )
                    self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
                    self.fc = torch.nn.Sequential(
                        torch.nn.Linear(128, 64),
                        torch.nn.ReLU(),
                        torch.nn.Linear(64, 2)
                    )
                
                def forward(self, x):
                    x = self.embedding(x)
                    x = self.transformer(x)
                    x = x.mean(dim=1)
                    return self.fc(x)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            dummy_model = DummyTransformer().to(device)
            dummy_model.eval()
            
            # Create dummy input tensor (batch=1, seq=60, features=60)
            dummy_input = torch.randn(1, 60, 60).to(device)
            
            # Calculate GFLOPs and Params
            macs, params = profile(dummy_model, inputs=(dummy_input,), verbose=False)
            gflops = (macs * 2) / 1e9  # Convert MACs to GFLOPs
            
            self.model_params = params
            self.model_gflops = gflops
            
            print(f"  Model Type:       DummyTransformer (placeholder)")
            print(f"  Input Shape:      (1, 60, 60)")
            print(f"  Parameters:       {params / 1e6:.4f} M ({params:,.0f})")
            print(f"  GFLOPs:           {gflops:.4f} GFLOPs")
            print(f"  Device:           {device}")
            print("="*60 + "\n")
            
            # Clean up
            del dummy_model, dummy_input
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except ImportError:
            print("[WARNING] thop not installed. Run: pip install thop")
            print("[INFO] Skipping GFLOPs/Params calculation")
        except Exception as e:
            print(f"[WARNING] Could not calculate model profile: {e}")
        
    def load_model(self):
        """Load YOLOv11-Pose model."""
        try:
            # Use yolo11l-pose (large model - more accurate)
            self.model = YOLO("yolo11l-pose.pt")
            print("[INFO] YOLOv11l-Pose model loaded successfully")
        except Exception as e:
            print(f"[WARNING] Could not load YOLOv11l-Pose, trying yolo11n-pose: {e}")
            try:
                self.model = YOLO("yolo11n-pose.pt")
                print("[INFO] YOLOv11n-Pose model loaded successfully")
            except Exception as e2:
                print(f"[WARNING] Could not load yolo11n-pose, trying yolov8l-pose: {e2}")
                try:
                    self.model = YOLO("yolov8l-pose.pt")
                    print("[INFO] YOLOv8l-Pose model loaded successfully")
                except Exception as e3:
                    print(f"[ERROR] Could not load pose model: {e3}")
                    self.model = None
    
    def set_source(self, source):
        """Set video source (0=webcam, path=video file, rtsp=stream)."""
        self.source = source
    
    def run(self):
        """Main loop for video capture and inference."""
        import time
        
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            print("[ERROR] No model available, thread exiting")
            return
        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            return
        
        # Reset CUDA memory stats at the start
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        self.running = True
        frame_count = 0
        fps_start_time = cv2.getTickCount()
        
        # Start performance tracking
        self.metrics.start()
        
        # Warm-up counter
        warmup_frames = 30
        warmup_done = False
        
        print("\n" + "="*60)
        print("  STARTING REAL-TIME INFERENCE")
        print("="*60)
        print(f"  Warm-up frames: {warmup_frames}")
        print(f"  Logging interval: every 30 frames")
        print("="*60 + "\n")
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Try to reopen for streams
                if isinstance(self.source, str) and self.source.startswith(('rtsp', 'http')):
                    cap.release()
                    cap = cv2.VideoCapture(self.source)
                    if cap.isOpened():
                        continue
                break
            
            # Sync CUDA before timing (critical for accurate GPU measurement)
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            frame_start = time.time()
            frame_count += 1
            
            # ========================================
            # A. PREPROCESSING (YOLO detection)
            # ========================================
            pre_start = time.time()
            
            # Run YOLO inference
            results = self.model(frame, verbose=False, conf=0.5, kpt=True)
            
            # Extract keypoints
            keypoints_data = []
            for result in results:
                if result.keypoints is not None and len(result.keypoints) > 0:
                    kpts = result.keypoints.data[0]
                    for kpt in kpts:
                        x, y, conf = kpt[0].item(), kpt[1].item(), kpt[2].item()
                        keypoints_data.append((x, y, conf))
            
            valid_kpts = [k for k in keypoints_data if k[2] > 0.5]
            num_valid_kpts = len(valid_kpts)
            
            # Sync after preprocessing
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
            pre_end = time.time()
            pre_ms = (pre_end - pre_start) * 1000
            
            # ========================================
            # B. INFERENCE (Transformer / P_fall calculation)
            # ========================================
            inf_start = time.time()
            
            # Extract keypoints dict for P_fall
            if num_valid_kpts >= 10:
                coco_names = [
                    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
                ]
                keypoints_dict = {}
                for i, name in enumerate(coco_names):
                    if i < len(keypoints_data):
                        x, y, conf = keypoints_data[i]
                        keypoints_dict[name] = (int(x), int(y), conf)
                
                p_fall = self.calculate_p_fall(keypoints_dict)
            else:
                p_fall = 0.0
            
            # Sync after inference
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
            inf_end = time.time()
            inf_ms = (inf_end - inf_start) * 1000
            
            # ========================================
            # C. POSTPROCESSING (Draw boxes, emit signals)
            # ========================================
            post_start = time.time()
            
            # Draw skeleton and info
            if num_valid_kpts >= 10:
                processed_frame = self._draw_skeleton(frame.copy(), valid_kpts, p_fall, num_valid_kpts)
            else:
                processed_frame = self._draw_insufficient_keypoints(frame.copy(), num_valid_kpts)
            
            # Add FPS overlay
            if self.show_fps_overlay:
                cv2.putText(processed_frame, f"FPS: {self.fps:.1f}", 
                           (10, processed_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            post_end = time.time()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.synchronize()
            post_ms = (post_end - post_start) * 1000
            
            # ========================================
            # Calculate FPS
            # ========================================
            frame_end = time.time()
            fps_end_time = cv2.getTickCount()
            elapsed = (fps_end_time - fps_start_time) / cv2.getTickCountFrequency()
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = cv2.getTickCount()
            
            # Update peak VRAM
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # ========================================
            # TERMINAL LOGGING (every 30 frames, after warm-up)
            # ========================================
            if frame_count > warmup_frames and frame_count % 30 == 0:
                vram_str = f"{self.peak_vram_mb:.1f}" if TORCH_AVAILABLE and torch.cuda.is_available() else "N/A"
                print(f"[Frame {frame_count}] Pre={pre_ms:.1f}ms | Infer={inf_ms:.1f}ms | Post={post_ms:.1f}ms | VRAM: {vram_str} MB")
            
            if frame_count == warmup_frames and not warmup_done:
                print(f"[INFO] Warm-up complete! Starting performance measurement...")
                warmup_done = True
            
            # Emit signals to update GUI
            self.frame_signal.emit(processed_frame)
            self.fps_signal.emit(self.fps)
            self.pfall_signal.emit(p_fall)
            self.keypoints_signal.emit(num_valid_kpts)
        
        # Stop tracking and generate report
        self.metrics.stop()
        
        # Final report
        if self.track_detailed_metrics:
            print("\n" + "="*60)
            print("  FINAL PERFORMANCE REPORT")
            print("="*60)
            
            summary = self.metrics.get_summary()
            breakdown = self.metrics.get_total_processing_breakdown()
            
            print(f"  Total Frames:       {summary['total_frames']}")
            print(f"  Duration:           {summary['duration_seconds']:.2f}s")
            print(f"  Average FPS:        {summary['average_fps']:.2f}")
            print(f"  Min/Max FPS:       {summary['min_fps']:.2f} / {summary['max_fps']:.2f}")
            print(f"  Average Latency:   {summary['average_latency_ms']:.2f} ms")
            print(f"  Peak VRAM:         {self.peak_vram_mb:.2f} MB")
            print(f"  RTF @30 FPS:       {summary['real_time_factor']:.3f}")
            
            if self.model_params > 0:
                print(f"  Model Params:       {self.model_params / 1e6:.4f} M")
                print(f"  Model GFLOPs:       {self.model_gflops:.4f} GFLOPs")
            
            print("="*60 + "\n")
            
            report = self.metrics.generate_report()
            self.report_signal.emit(report)
        
        cap.release()
        self.running = False
        self.finished_signal.emit()
    
    def _extract_keypoints_dict(self, kpts):
        """Convert COCO keypoints to dictionary."""
        coco_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        keypoints_dict = {}
        for i, name in enumerate(coco_names):
            if i < len(kpts):
                x, y, conf = kpts[i][0].item(), kpts[i][1].item(), kpts[i][2].item()
                keypoints_dict[name] = (int(x), int(y), conf)
        
        return keypoints_dict
    
    def _draw_skeleton(self, frame, valid_kpts, p_fall, keypoints_count):
        """Draw skeleton, bbox, and info on frame."""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Get all keypoint coordinates
        kpt_coords = [(kp[0], kp[1]) for kp in valid_kpts]
        
        if len(kpt_coords) < 2:
            return output
        
        # Calculate bounding box from keypoints
        x_coords = [p[0] for p in kpt_coords]
        y_coords = [p[1] for p in kpt_coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        # Color based on fall status (using dynamic threshold)
        if p_fall > self.FALL_THRESHOLD and keypoints_count >= 10:
            bbox_color = (0, 0, 255)  # Red
            text_color = (0, 0, 255)
        else:
            bbox_color = (0, 255, 0)  # Green
            text_color = (255, 255, 255)
        
        # Draw bounding box
        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), bbox_color, 2)
        
        # COCO skeleton connections
        skeleton = [
            (5, 6),   # shoulders
            (5, 7), (7, 9),     # left arm
            (6, 8), (8, 10),    # right arm
            (5, 11), (6, 12),   # torso
            (11, 12),           # hips
            (11, 13), (13, 15), # left leg
            (12, 14), (14, 16)  # right leg
        ]
        
        # Draw skeleton connections
        for idx, (start_idx, end_idx) in enumerate(skeleton):
            if start_idx < len(valid_kpts) and end_idx < len(valid_kpts):
                pt1 = (valid_kpts[start_idx][0], valid_kpts[start_idx][1])
                pt2 = (valid_kpts[end_idx][0], valid_kpts[end_idx][1])
                cv2.line(output, pt1, pt2, (255, 255, 0), 2)
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(valid_kpts):
            cv2.circle(output, (x, y), 4, (0, 255, 255), -1)
            cv2.circle(output, (x, y), 4, (0, 0, 0), 1)
        
        # Draw info text on frame
        info_text = f"Keypoints: {keypoints_count}/17"
        cv2.putText(output, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        pfall_text = f"P_fall: {p_fall*100:.1f}%"
        cv2.putText(output, pfall_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # FALL ALERT: Big red text at top center (using dynamic threshold)
        if p_fall > self.FALL_THRESHOLD and keypoints_count >= 10:
            alert_text = "FALL DETECTED!"
            font_scale = 1.5
            thickness = 3
            
            # Get text size for centering
            (text_width, text_height), baseline = cv2.getTextSize(
                alert_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            text_x = (w - text_width) // 2
            text_y = 80
            
            # Draw red background rectangle
            cv2.rectangle(output, 
                         (text_x - 10, text_y - text_height - 10),
                         (text_x + text_width + 10, text_y + baseline + 10),
                         (0, 0, 255), -1)
            
            # Draw white text
            cv2.putText(output, alert_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return output
    
    def _draw_insufficient_keypoints(self, frame, keypoints_count):
        """Draw message when not enough keypoints detected."""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Draw warning
        text = f"Keypoints: {keypoints_count}/17 (Need >=10)"
        cv2.putText(output, text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        return output
    
    def calculate_p_fall(self, keypoints):
        """
        Calculate fall probability from keypoints.
        
        This is a PLACEHOLDER method - replace with your PyTorch Transformer model.
        
        Current implementation uses simple heuristics:
        - Body angle analysis
        - Height-to-width ratio
        - Vertical position analysis
        """
        if not keypoints or len(keypoints) < 10:
            return 0.0
        
        # Check required keypoints
        required = ['left_shoulder', 'right_shoulder', 
                   'left_hip', 'right_hip',
                   'left_knee', 'right_knee',
                   'left_ankle', 'right_ankle']
        
        missing = [k for k in required if k not in keypoints]
        if missing:
            return 0.0
        
        prob = 0.0
        
        # 1. Calculate body angle (torso)
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        left_hip = keypoints['left_hip']
        right_hip = keypoints['right_hip']
        
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        
        # Angle from vertical
        dx = shoulder_center_x - hip_center_x
        dy = shoulder_center_y - hip_center_y
        
        if abs(dy) > 0:
            angle = abs(np.arctan2(dx, dy) * 180 / np.pi)
            
            # Body tilted significantly
            if angle > 45:
                prob += 0.4 * (angle / 90)
            if angle > 60:
                prob += 0.2
        
        # 2. Check if ankles are near shoulder height (lying down)
        left_ankle = keypoints['left_ankle']
        right_ankle = keypoints['right_ankle']
        
        ankle_avg_y = (left_ankle[1] + right_ankle[1]) / 2
        shoulder_avg_y = (left_shoulder[1] + right_shoulder[1])
        
        # If ankles are at similar height or below shoulders
        if ankle_avg_y > shoulder_avg_y - 100:
            prob += 0.3
        
        # 3. Body aspect ratio (lying down = wider than tall)
        body_width = abs(right_shoulder[0] - left_shoulder[0])
        body_height = abs(hip_center_y - shoulder_center_y)
        
        if body_height > 0:
            ratio = body_width / body_height
            if ratio > 1.5:
                prob += 0.2
        
        # 4. Knee positions (are legs bent/straight out?)
        left_knee = keypoints['left_knee']
        right_knee = keypoints['right_knee']
        
        # If knees are lower than hips (crouching/laying)
        if left_knee[1] > left_hip[1] - 20 or right_knee[1] > right_hip[1] - 20:
            prob += 0.1
        
        return min(1.0, prob)
    
    def stop(self):
        """Stop the thread."""
        self.running = False


class FallDetectionGUI(QMainWindow):
    """Main GUI window for Fall Detection."""

    # Default threshold (used if config not found)
    DEFAULT_FALL_THRESHOLD = 0.5
    DEFAULT_MODEL_DIR = "models"

    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.current_fps = 0.0
        self.current_pfall = 0.0
        self.current_keypoints = 0

        # Load optimal threshold from config
        self.FALL_THRESHOLD = self._load_threshold()

        self.init_ui()

    def _load_threshold(self) -> float:
        """Load optimal threshold from threshold_config.json."""
        import json
        import os

        # Try multiple possible config locations
        possible_paths = [
            os.path.join(self.DEFAULT_MODEL_DIR, "threshold_config.json"),
            os.path.join(self.DEFAULT_MODEL_DIR, "best_model_fold0", "threshold_config.json"),
            "threshold_config.json",
        ]

        for config_path in possible_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    threshold = config.get('optimal_threshold', self.DEFAULT_FALL_THRESHOLD)
                    print(f"[INFO] Loaded threshold {threshold:.4f} from {config_path}")
                    return threshold
                except Exception as e:
                    print(f"[WARNING] Failed to load threshold from {config_path}: {e}")

        print(f"[INFO] Using default threshold {self.DEFAULT_FALL_THRESHOLD:.4f}")
        return self.DEFAULT_FALL_THRESHOLD
    
    def init_ui(self):
        """Initialize the GUI."""
        self.setWindowTitle("Fall Detection - YOLOv11-Pose")
        self.setFixedSize(660, 620)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #ffffff; }
            QPushButton {
                background-color: #16213e;
                color: #00d9ff;
                border: 1px solid #00d9ff;
                padding: 8px 16px;
                font-size: 12px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #0f3460;
            }
            QPushButton:pressed {
                background-color: #00d9ff;
                color: #1a1a2e;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("REAL-TIME FALL DETECTION")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont("Arial", 14, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #00d9ff;")
        main_layout.addWidget(title_label)
        
        # Alert label (hidden by default)
        self.alert_label = QLabel("")
        self.alert_label.setAlignment(Qt.AlignCenter)
        self.alert_label.setStyleSheet("color: red; font-size: 18px; font-weight: bold;")
        main_layout.addWidget(self.alert_label)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setFrameShape(QFrame.Box)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Chua co video")
        main_layout.addWidget(self.video_label)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.upload_video)
        
        self.webcam_btn = QPushButton("Start Webcam")
        self.webcam_btn.clicked.connect(self.start_webcam)
        
        self.stream_btn = QPushButton("Open Stream Link")
        self.stream_btn.clicked.connect(self.open_stream_link)
        
        self.report_btn = QPushButton("Show Report")
        self.report_btn.clicked.connect(self.show_report)
        self.report_btn.setEnabled(False)
        
        button_layout.addWidget(self.upload_btn)
        button_layout.addWidget(self.webcam_btn)
        button_layout.addWidget(self.stream_btn)
        button_layout.addWidget(self.report_btn)
        
        main_layout.addLayout(button_layout)
        
        # Status labels
        self.fps_label = QLabel("FPS: 0.0")
        self.pfall_label = QLabel("P_fall: 0.00%")
        self.keypoints_label = QLabel("Keypoints: 0/17")
        self.latency_label = QLabel("Latency: 0 ms")
        
        main_layout.addWidget(self.fps_label)
        main_layout.addWidget(self.pfall_label)
        main_layout.addWidget(self.keypoints_label)
        main_layout.addWidget(self.latency_label)
        
        # Detailed metrics
        self.detailed_metrics_label = QLabel("")
        main_layout.addWidget(self.detailed_metrics_label)
    
        # Store last report
        self.last_report = ""
    
    def show_report(self):
        """Display performance report in a dialog."""
        if self.last_report:
            from PyQt5.QtWidgets import QDialog
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Performance Report")
            dialog.setMinimumSize(600, 500)
            
            layout = QVBoxLayout(dialog)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setPlainText(self.last_report)
            text_edit.setStyleSheet("""
                QTextEdit {
                    background-color: #0f0f23;
                    color: #00ff88;
                    font-family: 'Courier New';
                    font-size: 11px;
                }
            """)
            
            layout.addWidget(text_edit)
            
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.exec_()
    
    def upload_video(self):
        """Open file dialog to select video."""
        from PyQt5.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.flv);;All files (*)"
        )
        
        if file_path:
            self.start_video_source(file_path)
    
    def start_webcam(self):
        """Start webcam capture."""
        self.start_video_source(0)
    
    def open_stream_link(self):
        """Show input dialog for RTSP/HTTP stream URL."""
        stream_url, ok = QInputDialog.getText(
            self, "Stream URL", "Enter RTSP/HTTP stream URL:")
        
        if ok and stream_url:
            self.start_video_source(stream_url)
    
    def start_video_source(self, source):
        """Start video thread with given source."""
        # Stop existing thread
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        
        # Create new thread
        self.video_thread = VideoThread()
        self.video_thread.set_source(source)
        
        # Connect signals
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.pfall_signal.connect(self.update_pfall)
        self.video_thread.keypoints_signal.connect(self.update_keypoints)
        self.video_thread.metrics_signal.connect(self.update_metrics)
        self.video_thread.report_signal.connect(self.handle_report)
        self.video_thread.finished_signal.connect(self.handle_thread_finished)
        
        # Start thread
        self.video_thread.start()
        
        # Update button states
        self.webcam_btn.setEnabled(False)
        self.upload_btn.setEnabled(False)
        self.stream_btn.setEnabled(False)
        self.report_btn.setEnabled(False)
    
    def handle_report(self, report):
        """Store the performance report."""
        self.last_report = report
        self.report_btn.setEnabled(True)
    
    def handle_thread_finished(self):
        """Handle thread finishing."""
        self.webcam_btn.setEnabled(True)
        self.upload_btn.setEnabled(True)
        self.stream_btn.setEnabled(True)
    
    def update_frame(self, frame):
        """Update video display with new frame."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to label size
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)
    
    def update_fps(self, fps):
        """Update FPS display."""
        self.current_fps = fps
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_pfall(self, pfall):
        """Update P_fall display."""
        self.current_pfall = pfall
        self.pfall_label.setText(f"P_fall: {pfall*100:.2f}%")

        # Update alert label (using dynamic threshold)
        if pfall > self.FALL_THRESHOLD and self.current_keypoints >= 10:
            self.alert_label.setText("FALL DETECTED!")
        else:
            self.alert_label.setText("")
    
    def update_keypoints(self, count):
        """Update keypoints display."""
        self.current_keypoints = count
        self.keypoints_label.setText(f"Keypoints: {count}/17")
    
    def update_metrics(self, metrics):
        """Update detailed metrics display."""
        avg_fps = metrics.get('average_fps', 0)
        avg_latency = metrics.get('average_latency_ms', 0)
        
        self.latency_label.setText(f"Latency: {avg_latency:.1f} ms")
        
        self.detailed_metrics_label.setText(
            f"Avg FPS: {avg_fps:.1f} | Min: {metrics.get('min_fps', 0):.1f} | "
            f"Max: {metrics.get('max_fps', 0):.1f} | RTF: {metrics.get('real_time_factor', 0):.2f}"
        )
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = FallDetectionGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Fall Detection GUI')
    parser.add_argument('--video', '-v', type=str, default='0',
                       help='Video path or 0 for webcam, rtsp:// for stream')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = FallDetectionGUI()
    window.show()
    
    # Bắt đầu video ngay khi khởi động nếu có source
    if args.video:
        window.start_video_source(args.video)
    
    sys.exit(app.exec_())
