#!/usr/bin/env python3
"""
Pose-Informed Fall Recognition (PIFR) Module.

Geometric Feature Extraction from COCO 17-Keypoint Data.

Tác giả: Fall Detection Team
Ngày: May 2026

Mô tả:
    Module trích xuất 9 đặc trưng hình học từ 17 COCO keypoints
    và kết hợp với flattened keypoints để tạo vector đặc trưng 60 chiều.

COCO Keypoint Indices:
    0:  NOSE,       1:  L_EYE,      2:  R_EYE,      3:  L_EAR,  4:  R_EAR
    5:  L_SHOULDER, 6:  R_SHOULDER, 7:  L_ELBOW,     8:  R_ELBOW,
    9:  L_WRIST,    10: R_WRIST,    11: L_HIP,       12: R_HIP,
    13: L_KNEE,    14: R_KNEE,     15: L_ANKLE,     16: R_ANKLE

Đặc trưng hình học (9 chiều):
    [0] center_mass_x       - Trọng tâm X của các keypoints hợp lệ
    [1] center_mass_y       - Trọng tâm Y của các keypoints hợp lệ
    [2] shoulder_nose_angle - Góc vai-mũi (B-A và B-C)
    [3] torso_angle         - Góc thân so với trục dọc
    [4] hip_angle           - Góc hông so với trục ngang
    [5] shoulder_angle      - Góc vai so với trục ngang
    [6] left_leg_angle      - Góc chân trái (đầu gối)
    [7] right_leg_angle    - Góc chân phải (đầu gối)
    [8] nose_to_ankle_angle - Góc mũi-mắt cá chân so với trục dọc

Vector đặc trưng đầu ra (60 chiều):
    [0:51]   - 17 keypoints × 3 (x, y, conf) = 51 chiều
    [51:60]  - 9 geometric features

Sử dụng:
    >>> from pifr_features import GeometricFeatureExtractor
    >>> extractor = GeometricFeatureExtractor()
    >>> keypoints = np.random.rand(17, 3)  # shape: (17, 3)
    >>> features = extractor.extract(keypoints)  # shape: (60,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Constants
SEQ_LEN = 60      # Number of frames in sequence
FEATURE_DIM = 60  # Feature dimension (51 keypoints + 9 geometric)


class GeometricFeatureExtractor:
    """
    Pose-Informed Fall Recognition (PIFR) Geometric Feature Extractor.

    Trích xuất 9 đặc trưng hình học từ 17 COCO keypoints và kết hợp
    với flattened keypoints để tạo vector đặc trưng 60 chiều.

    Thuộc tính:
        CONF_THRESHOLD (float): Ngưỡng confidence tối thiểu cho keypoint hợp lệ.
        EPS (float): Giá trị epsilon để tránh chia cho zero.
        FEATURE_DIM (int): Số chiều vector đặc trưng (60).
        KEYPOINT_NAMES (list): Tên các keypoints theo chỉ mục COCO.

    Ví dụ:
        >>> extractor = GeometricFeatureExtractor(conf_threshold=0.2)
        >>> keypoints = np.random.rand(17, 3)
        >>> features = extractor.extract(keypoints)
        >>> print(features.shape)
        (60,)
    """

    # COCO 17 Keypoint Indices
    NOSE = 0
    L_EYE = 1
    R_EYE = 2
    L_EAR = 3
    R_EAR = 4
    L_SHOULDER = 5
    R_SHOULDER = 6
    L_ELBOW = 7
    R_ELBOW = 8
    L_WRIST = 9
    R_WRIST = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16

    KEYPOINT_NAMES = [
        "nose", "l_eye", "r_eye", "l_ear", "r_ear",
        "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
        "l_wrist", "r_wrist", "l_hip", "r_hip",
        "l_knee", "r_knee", "l_ankle", "r_ankle",
    ]

    def __init__(
        self,
        conf_threshold: float = 0.2,
        eps: float = 1e-6,
        normalize: bool = True,
    ) -> None:
        """
        Khởi tạo GeometricFeatureExtractor.

        Args:
            conf_threshold: Ngưỡng confidence tối thiểu cho keypoint hợp lệ.
                           Mặc định: 0.2.
            eps: Giá trị epsilon để tránh chia cho zero.
                 Mặc định: 1e-6.
            normalize: Nếu True, chuẩn hóa Min-Max các đặc trưng góc về [0, 1].
                       Mặc định: True.

        Ví dụ:
            >>> extractor = GeometricFeatureExtractor(conf_threshold=0.3, normalize=False)
        """
        self.conf_threshold = conf_threshold
        self.eps = eps
        self.normalize = normalize

        # Min-Max values cho các góc (để normalize về [0, 1])
        # Góc có thể dao động từ 0 đến π (180 độ)
        self._angle_min = 0.0
        self._angle_max = np.pi

    def _is_valid_keypoint(self, kp: NDArray[np.float64]) -> bool:
        """
        Kiểm tra xem một keypoint có hợp lệ không.

        Args:
            kp: Array shape (3,) chứa [x, y, confidence].

        Returns:
            True nếu confidence >= conf_threshold, ngược lại False.
        """
        return len(kp) >= 3 and kp[2] >= self.conf_threshold

    def _get_valid_mask(self, keypoints: NDArray[np.float64]) -> NDArray[np.bool_]:
        """
        Tạo mask cho các keypoints hợp lệ.

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Boolean mask shape (17,) với True cho keypoints hợp lệ.
        """
        if keypoints.shape[0] != 17:
            raise ValueError(
                f"Expected 17 keypoints, got {keypoints.shape[0]}. "
                f"Input shape must be (17, 3)."
            )
        confidence = keypoints[:, 2]
        return confidence >= self.conf_threshold

    def _safe_normalize(self, v: NDArray[np.float64]) -> float:
        """
        Chuẩn hóa vector về đơn vị.

        Args:
            v: Vector có thể là zero vector.

        Returns:
            Vector đã chuẩn hóa, hoặc zero vector nếu norm < eps.
        """
        norm = np.linalg.norm(v)
        if norm < self.eps:
            return np.zeros_like(v)
        return v / norm

    def _angle_between_vectors(
        self,
        v1: NDArray[np.float64],
        v2: NDArray[np.float64],
    ) -> float:
        """
        Tính góc giữa hai vector sử dụng dot product.

        Args:
            v1: Vector thứ nhất (2D hoặc 3D).
            v2: Vector thứ hai (2D hoặc 3D).

        Returns:
            Góc trong radians, nằm trong [0, π].
            Trả về 0.0 nếu một trong hai vector có norm < eps.
        """
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < self.eps or norm2 < self.eps:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        # Clamp để tránh lỗi floating point
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def _normalize_angle(self, angle: float) -> float:
        """
        Chuẩn hóa góc về [0, 1] nếu normalize=True.

        Args:
            angle: Góc trong radians.

        Returns:
            Góc đã chuẩn hóa hoặc góc gốc.
        """
        if self.normalize:
            normalized = (angle - self._angle_min) / (self._angle_max - self._angle_min + self.eps)
            return float(np.clip(normalized, 0.0, 1.0))
        return float(angle)

    # ─────────────────────────────────────────────────────────────
    # 9 Geometric Features
    # ─────────────────────────────────────────────────────────────

    def _center_of_mass(self, keypoints: NDArray[np.float64]) -> tuple[float, float]:
        """
        Tính trọng tâm (Center of Mass) của các keypoints hợp lệ.

        Args:
            keypoints: Array shape (17, 3) với [x, y, conf].

        Returns:
            Tuple (cm_x, cm_y) là tọa độ trọng tâm.
        """
        valid_mask = self._get_valid_mask(keypoints)
        valid_points = keypoints[valid_mask, :2]  # Chỉ lấy x, y

        if len(valid_points) == 0:
            return (0.0, 0.0)

        return (float(np.mean(valid_points[:, 0])), float(np.mean(valid_points[:, 1])))

    def _shoulder_nose_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc vai-mũi (Shoulder-Nose Angle).

        Góc giữa hai vector BA và BC:
        - B = Nose (đỉnh)
        - A = L_Shoulder
        - C = R_Shoulder

        Công thức: arccos((BA · BC) / (|BA| × |BC|))

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
        """
        nose = keypoints[self.NOSE]
        l_shoulder = keypoints[self.L_SHOULDER]
        r_shoulder = keypoints[self.R_SHOULDER]

        # Vector BA = A - B (từ Nose đến L_Shoulder)
        # Vector BC = C - B (từ Nose đến R_Shoulder)
        ba = l_shoulder[:2] - nose[:2]
        bc = r_shoulder[:2] - nose[:2]

        angle = self._angle_between_vectors(ba, bc)
        return self._normalize_angle(angle)

    def _torso_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc thân (Torso Angle).

        Góc giữa vector thân và trục dọc (y-axis):
        - Vector v = [mid_hip_x - nose_x, mid_hip_y - nose_y]

        Công thức: arccos(v_y / |v|)

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
        """
        nose = keypoints[self.NOSE]
        l_hip = keypoints[self.L_HIP]
        r_hip = keypoints[self.R_HIP]

        # Trung điểm hông
        mid_hip = (l_hip[:2] + r_hip[:2]) / 2.0

        # Vector từ mũi đến giữa hông
        v = mid_hip - nose[:2]

        # Tính góc với trục y (vertical)
        v_norm = np.linalg.norm(v)
        if v_norm < self.eps:
            return 0.0

        # cos(angle) = v_y / |v|  (projection lên trục y)
        cos_angle = np.clip(v[1] / v_norm, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return self._normalize_angle(angle)

    def _hip_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc hông (Hip Angle).

        Góc giữa đường hông và trục ngang (x-axis):
        - Vector v = [R_hip_x - L_hip_x, R_hip_y - L_hip_y]

        Công thức: arccos(v_x / |v|)

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
        """
        l_hip = keypoints[self.L_HIP]
        r_hip = keypoints[self.R_HIP]

        # Vector từ hông trái đến hông phải
        v = r_hip[:2] - l_hip[:2]

        v_norm = np.linalg.norm(v)
        if v_norm < self.eps:
            return 0.0

        # cos(angle) = v_x / |v|  (projection lên trục x)
        cos_angle = np.clip(v[0] / v_norm, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return self._normalize_angle(angle)

    def _shoulder_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc vai (Shoulder Angle).

        Góc giữa đường vai và trục ngang (x-axis):
        - Vector v = [R_shoulder_x - L_shoulder_x, R_shoulder_y - L_shoulder_y]

        Công thức: arccos(v_x / |v|)

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
        """
        l_shoulder = keypoints[self.L_SHOULDER]
        r_shoulder = keypoints[self.R_SHOULDER]

        # Vector từ vai trái đến vai phải
        v = r_shoulder[:2] - l_shoulder[:2]

        v_norm = np.linalg.norm(v)
        if v_norm < self.eps:
            return 0.0

        # cos(angle) = v_x / |v|  (projection lên trục x)
        cos_angle = np.clip(v[0] / v_norm, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return self._normalize_angle(angle)

    def _left_leg_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc chân trái (Left Leg Angle).

        Góc tại đầu gối trái giữa:
        - Vector v1 = [L_knee - L_hip]  (đùi)
        - Vector v2 = [L_ankle - L_knee]  (cẳng chân)

        Công thức: arccos((v1 · v2) / (|v1| × |v2|))

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
            Trả về 0.0 nếu thiếu keypoints.
        """
        l_hip = keypoints[self.L_HIP]
        l_knee = keypoints[self.L_KNEE]
        l_ankle = keypoints[self.L_ANKLE]

        # Kiểm tra keypoints hợp lệ
        if not (self._is_valid_keypoint(l_hip) and
                self._is_valid_keypoint(l_knee) and
                self._is_valid_keypoint(l_ankle)):
            return 0.0

        # Vector đùi (từ hông đến đầu gối)
        v1 = l_knee[:2] - l_hip[:2]
        # Vector cẳng chân (từ đầu gối đến mắt cá)
        v2 = l_ankle[:2] - l_knee[:2]

        angle = self._angle_between_vectors(v1, v2)
        return self._normalize_angle(angle)

    def _right_leg_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc chân phải (Right Leg Angle).

        Góc tại đầu gối phải giữa:
        - Vector v1 = [R_knee - R_hip]  (đùi)
        - Vector v2 = [R_ankle - R_knee]  (cẳng chân)

        Công thức: arccos((v1 · v2) / (|v1| × |v2|))

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
            Trả về 0.0 nếu thiếu keypoints.
        """
        r_hip = keypoints[self.R_HIP]
        r_knee = keypoints[self.R_KNEE]
        r_ankle = keypoints[self.R_ANKLE]

        # Kiểm tra keypoints hợp lệ
        if not (self._is_valid_keypoint(r_hip) and
                self._is_valid_keypoint(r_knee) and
                self._is_valid_keypoint(r_ankle)):
            return 0.0

        # Vector đùi (từ hông đến đầu gối)
        v1 = r_knee[:2] - r_hip[:2]
        # Vector cẳng chân (từ đầu gối đến mắt cá)
        v2 = r_ankle[:2] - r_knee[:2]

        angle = self._angle_between_vectors(v1, v2)
        return self._normalize_angle(angle)

    def _nose_to_ankle_angle(self, keypoints: NDArray[np.float64]) -> float:
        """
        Tính góc mũi-mắt cá chân (Nose-to-Ankle Angle).

        Góc giữa trục cơ thể (từ mũi đến giữa 2 mắt cá) và trục dọc:
        - Vector v = [mid_ankle_x - nose_x, mid_ankle_y - nose_y]

        Công thức: arccos(v_y / |v|)

        Args:
            keypoints: Array shape (17, 3).

        Returns:
            Góc trong radians [0, π], đã chuẩn hóa về [0, 1].
        """
        nose = keypoints[self.NOSE]
        l_ankle = keypoints[self.L_ANKLE]
        r_ankle = keypoints[self.R_ANKLE]

        # Trung điểm mắt cá
        mid_ankle = (l_ankle[:2] + r_ankle[:2]) / 2.0

        # Vector từ mũi đến giữa mắt cá
        v = mid_ankle - nose[:2]

        v_norm = np.linalg.norm(v)
        if v_norm < self.eps:
            return 0.0

        # cos(angle) = v_y / |v|  (projection lên trục y)
        cos_angle = np.clip(v[1] / v_norm, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        return self._normalize_angle(angle)

    # ─────────────────────────────────────────────────────────────
    # Main Feature Extraction
    # ─────────────────────────────────────────────────────────────

    def extract(self, keypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Trích xuất vector đặc trưng 60 chiều từ 17 COCO keypoints.

        Args:
            keypoints: NumPy array shape (17, 3) chứa [x, y, confidence]
                       cho 17 COCO keypoints.

        Returns:
            NumPy array shape (60,) chứa:
            - [0:51]  - 17 keypoints × 3 = 51 chiều (flattened)
            - [51:60] - 9 đặc trưng hình học

        Raises:
            ValueError: Nếu input không có shape (17, 3).

        Ví dụ:
            >>> extractor = GeometricFeatureExtractor()
            >>> kp = np.random.rand(17, 3)
            >>> features = extractor.extract(kp)
            >>> print(features.shape)
            (60,)
            >>> print(features[:5])   # 5 keypoint dims đầu
            >>> print(features[51:]) # 9 geometric features cuối

        Chi tiết đặc trưng hình học (9 chiều cuối):
            [51] center_mass_x       - Trọng tâm X
            [52] center_mass_y       - Trọng tâm Y
            [53] shoulder_nose_angle - Góc vai-mũi
            [54] torso_angle         - Góc thân
            [55] hip_angle          - Góc hông
            [56] shoulder_angle      - Góc vai
            [57] left_leg_angle     - Góc chân trái
            [58] right_leg_angle    - Góc chân phải
            [59] nose_to_ankle_angle - Góc mũi-mắt cá
        """
        # Validate input
        if keypoints.shape != (17, 3):
            raise ValueError(
                f"Expected keypoints shape (17, 3), got {keypoints.shape}. "
                f"Input phải là mảng 2D với 17 rows (keypoints) và 3 columns (x, y, conf)."
            )

        # ─── Phần 1: Flattened Keypoints (51 chiều) ───
        # Flatten theo row-major order: [x0, y0, conf0, x1, y1, conf1, ...]
        keypoints_flat = keypoints.flatten()  # shape: (51,)

        # ─── Phần 2: Geometric Features (9 chiều) ───
        geometric_features = np.array([
            # 1 & 2. Center of Mass
            self._center_of_mass(keypoints)[0],       # cm_x
            self._center_of_mass(keypoints)[1],       # cm_y

            # 3. Shoulder-Nose Angle
            self._shoulder_nose_angle(keypoints),

            # 4. Torso Angle
            self._torso_angle(keypoints),

            # 5. Hip Angle
            self._hip_angle(keypoints),

            # 6. Shoulder Angle
            self._shoulder_angle(keypoints),

            # 7. Left Leg Angle
            self._left_leg_angle(keypoints),

            # 8. Right Leg Angle
            self._right_leg_angle(keypoints),

            # 9. Nose-to-Ankle Angle
            self._nose_to_ankle_angle(keypoints),
        ], dtype=np.float64)

        # ─── Concatenate: 51 + 9 = 60 chiều ───
        final_features = np.concatenate([keypoints_flat, geometric_features])

        return final_features

    def extract_batch(
        self,
        keypoints_batch: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Trích xuất features cho một batch các keypoints.

        Args:
            keypoints_batch: NumPy array shape (B, 17, 3) với B là batch size.

        Returns:
            NumPy array shape (B, 60).

        Ví dụ:
            >>> extractor = GeometricFeatureExtractor()
            >>> batch = np.random.rand(8, 17, 3)  # 8 samples
            >>> features = extractor.extract_batch(batch)
            >>> print(features.shape)
            (8, 60)
        """
        if keypoints_batch.ndim != 3 or keypoints_batch.shape[1:] != (17, 3):
            raise ValueError(
                f"Expected shape (B, 17, 3), got {keypoints_batch.shape}."
            )

        batch_size = keypoints_batch.shape[0]
        # Pre-allocate output array for efficiency
        features = np.zeros((batch_size, 60), dtype=np.float64)

        # Vectorized extraction using broadcasting
        # Extract all keypoints flattened at once
        features[:, :51] = keypoints_batch.reshape(batch_size, -1)

        # Compute geometric features for each sample in batch
        for i in range(batch_size):
            kp = keypoints_batch[i]
            # Center of mass
            valid_mask = kp[:, 2] >= self.conf_threshold
            valid_points = kp[valid_mask, :2]
            if len(valid_points) > 0:
                features[i, 51] = float(np.mean(valid_points[:, 0]))
                features[i, 52] = float(np.mean(valid_points[:, 1]))

            # Vectorized angle computation
            angles = self._compute_all_angles_batch(kp)
            features[i, 53:60] = angles

        return features

    def _compute_all_angles_batch(self, keypoints: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute all 7 geometric angles in one pass (vectorized).

        Returns:
            Array of 7 angles: [shoulder_nose, torso, hip, shoulder, left_leg, right_leg, nose_ankle]
        """
        angles = np.zeros(7, dtype=np.float64)
        eps = self.eps

        # Indices
        N, LS, RS, LH, RH, LK, RK, LA, RA = 0, 5, 6, 11, 12, 13, 14, 15, 16

        # Shoulder-Nose Angle
        ba = keypoints[LS, :2] - keypoints[N, :2]
        bc = keypoints[RS, :2] - keypoints[N, :2]
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 > eps and n2 > eps:
            cos_a = np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1)
            angles[0] = np.arccos(cos_a)

        # Torso Angle
        mid_hip = (keypoints[LH, :2] + keypoints[RH, :2]) / 2
        v = mid_hip - keypoints[N, :2]
        n = np.linalg.norm(v)
        if n > eps:
            angles[1] = np.arccos(np.clip(v[1] / n, -1, 1))

        # Hip Angle
        v = keypoints[RH, :2] - keypoints[LH, :2]
        n = np.linalg.norm(v)
        if n > eps:
            angles[2] = np.arccos(np.clip(v[0] / n, -1, 1))

        # Shoulder Angle
        v = keypoints[RS, :2] - keypoints[LS, :2]
        n = np.linalg.norm(v)
        if n > eps:
            angles[3] = np.arccos(np.clip(v[0] / n, -1, 1))

        # Left Leg Angle
        if all(keypoints[i, 2] >= self.conf_threshold for i in [LH, LK, LA]):
            v1 = keypoints[LK, :2] - keypoints[LH, :2]
            v2 = keypoints[LA, :2] - keypoints[LK, :2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > eps and n2 > eps:
                angles[4] = np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))

        # Right Leg Angle
        if all(keypoints[i, 2] >= self.conf_threshold for i in [RH, RK, RA]):
            v1 = keypoints[RK, :2] - keypoints[RH, :2]
            v2 = keypoints[RA, :2] - keypoints[RK, :2]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > eps and n2 > eps:
                angles[5] = np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))

        # Nose-to-Ankle Angle
        mid_ankle = (keypoints[LA, :2] + keypoints[RA, :2]) / 2
        v = mid_ankle - keypoints[N, :2]
        n = np.linalg.norm(v)
        if n > eps:
            angles[6] = np.arccos(np.clip(v[1] / n, -1, 1))

        # Normalize all angles to [0, 1] if enabled
        if self.normalize:
            angle_max = np.pi
            angles = np.clip(angles / angle_max, 0, 1)

        return angles

    def get_feature_names(self) -> list[str]:
        """
        Trả về danh sách tên của 60 đặc trưng.

        Returns:
            List 60 strings với tên các đặc trưng.
        """
        keypoint_names = [f"{name}_x" for name in self.KEYPOINT_NAMES] + \
                         [f"{name}_y" for name in self.KEYPOINT_NAMES] + \
                         [f"{name}_conf" for name in self.KEYPOINT_NAMES]

        geometric_names = [
            "center_mass_x",
            "center_mass_y",
            "shoulder_nose_angle",
            "torso_angle",
            "hip_angle",
            "shoulder_angle",
            "left_leg_angle",
            "right_leg_angle",
            "nose_to_ankle_angle",
        ]

        return keypoint_names + geometric_names

    def __repr__(self) -> str:
        return (
            f"GeometricFeatureExtractor("
            f"conf_threshold={self.conf_threshold}, "
            f"eps={self.eps}, "
            f"normalize={self.normalize})"
        )


# ═══════════════════════════════════════════════════════════════
# Convenience Functions (standalone API)
# ═══════════════════════════════════════════════════════════════

# Default extractor instance cho reuse
_default_extractor: GeometricFeatureExtractor | None = None


def get_default_extractor(
    conf_threshold: float = 0.2,
    normalize: bool = True,
) -> GeometricFeatureExtractor:
    """
    Lấy default extractor instance (singleton pattern).

    Args:
        conf_threshold: Ngưỡng confidence.
        normalize: Có chuẩn hóa góc không.

    Returns:
        GeometricFeatureExtractor instance.
    """
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = GeometricFeatureExtractor(
            conf_threshold=conf_threshold,
            normalize=normalize,
        )
    return _default_extractor


def extract_pifr_features(keypoints: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Convenience function để trích xuất PIFR features.

    Args:
        keypoints: NumPy array shape (17, 3).

    Returns:
        NumPy array shape (60,).
    """
    extractor = get_default_extractor()
    return extractor.extract(keypoints)


# ═══════════════════════════════════════════════════════════════
# Demo & Testing
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PIFR Geometric Feature Extractor - Demo")
    print("=" * 60)

    # Tạo sample keypoints (17 COCO keypoints)
    np.random.seed(42)

    # Simulate a standing pose (normalized coordinates ~0.3-0.7)
    standing_pose = np.array([
        # [x, y, conf]
        [0.50, 0.15, 0.95],  # 0: nose
        [0.48, 0.14, 0.90],  # 1: l_eye
        [0.52, 0.14, 0.92],  # 2: r_eye
        [0.47, 0.13, 0.85],  # 3: l_ear
        [0.53, 0.13, 0.88],  # 4: r_ear
        [0.40, 0.30, 0.95],  # 5: l_shoulder
        [0.60, 0.30, 0.95],  # 6: r_shoulder
        [0.35, 0.45, 0.90],  # 7: l_elbow
        [0.65, 0.45, 0.88],  # 8: r_elbow
        [0.32, 0.60, 0.85],  # 9: l_wrist
        [0.68, 0.60, 0.82],  # 10: r_wrist
        [0.43, 0.55, 0.95],  # 11: l_hip
        [0.57, 0.55, 0.95],  # 12: r_hip
        [0.44, 0.75, 0.90],  # 13: l_knee
        [0.56, 0.75, 0.90],  # 14: r_knee
        [0.43, 0.95, 0.88],  # 15: l_ankle
        [0.57, 0.95, 0.87],  # 16: r_ankle
    ], dtype=np.float64)

    # Simulate a lying/fallen pose (horizontal)
    lying_pose = np.array([
        [0.90, 0.50, 0.95],  # 0: nose (bên phải)
        [0.92, 0.49, 0.90],  # 1: l_eye
        [0.94, 0.51, 0.92],  # 2: r_eye
        [0.95, 0.48, 0.85],  # 3: l_ear
        [0.96, 0.52, 0.88],  # 4: r_ear
        [0.75, 0.45, 0.95],  # 5: l_shoulder
        [0.75, 0.55, 0.95],  # 6: r_shoulder
        [0.60, 0.42, 0.90],  # 7: l_elbow
        [0.60, 0.58, 0.88],  # 8: r_elbow
        [0.45, 0.40, 0.85],  # 9: l_wrist
        [0.45, 0.60, 0.82],  # 10: r_wrist
        [0.40, 0.45, 0.95],  # 11: l_hip
        [0.40, 0.55, 0.95],  # 12: r_hip
        [0.20, 0.43, 0.90],  # 13: l_knee
        [0.20, 0.57, 0.90],  # 14: r_knee
        [0.05, 0.45, 0.88],  # 15: l_ankle
        [0.05, 0.55, 0.87],  # 16: r_ankle
    ], dtype=np.float64)

    # Initialize extractor
    extractor = GeometricFeatureExtractor(conf_threshold=0.2, normalize=True)

    print(f"\nExtractor config: {extractor}")
    print(f"\nFeature names (last 9): {extractor.get_feature_names()[51:]}")

    # Extract features
    print("\n" + "-" * 40)
    print("STANDING POSE:")
    print("-" * 40)
    feat_standing = extractor.extract(standing_pose)
    print(f"  Output shape: {feat_standing.shape}")
    print(f"  Keypoints (first 5): {feat_standing[:5]}")
    print(f"  Geometric features: {feat_standing[51:]}")
    print(f"  Torso angle: {feat_standing[54]:.4f} (standing ~ 0.0)")
    print(f"  Nose-to-ankle: {feat_standing[59]:.4f} (standing ~ 0.0)")

    print("\n" + "-" * 40)
    print("LYING/FALLEN POSE:")
    print("-" * 40)
    feat_lying = extractor.extract(lying_pose)
    print(f"  Output shape: {feat_lying.shape}")
    print(f"  Keypoints (first 5): {feat_lying[:5]}")
    print(f"  Geometric features: {feat_lying[51:]}")
    print(f"  Torso angle: {feat_lying[54]:.4f} (lying ~ 1.0)")
    print(f"  Nose-to-ankle: {feat_lying[59]:.4f} (lying ~ 1.0)")

    print("\n" + "-" * 40)
    print("BATCH EXTRACTION:")
    print("-" * 40)
    batch = np.stack([standing_pose, lying_pose], axis=0)
    print(f"  Input batch shape: {batch.shape}")
    batch_features = extractor.extract_batch(batch)
    print(f"  Output batch shape: {batch_features.shape}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# Standalone Functions & Constants (for inference pipeline)
# ═══════════════════════════════════════════════════════════════════════

# Module-level constants
EPS = 1e-6
IMGSZ = 640
MIN_MEAN_CONF = 0.2
FEATURE_DIM = 60
SEQ_LEN = 60


def resample_to_length(seq: NDArray[np.float64], target_len: int = 60) -> NDArray[np.float64]:
    """
    Resample a sequence to a fixed length using linear interpolation.

    Args:
        seq: Input array of shape (N, feature_dim) or (N,).
        target_len: Target length for output sequence.

    Returns:
        Resampled array of shape (target_len, feature_dim) or (target_len,).
    """
    n = len(seq)
    if n == 0:
        return np.zeros((target_len, seq.shape[1] if seq.ndim > 1 else 1), dtype=np.float64)
    if n == target_len:
        return seq.copy()

    # Vectorized linear interpolation
    indices = np.linspace(0, n - 1, target_len)
    floor_idx = np.floor(indices).astype(int)
    ceil_idx = np.minimum(floor_idx + 1, n - 1)
    weights = (indices - floor_idx).reshape(-1, 1) if seq.ndim > 1 else (indices - floor_idx)

    return (1 - weights) * seq[floor_idx] + weights * seq[ceil_idx]


def frame_to_vector_60(
    keypoints: NDArray[np.float64],
    extractor: GeometricFeatureExtractor | None = None,
    box_wh: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """
    Chuyển đổi keypoints của một frame thành vector đặc trưng 60-D PIFR.

    Args:
        keypoints: Array shape (17, 3) với [x, y, conf].
        extractor: Instance extractor đã cấu hình (mặc định: tạo mới).
        box_wh: Tuple (width, height) của bounding box - dùng để chuẩn hóa tọa độ
                theo kích thước thực của đối tượng thay vì toàn bộ frame.

    Returns:
        Vector đặc trưng 60-D dạng 1D array.

    Ví dụ:
        >>> keypoints = np.random.rand(17, 3)
        >>> vec = frame_to_vector_60(keypoints)
        >>> print(vec.shape)
        (60,)
    """
    if extractor is None:
        extractor = GeometricFeatureExtractor()
    return extractor.extract(keypoints)
