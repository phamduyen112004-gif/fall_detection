"""
Đặc trưng PIFR + keypoint COCO: 60 chiều / frame = 51 keypoint + 9 hình học.

9 hình học (thống nhất toàn dự án): com_x, com_y, torso_angle, nose_to_ankle_angle,
hip_angle, shoulder_angle, left_leg_angle, right_leg_angle, shoulder_nose_angle.
(không dùng bbox_aspect_ratio trong vector 60-D; bbox chỉ truyền cho API tương thích.)
"""

from __future__ import annotations

import math

import numpy as np

# --- COCO 17 ---
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

MIN_MEAN_CONF = 0.2
KPT_CONF_TH = 0.2
EPS = 1e-6
IMGSZ = 640
SEQ_LEN = 60
FEATURE_DIM = 60  # 51 + 9


def angle_vertical(pt1_xy: np.ndarray, pt2_xy: np.ndarray) -> float:
    """arccos(v_y / ||v||), v = pt2 - pt1."""
    v = pt2_xy.astype(np.float64) - pt1_xy.astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return 0.0
    c = float(np.clip(v[1] / n, -1.0, 1.0))
    return float(math.acos(c))


def angle_horizontal(pt1_xy: np.ndarray, pt2_xy: np.ndarray) -> float:
    """arccos(v_x / ||v||), v = pt2 - pt1."""
    v = pt2_xy.astype(np.float64) - pt1_xy.astype(np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return 0.0
    c = float(np.clip(v[0] / n, -1.0, 1.0))
    return float(math.acos(c))


def angle_at_b(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
    """Góc tại B giữa BA và BC."""
    ba = pa.astype(np.float64) - pb.astype(np.float64)
    bc = pc.astype(np.float64) - pb.astype(np.float64)
    n1 = float(np.linalg.norm(ba))
    n2 = float(np.linalg.norm(bc))
    if n1 < EPS or n2 < EPS:
        return 0.0
    c = float(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))
    return float(math.acos(c))


def leg_angle(p_hip: np.ndarray, p_knee: np.ndarray, p_ankle: np.ndarray) -> float:
    v1 = p_knee.astype(np.float64) - p_hip.astype(np.float64)
    v2 = p_ankle.astype(np.float64) - p_knee.astype(np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < EPS or n2 < EPS:
        return 0.0
    c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.acos(c))


def smart_mid_xy(
    k: np.ndarray,
    ia: int,
    ib: int,
    com_x: float,
    com_y: float,
    th: float = KPT_CONF_TH,
) -> tuple[float, float]:
    """
    Trung điểm thông minh (x,y) đã chuẩn hóa:
    cả hai >= th -> trung bình; một phía -> phía đó; cả hai < th -> COM.
    """
    ca, cb = float(k[ia, 2]), float(k[ib, 2])
    xa, ya = float(k[ia, 0]), float(k[ia, 1])
    xb, yb = float(k[ib, 0]), float(k[ib, 1])
    if ca >= th and cb >= th:
        return (xa + xb) * 0.5, (ya + yb) * 0.5
    if ca >= th:
        return xa, ya
    if cb >= th:
        return xb, yb
    return com_x, com_y


def compute_geometry_9(
    k_norm: np.ndarray,
    bbox_wh: tuple[float, float],
) -> np.ndarray:
    """
    9 đặc trưng (không bbox_aspect_ratio). bbox_wh giữ tham số để API đồng nhất.
    """
    _, _ = bbox_wh  # giữ API; không dùng trong 9 chiều hiện tại

    xy = k_norm[:, :2].astype(np.float64)
    center_mass_x = float(np.mean(xy[:, 0]))
    center_mass_y = float(np.mean(xy[:, 1]))

    nose = np.array(
        [float(k_norm[NOSE, 0]), float(k_norm[NOSE, 1])],
        dtype=np.float64,
    )

    mhx, mhy = smart_mid_xy(k_norm, L_HIP, R_HIP, center_mass_x, center_mass_y)
    mid_hip = np.array([mhx, mhy], dtype=np.float64)
    ax, ay = smart_mid_xy(k_norm, L_ANKLE, R_ANKLE, center_mass_x, center_mass_y)
    mid_ankle = np.array([ax, ay], dtype=np.float64)
    _ = smart_mid_xy(k_norm, L_SHOULDER, R_SHOULDER, center_mass_x, center_mass_y)

    torso_angle = angle_vertical(nose, mid_hip)
    nose_to_ankle_angle = angle_vertical(nose, mid_ankle)

    cl, cr = float(k_norm[L_HIP, 2]), float(k_norm[R_HIP, 2])
    hip_angle = (
        angle_horizontal(xy[L_HIP], xy[R_HIP])
        if cl >= KPT_CONF_TH and cr >= KPT_CONF_TH
        else 0.0
    )

    sl, sr = float(k_norm[L_SHOULDER, 2]), float(k_norm[R_SHOULDER, 2])
    shoulder_angle = (
        angle_horizontal(xy[L_SHOULDER], xy[R_SHOULDER])
        if sl >= KPT_CONF_TH and sr >= KPT_CONF_TH
        else 0.0
    )

    def leg_ok(ih: int, ik: int, ia: int) -> bool:
        return (
            float(k_norm[ih, 2]) >= KPT_CONF_TH
            and float(k_norm[ik, 2]) >= KPT_CONF_TH
            and float(k_norm[ia, 2]) >= KPT_CONF_TH
        )

    left_leg_angle = (
        leg_angle(xy[L_HIP], xy[L_KNEE], xy[L_ANKLE]) if leg_ok(L_HIP, L_KNEE, L_ANKLE) else 0.0
    )
    right_leg_angle = (
        leg_angle(xy[R_HIP], xy[R_KNEE], xy[R_ANKLE]) if leg_ok(R_HIP, R_KNEE, R_ANKLE) else 0.0
    )

    shoulder_nose_angle = angle_at_b(xy[L_SHOULDER], xy[NOSE], xy[R_SHOULDER])

    return np.array(
        [
            center_mass_x,
            center_mass_y,
            torso_angle,
            nose_to_ankle_angle,
            hip_angle,
            shoulder_angle,
            left_leg_angle,
            right_leg_angle,
            shoulder_nose_angle,
        ],
        dtype=np.float32,
    )


def flatten_keypoints(k_norm: np.ndarray) -> np.ndarray:
    return k_norm.reshape(-1).astype(np.float32)


def frame_to_vector_60(
    k_norm: np.ndarray,
    bbox_wh: tuple[float, float],
) -> np.ndarray:
    g9 = compute_geometry_9(k_norm, bbox_wh)
    return np.concatenate([flatten_keypoints(k_norm), g9], axis=0)


def resample_to_length(seq: np.ndarray, target_len: int) -> np.ndarray:
    t, f = seq.shape[0], seq.shape[1]
    if t == 0:
        return np.zeros((target_len, f), dtype=np.float32)
    if t == 1:
        return np.tile(seq.astype(np.float32), (target_len, 1))
    if t >= target_len:
        idx = np.linspace(0, t - 1, target_len, dtype=np.float64)
        idx = np.clip(np.round(idx).astype(np.int64), 0, t - 1)
        return seq[idx].astype(np.float32)
    pad = np.tile(seq[-1:], (target_len - t, 1))
    return np.vstack([seq.astype(np.float32), pad])
