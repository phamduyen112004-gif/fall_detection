"""Unit tests cho PIFR + resample."""

from __future__ import annotations

import numpy as np
import pytest

from pathlib import Path

from src.groups import group_id_from_clip_path
from src.pifr_features import (
    FEATURE_DIM,
    SEQ_LEN,
    compute_geometry_9,
    frame_to_vector_60,
    resample_to_length,
    smart_mid_xy,
)


def test_smart_mid_both_visible() -> None:
    k = np.zeros((17, 3), dtype=np.float32)
    k[5, :] = [0.3, 0.4, 0.9]
    k[6, :] = [0.5, 0.6, 0.9]
    x, y = smart_mid_xy(k, 5, 6, 0.5, 0.5)
    assert abs(x - 0.4) < 1e-5 and abs(y - 0.5) < 1e-5


def test_smart_mid_fallback_com() -> None:
    k = np.zeros((17, 3), dtype=np.float32)
    k[5, 2] = 0.1
    k[6, 2] = 0.1
    x, y = smart_mid_xy(k, 5, 6, 0.25, 0.75)
    assert x == 0.25 and y == 0.75


def test_frame_vector_shape() -> None:
    kn = np.random.rand(17, 3).astype(np.float32)
    kn[:, 2] = 0.5
    v = frame_to_vector_60(kn, (100.0, 200.0))
    assert v.shape == (60,)


def test_resample_to_60() -> None:
    seq = np.random.randn(10, FEATURE_DIM).astype(np.float32)
    out = resample_to_length(seq, SEQ_LEN)
    assert out.shape == (SEQ_LEN, FEATURE_DIM)


def test_geometry_no_nan() -> None:
    kn = np.random.rand(17, 3).astype(np.float32)
    kn[:, 2] = 0.5
    g = compute_geometry_9(kn, (64.0, 64.0))
    assert g.shape == (9,)
    assert np.all(np.isfinite(g))


def test_group_id_gmdcsa() -> None:
    g = group_id_from_clip_path(Path("gmdcsa_subject2_clip.mp4"))
    assert g == "gmdcsa_subject2"


def test_group_id_unique_path() -> None:
    p = Path("unique_clip_only_name.mp4")
    assert group_id_from_clip_path(p) == str(p.resolve())


def test_group_id_urfd_folder() -> None:
    assert group_id_from_clip_path(Path("urfd_fall_sequence01")) == "urfd_fall_sequence01"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
