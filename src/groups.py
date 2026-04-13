"""Nhóm clip để chia train/val (tránh leakage cùng subject / cùng chuỗi URFD)."""

from __future__ import annotations

import re
from pathlib import Path


def group_id_from_clip_path(path: Path) -> str:
    """
    - GMDCSA: gmdcsa_subject3_*.mp4 -> 'gmdcsa_subject3' (cùng subject = cùng nhóm).
    - URFD: thư mục urfd_fall_* / urfd_adl_* -> tên thư mục (mỗi clip một nhóm).
    - Khác: đường dẫn tuyệt đối (mỗi clip một nhóm).
    """
    name = path.name
    m = re.match(r"(gmdcsa_subject\d+)_", name, re.I)
    if m:
        return m.group(1).lower()
    m2 = re.match(r"^(urfd_(?:fall|adl)_[\w\-.]+)", name, re.I)
    if m2:
        return m2.group(1).lower()
    return str(path.resolve())
