from __future__ import annotations

from pathlib import Path
import zipfile

from prepare_dataset import extract_urfd_clips


def _write_text(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_extract_urfd_clips_supports_zip_and_folder(tmp_path: Path) -> None:
    urfd_root = tmp_path / "URFD"
    aio_root = tmp_path / "AIO_Dataset"

    # ADL zip clip (đặt trong thư mục lồng để kiểm tra rglob)
    adl_zip = urfd_root / "ADL" / "nested" / "adl-01-cam0-rgb.zip"
    adl_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(adl_zip, "w") as zf:
        zf.writestr("frame001.jpg", "dummy")

    # Cùng stem với zip -> phải bị bỏ qua ở nhánh folder (tránh trùng)
    _write_text(urfd_root / "ADL" / "adl-01-cam0-rgb" / "frame002.jpg")

    # Fall clip dạng folder đã giải nén
    _write_text(urfd_root / "Fall" / "fall-01-cam0-rgb" / "frame001.jpg")

    n = extract_urfd_clips(urfd_root, aio_root)
    assert n == 2

    out_adl = aio_root / "nofall" / "urfd_adl_adl-01-cam0-rgb"
    out_fall = aio_root / "fall" / "urfd_fall_fall-01-cam0-rgb"
    assert out_adl.is_dir()
    assert out_fall.is_dir()
    assert any(out_adl.rglob("*"))
    assert any(out_fall.rglob("*"))


def test_extract_urfd_clips_returns_zero_when_missing_layout(tmp_path: Path) -> None:
    urfd_root = tmp_path / "URFD"
    aio_root = tmp_path / "AIO_Dataset"
    urfd_root.mkdir(parents=True, exist_ok=True)

    n = extract_urfd_clips(urfd_root, aio_root)
    assert n == 0

