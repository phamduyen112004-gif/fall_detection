"""
Wrapper chạy ablation cho pipeline góc (rule-based).

Kaggle:
  python -m src.eval.ablation_runner --help
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run rule-based ablation study")
    ap.add_argument("--video", action="append", default=[], help="Đường dẫn video (có thể lặp).")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--fetch-samples", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir or Path(os.environ.get("FALL_WORK_ROOT", "/kaggle/working")) / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(root / "tools" / "ablation_study.py")]
    if args.fetch_samples:
        cmd.append("--fetch-samples")
    for v in args.video:
        cmd.extend(["--video", v])
    cmd.extend(["--out-dir", str(out_dir)])

    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

