"""CLI-обёртка над pipeline.process_video.

Usage:
    python scripts/run_pipeline.py --video data/labeled/43_15/43_15.mp4 \\
                                   --output outputs/43_15.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

# from pipeline.pipeline import process_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raise SystemExit("Stub. Реализация в Д3.")


if __name__ == "__main__":
    main()
