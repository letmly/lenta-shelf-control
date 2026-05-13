"""Скрипт скачивания видео из общего облака команды.

Источник — заполнить, когда определимся (Yandex.Disk / Google Drive / S3).
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_ROOT = Path(__file__).parent.parent / "data"

# TODO: заполнить, когда сложится shared-storage
VIDEO_URLS: dict[str, str] = {
    "labeled/25_12-20/25_12-20.mp4": "TBD",
    "labeled/26_12-20/26_12-20.mp4": "TBD",
    "labeled/43_15/43_15.mp4":       "TBD",
    "unlabeled/25_12-20.mp4":        "TBD",
    "unlabeled/26_12-20.mp4":        "TBD",
    "unlabeled/26_2-10.mp4":         "TBD",
}


def main() -> int:
    print("Stub. Заполнить VIDEO_URLS и реализовать загрузку.")
    print(f"Файлы будут сохраняться в {DATA_ROOT}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
