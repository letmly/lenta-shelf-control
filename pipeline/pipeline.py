"""Оркестратор: видео → DataFrame со всеми колонками ТЗ.

Шаги:
    1. iter_frames(video)         — sampling 1–3 fps
    2. detector.predict           — bbox ценников
    3. tracker.update             — track_id
    4. для каждого завершённого трека:
        - best_frame              — лучший кадр
        - qr.decode               — QR-поля
        - color.classify          — цвет
        - template.classify       — шаблон 1..7
        - ocr.read + parser.parse — текстовые поля
        - aggregator.vote         — голосование между кадрами
    5. сборка строки CSV
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CSV_COLUMNS = [
    # с ценника
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
    # из QR
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]


def process_video(video_path: str | Path) -> pd.DataFrame:
    """Главный entrypoint пайплайна."""
    raise NotImplementedError("Д3: end-to-end оркестрация")
