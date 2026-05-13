"""Маппинг OCR-боксов в CSV-поля по template ROI.

Идея: каждый из 7 шаблонов имеет фиксированную раскладку. Координаты ROI
заданы в относительных коорд. ценника (0..1). Для каждого ROI собираем все
OCR-боксы, попавшие в него, и применяем regex-нормализатор по типу поля.

TODO Д3:
    1. Заполнить TEMPLATE_ROI по pptx-расшифровке (см. docs/03-templates.md).
    2. Реализовать pick_value().
"""
from __future__ import annotations

import re

# Тип поля → regex для извлечения значения из строки OCR
FIELD_REGEX = {
    "price":      re.compile(r"(\d+[\s.,]?\d{2})"),
    "barcode":    re.compile(r"(\d{8,14})"),
    "id_sku":     re.compile(r"(\d{10,12})"),
    "datetime":   re.compile(r"(\d{2}\.\d{2}\.\d{4}\s+\d{1,2}:\d{2})"),
    "discount":   re.compile(r"(-?\d+\s*[%₽р])"),
}

# Относительные ROI на ценнике: (x1, y1, x2, y2), все в [0..1]
# TODO Д3: уточнить по pptx-расшифровке для каждого из 7 шаблонов
TEMPLATE_ROI: dict[int, dict[str, tuple[float, float, float, float]]] = {
    1: {  # базовый
        "product_name":    (0.05, 0.05, 0.95, 0.25),
        "price_default":   (0.05, 0.60, 0.50, 0.80),
        "price_card":      (0.05, 0.30, 0.95, 0.60),
        "discount_amount": (0.70, 0.05, 0.95, 0.25),
        "barcode":         (0.05, 0.85, 0.50, 0.95),
        "id_sku":          (0.50, 0.85, 0.95, 0.95),
        "print_datetime":  (0.05, 0.95, 0.95, 1.00),
    },
    # 2..7: TODO
}


def parse(crop_w: int, crop_h: int, ocr_boxes, template_id: int) -> dict[str, str]:
    """Возвращает {csv_field: value} для одного ценника."""
    raise NotImplementedError("Д3: парсер шаблонов")
