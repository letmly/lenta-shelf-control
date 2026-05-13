"""Обёртка PaddleOCR (ru)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OCRBox:
    text: str
    conf: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class OCREngine:
    def __init__(self, lang: str = "ru") -> None:
        self.lang = lang
        # TODO: from paddleocr import PaddleOCR; self.engine = PaddleOCR(lang=lang, use_angle_cls=True)

    def read(self, crop) -> list[OCRBox]:
        raise NotImplementedError("Д3: подключить PaddleOCR")
