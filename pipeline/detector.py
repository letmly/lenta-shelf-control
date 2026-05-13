"""YOLO детектор ценников.

TODO Д2: fine-tune YOLOv8n на bbox из data/labeled/*/*.csv
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    conf: float


class PriceTagDetector:
    def __init__(self, weights: Path | str = "yolov8n.pt") -> None:
        self.weights = Path(weights)
        # TODO: from ultralytics import YOLO; self.model = YOLO(self.weights)

    def predict(self, frame) -> list[Detection]:
        raise NotImplementedError("Д2: подключить ultralytics.YOLO.predict")
