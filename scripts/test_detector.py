"""Тест OpenFoodFacts детектора на одном кадре нашего видео.

Цель: понять, насколько хорошо pretrained YOLOv11x ловит наши ценники.
Прогон на 5 кадрах из 26_12-20 → визуально + метрика recall по GT.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "_third_party/yolo_weights/weights/best.pt"
OUT = ROOT / "outputs" / "detector_test"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


print(f"Loading YOLO from {WEIGHTS}...")
model = YOLO(str(WEIGHTS))
print(f"Model loaded. Classes: {model.names}")

# тестируем на нескольких таймстампах
csv_path = ROOT / "data/labeled/26_12-20/26_12-20.csv"
mp4_path = ROOT / "data/labeled/26_12-20/26_12-20.mp4"

df = pd.read_csv(csv_path, encoding="utf-8")
for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
    df[c] = df[c].apply(parse_ru)
df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

cap = cv2.VideoCapture(str(mp4_path))

# берём 5 timestamps, прогоняем
test_timestamps = sorted(df["frame_timestamp"].unique())[::len(df)//5][:5]
print(f"\nTesting on {len(test_timestamps)} timestamps...")

total_gt = 0; total_detected = 0; total_matched = 0
for ts in test_timestamps:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
    ok, frame = cap.read()
    if not ok: continue

    # GT bbox'ы для этого ts
    gts = df[df["frame_timestamp"] == ts]
    gt_boxes = [(r["x_min"], r["y_min"], r["x_max"], r["y_max"]) for _, r in gts.iterrows()]

    # инференс YOLO
    results = model.predict(frame, conf=0.05, iou=0.5, imgsz=1920, verbose=False, max_det=300)
    pred_boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            pred_boxes.append(tuple(box))

    # сравниваем
    matched = 0
    for gt in gt_boxes:
        best_iou = max((iou(gt, p) for p in pred_boxes), default=0)
        if best_iou >= 0.2:
            matched += 1
    total_gt += len(gt_boxes)
    total_detected += len(pred_boxes)
    total_matched += matched

    print(f"  ts={int(ts)}ms  GT={len(gt_boxes)}  detected={len(pred_boxes)}  matched_IoU>=0.3={matched}")

    # визуализация
    vis = frame.copy()
    # GT — зелёные
    for gt in gt_boxes:
        x1, y1, x2, y2 = map(int, gt)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(vis, "GT", (x1, max(y1-10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    # PRED — синие
    for i, p in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, p)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 3)
        cv2.putText(vis, f"#{i}", (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 100, 0), 2)
    vis_small = cv2.resize(vis, (1920, 1080))
    cv2.imwrite(str(OUT / f"frame_ts{int(ts):06d}.jpg"), vis_small)

cap.release()
print(f"\n=== Summary ===")
print(f"  Total GT:            {total_gt}")
print(f"  Total detected:      {total_detected}")
print(f"  Matched (IoU>=0.3):  {total_matched}/{total_gt}  ({total_matched/total_gt:.0%})")
print(f"\n  Visualizations saved to {OUT}/")
