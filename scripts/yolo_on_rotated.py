"""Тест openfoodfacts YOLOv11x на rotated frame (CCW 90°).

Гипотеза: модель обучена на ровно-ориентированных кадрах европейских магазинов.
На rotated frame она должна детектить ценники Ленты точнее чем red-mask.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "_third_party/yolo_weights/weights/best.pt"
OUT = ROOT / "outputs" / "yolo_rotated"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def rotate_bbox_ccw90(bbox, H_orig, W_orig):
    """Координаты bbox: (x1,y1,x2,y2) в original → в rotated CCW 90°.
    rotated has shape (W_orig, H_orig) — swapped.
    Mapping: new_x = y, new_y = W_orig - x_new_max
    For CCW90: (x,y) -> (y, W-x)
    bbox(x1,y1,x2,y2): new corners are (y1, W-x2), (y2, W-x1)
    """
    x1, y1, x2, y2 = bbox
    nx1, ny1 = y1, W_orig - x2
    nx2, ny2 = y2, W_orig - x1
    return (nx1, ny1, nx2, ny2)


def rotate_bbox_back(bbox, H_orig, W_orig):
    """rotated CCW 90° back to original. Inverse: CW 90°.
    On rotated frame (W_orig, H_orig), bbox (x1,y1,x2,y2).
    Back: (x,y)_rot -> (W_orig - y, x)
    """
    x1, y1, x2, y2 = bbox
    nx1, ny1 = W_orig - y2, x1
    nx2, ny2 = W_orig - y1, x2
    return (nx1, ny1, nx2, ny2)


print(f"Loading YOLO {WEIGHTS}...")
model = YOLO(str(WEIGHTS))
print(f"Classes: {model.names}")

# Тест на 26_12-20
df = pd.read_csv(ROOT / "data/labeled/26_12-20/26_12-20.csv", encoding="utf-8")
for c in ("x_min","y_min","x_max","y_max","frame_timestamp"):
    df[c] = df[c].apply(parse_ru)
df = df.dropna(subset=["x_min","y_min","x_max","y_max","frame_timestamp"]).reset_index(drop=True)

cap = cv2.VideoCapture(str(ROOT / "data/labeled/26_12-20/26_12-20.mp4"))

# Берём несколько разных timestamps
test_ts = sorted(df["frame_timestamp"].unique())[::max(1, len(df)//5)][:5]

total_gt = 0; total_matched = 0; total_pred = 0

for ts in test_ts:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
    ok, frame = cap.read()
    if not ok: continue
    H_orig, W_orig = frame.shape[:2]
    # Rotate CCW
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    H_rot, W_rot = rotated.shape[:2]

    # YOLO на rotated, разные conf
    results = model.predict(rotated, conf=0.1, iou=0.5, imgsz=1920, verbose=False, max_det=200)
    pred_rot = []
    if results[0].boxes is not None:
        for box, conf in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.conf.cpu().numpy()):
            pred_rot.append((tuple(box), float(conf)))

    # Переводим pred bbox обратно в original orientation для сравнения с GT
    pred_orig = []
    for (bb, conf) in pred_rot:
        bb_orig = rotate_bbox_back(bb, H_orig, W_orig)
        pred_orig.append((bb_orig, conf))

    # GT на этом ts
    gts = df[df["frame_timestamp"] == ts]
    gt_boxes = [(r["x_min"], r["y_min"], r["x_max"], r["y_max"]) for _, r in gts.iterrows()]

    matched = 0
    for gt in gt_boxes:
        best = max((iou(gt, p[0]) for p in pred_orig), default=0)
        if best >= 0.3: matched += 1
    total_gt += len(gt_boxes)
    total_pred += len(pred_orig)
    total_matched += matched

    print(f"  ts={int(ts)}ms  GT={len(gt_boxes)}  pred={len(pred_orig)}  matched_IoU>=0.3={matched}")

    # визуализация (на rotated frame)
    vis = rotated.copy()
    for bb, conf in pred_rot:
        x1,y1,x2,y2 = map(int, bb)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (255, 100, 0), 4)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
    # GT в original → нужно тоже rotate в visual
    for gt in gt_boxes:
        gt_rot = rotate_bbox_ccw90(gt, H_orig, W_orig)
        x1,y1,x2,y2 = map(int, gt_rot)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0, 255, 0), 4)
    cv2.imwrite(str(OUT / f"frame_ts{int(ts):06d}.jpg"), cv2.resize(vis, (720, 1280)))

cap.release()
print(f"\n=== SUMMARY ===")
print(f"  Total GT:       {total_gt}")
print(f"  Total pred:     {total_pred}")
print(f"  Matched IoU>=0.3: {total_matched}/{total_gt}  ({total_matched/total_gt:.0%})")
print(f"\n  Visualizations -> {OUT}")
