"""Baseline-детектор ценников через HSV красного цвета + морфология.

Идея: ценники Ленты в этих видео — ярко-красные на сером фоне.
Это сильный сигнал, который не зависит от обученной модели.

Алгоритм:
    1. HSV-маска красного (две зоны H: 0-10 и 170-180)
    2. Морфология (close + open) чтобы залатать дырки
    3. Connected components → bbox каждой компоненты
    4. Фильтр: площадь > 5000 px², aspect ratio в разумных пределах
    5. Расширение bbox вниз +50% чтобы захватить белую часть ценника

Считаем recall vs GT по IoU >= 0.3.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "redmask_detector"
OUT.mkdir(parents=True, exist_ok=True)

W4K, H4K = 3840, 2160


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


def detect_redmask(frame):
    """Возвращает список bbox-ов красных ценников.

    Используем HSV-маску красного цвета — основная характеристика
    акционных ценников Ленты.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask = m1 | m2
    # морфология
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    # connected components
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):  # 0 — фон
        x, y, w, h, area = stats[i]
        if area < 5000: continue
        # фильтр аспект ratio: 0.3 < w/h < 3 (ценник не сильно вытянут)
        ar = w / h if h > 0 else 0
        if not (0.25 < ar < 4): continue
        # Расширение вниз на 50% — захват белой части ценника
        ext_h = int(h * 1.5)
        bb = (x, y, x + w, min(H4K, y + ext_h))
        boxes.append(bb)
    return boxes, mask


def run_video(video_name: str, draw_every: int = 5):
    csv_path = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4_path = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    df = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

    cap = cv2.VideoCapture(str(mp4_path))
    total_gt = 0; matched_gt = 0; total_detected = 0; vis_saved = 0

    timestamps = sorted(df["frame_timestamp"].unique())
    for k, ts in enumerate(tqdm(timestamps, desc=video_name)):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue

        gts = df[df["frame_timestamp"] == ts]
        gt_boxes = [(r["x_min"], r["y_min"], r["x_max"], r["y_max"]) for _, r in gts.iterrows()]
        pred_boxes, mask = detect_redmask(frame)

        for gt in gt_boxes:
            best = max((iou(gt, p) for p in pred_boxes), default=0)
            if best >= 0.3: matched_gt += 1
        total_gt += len(gt_boxes)
        total_detected += len(pred_boxes)

        # сохраняем визуализации каждый N-й кадр
        if vis_saved < 8 and k % draw_every == 0:
            vis = frame.copy()
            for gt in gt_boxes:
                x1, y1, x2, y2 = map(int, gt)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 6)
            for p in pred_boxes:
                x1, y1, x2, y2 = map(int, p)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 4)
            vis_small = cv2.resize(vis, (1920, 1080))
            cv2.imwrite(str(OUT / f"{video_name}_ts{int(ts):06d}.jpg"), vis_small)
            vis_saved += 1
    cap.release()

    print(f"\n=== {video_name} ===")
    print(f"  Total GT bboxes:        {total_gt}")
    print(f"  Total predicted:        {total_detected}")
    print(f"  Matched IoU>=0.3:       {matched_gt}/{total_gt}  ({matched_gt/total_gt:.0%})")
    return {"video": video_name, "gt": total_gt, "pred": total_detected, "matched": matched_gt}


import json
results = [run_video(v) for v in ["26_12-20", "43_15"]]
total_gt = sum(r["gt"] for r in results)
total_matched = sum(r["matched"] for r in results)
total_pred = sum(r["pred"] for r in results)
print(f"\n=== TOTAL ===")
print(f"  GT:            {total_gt}")
print(f"  Pred:          {total_pred}")
print(f"  Matched:       {total_matched}/{total_gt}  ({total_matched/total_gt:.0%})")
(OUT / "report.json").write_text(json.dumps(results + [{"total_gt": total_gt, "total_matched": total_matched}], ensure_ascii=False, indent=2), encoding="utf-8")
