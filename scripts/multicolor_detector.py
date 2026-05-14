"""Multicolor детектор: ловит ВСЕ типы ценников Ленты.

Из docs/03-templates.md и docs/case/ГМ для ТК.pptx:
    - АПЦ красный (акция)
    - РПЦ белый (регулярная цена)
    - МНЦ белый (свежие)
    - МНЦ жёлтый (выпечка)
    - МНЦ зелёный (овощи-фрукты)
    - Распродажа, BOGOF — обычно красные

Алгоритм:
    1. Раздельные HSV-маски: red, yellow, green
    2. Дополнительно — маска БЕЛОГО (high V, low S)
    3. Морфология + connected components на каждой маске
    4. Объединяем bbox-ы, фильтруем по площади и aspect
    5. Расширение bbox вниз на 50% (захват белой части у красных)
    6. NMS чтобы убрать дубли между color-masks
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "multicolor_detector"
OUT.mkdir(parents=True, exist_ok=True)

HSV_RANGES = {
    "red":    [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))],
    "yellow": [((18, 80, 80), (35, 255, 255))],
    "green":  [((40, 60, 60), (85, 255, 255))],
    "white":  [((0, 0, 180), (180, 60, 255))],  # высокая V, низкая S
}


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


def nms(boxes_with_color, iou_th=0.3):
    """boxes_with_color = [(bbox, color), ...]. Возвращаем без перекрытий."""
    if not boxes_with_color: return []
    areas = [(b[0][2]-b[0][0])*(b[0][3]-b[0][1]) for b in boxes_with_color]
    idx = sorted(range(len(boxes_with_color)), key=lambda i: -areas[i])
    keep = []
    for i in idx:
        ok = True
        for j in keep:
            if iou(boxes_with_color[i][0], boxes_with_color[j][0]) > iou_th:
                ok = False; break
        if ok: keep.append(i)
    return [boxes_with_color[i] for i in keep]


def detect_one_color(frame, hsv, ranges, color_name, min_area=5000):
    """Детект ценников одного цвета."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, lo, hi)
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H = frame.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area: continue
        ar = w / h if h > 0 else 0
        if not (0.25 < ar < 4): continue
        # для коротких/широких ценников (красные акционные) — extend вниз
        # для квадратных/высоких (мини ценники) — оставляем
        if color_name == "red" and ar > 1.0:
            ext_h = int(h * 1.5)
            box = [x, y, x + w, min(H, y + ext_h)]
        else:
            box = [x, y, x + w, y + h]
        boxes.append((box, color_name))
    return boxes, mask


def detect_multicolor(frame, min_area=5000):
    """Главный метод детекции."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    all_boxes = []
    for color, ranges in HSV_RANGES.items():
        # для white — больший min_area (фон легко проходит)
        ma = min_area * 3 if color == "white" else min_area
        boxes, _ = detect_one_color(frame, hsv, ranges, color, min_area=ma)
        all_boxes.extend(boxes)
    # NMS между color-масками
    return nms(all_boxes, iou_th=0.3)


def run_video(video_name: str, draw_first_n=8):
    csv_path = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4_path = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    df = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

    cap = cv2.VideoCapture(str(mp4_path))
    total_gt = 0; matched_gt = 0; total_detected = 0
    color_counts = {"red": 0, "yellow": 0, "green": 0, "white": 0}
    vis_saved = 0
    timestamps = sorted(df["frame_timestamp"].unique())

    for k, ts in enumerate(tqdm(timestamps, desc=video_name)):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        gts = df[df["frame_timestamp"] == ts]
        gt_boxes = [(r["x_min"], r["y_min"], r["x_max"], r["y_max"]) for _, r in gts.iterrows()]
        pred = detect_multicolor(frame)
        pred_boxes = [b for b, _ in pred]
        for b, c in pred: color_counts[c] += 1

        for gt in gt_boxes:
            best = max((iou(gt, p) for p in pred_boxes), default=0)
            if best >= 0.3: matched_gt += 1
        total_gt += len(gt_boxes)
        total_detected += len(pred_boxes)

        if vis_saved < draw_first_n:
            vis = frame.copy()
            for gt in gt_boxes:
                x1, y1, x2, y2 = map(int, gt)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 6)
            color_bgr = {"red": (0, 0, 255), "yellow": (0, 255, 255),
                         "green": (0, 255, 0), "white": (255, 255, 255)}
            for b, c in pred:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color_bgr[c], 4)
                cv2.putText(vis, c, (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr[c], 2)
            vis_small = cv2.resize(vis, (1920, 1080))
            cv2.imwrite(str(OUT / f"{video_name}_ts{int(ts):06d}.jpg"), vis_small)
            vis_saved += 1
    cap.release()

    print(f"\n=== {video_name} ===")
    print(f"  GT: {total_gt}")
    print(f"  Predicted: {total_detected}")
    print(f"  Matched IoU>=0.3: {matched_gt}/{total_gt} ({matched_gt/total_gt:.0%})")
    print(f"  By color: {color_counts}")
    return {"video": video_name, "gt": total_gt, "matched": matched_gt,
            "pred": total_detected, "color_counts": color_counts}


results = [run_video(v) for v in ["26_12-20", "43_15"]]
total_gt = sum(r["gt"] for r in results); total_matched = sum(r["matched"] for r in results)
print(f"\n=== TOTAL ===")
print(f"  Matched: {total_matched}/{total_gt} ({total_matched/total_gt:.0%})")
(OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
