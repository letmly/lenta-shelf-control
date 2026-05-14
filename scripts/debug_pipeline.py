"""Debug: для нескольких matched ценников распечатать
GT vs pred + raw OCR output. Цель — понять что улучшать в парсере.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

import sys
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_v2 import detect_redmask, nms, parse_ocr, _items_from_ocr

ROOT = Path(__file__).resolve().parent.parent


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def pad(b, shape, pct=0.5):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


# Берём 5 ценников из 43_15 и прогоняем full debug pipeline на каждом
csv = ROOT / "data/labeled/43_15/43_15.csv"
mp4 = ROOT / "data/labeled/43_15/43_15.mp4"
df = pd.read_csv(csv, encoding="utf-8")
for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
    df[c] = df[c].apply(parse_ru)
df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

ocr = RapidOCR()
cap = cv2.VideoCapture(str(mp4))

GT_FIELDS = ['product_name','price_default','price_card','discount_amount','barcode','id_sku','print_datetime']

# Для каждого из 4 ценников: используем bbox от RED-DETECTOR (не GT), потом OCR
# Это ровно то что делает pipeline_v2

for i in range(min(4, len(df))):
    r = df.iloc[i]
    ts = r["frame_timestamp"]
    gt_bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
    ok, frame = cap.read()
    if not ok: continue

    print(f"\n{'='*70}")
    print(f"idx={i}  ts={int(ts)}ms")
    print(f"GT bbox: {gt_bbox}  ({gt_bbox[2]-gt_bbox[0]}x{gt_bbox[3]-gt_bbox[1]})")
    print(f"GT:")
    for f in GT_FIELDS:
        v = r.get(f, "")
        if pd.notna(v) and str(v).strip() not in ("", "нет"):
            print(f"  {f:18s} = {v!r}")

    # Прогоняем как pipeline_v2: detector → выбираем bbox с максимальным IoU к GT
    dets = nms(detect_redmask(frame), iou_th=0.3)
    if not dets:
        print(f"\n!! Detector нашёл 0 bbox на этом кадре")
        continue
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter/union if union > 0 else 0
    ious = [iou(gt_bbox, d) for d in dets]
    if max(ious) < 0.2:
        print(f"\n!! Detector не нашёл ценник близкий к GT (max IoU={max(ious):.2f})")
        continue
    det = dets[np.argmax(ious)]
    print(f"\nDetected bbox: {det} (IoU={max(ious):.2f})")

    x1, y1, x2, y2 = det
    crop = frame[y1:y2, x1:x2]
    crop_up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    print(f"Crop (after upscale x2): {crop_up.shape[1]}x{crop_up.shape[0]}")

    cv2.imwrite(str(ROOT / f"outputs/debug_pipeline/idx{i:02d}_crop_up.jpg"), crop_up)

    result, _ = ocr(crop_up)
    items = _items_from_ocr(result)
    print(f"\nOCR returned {len(items)} entries:")
    items.sort(key=lambda x: (x["y1"], x["x1"]))
    for it in items:
        print(f"  [y={it['y1']:4.0f}-{it['y2']:4.0f}  size={int(it['w']):3d}x{int(it['h']):3d}  conf={it['conf']:.2f}]  {it['text']!r}")

    parsed = parse_ocr(result)
    print(f"\nPARSED:")
    for f in GT_FIELDS:
        gt_val = str(r.get(f, ""))
        pred_val = parsed.get(f, "")
        match = "OK" if str(pred_val).lower().replace(",", ".") == gt_val.lower().replace(",", ".") else "MISS"
        print(f"  {f:18s} pred={pred_val!r:<40} gt={gt_val!r:<25} [{match}]")

cap.release()
print("\n[OK] crops saved to outputs/debug_pipeline/")
Path(ROOT / "outputs/debug_pipeline").mkdir(parents=True, exist_ok=True)
