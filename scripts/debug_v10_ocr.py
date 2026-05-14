"""Debug: для 5 матченных ценников v10 показать что вернул OCR.
Цель: понять почему product_name = 0/29 — парсер выкидывает или OCR не находит?
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import match_rows
from pipeline_v10 import detect_yolo, rotate_bbox_back_to_original
from ultralytics import YOLO


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


# Загружаем pred + gt для 26_12-20
pred = pd.read_csv(ROOT / "outputs/pipeline_v10/26_12-20.csv", encoding="utf-8")
gt = pd.read_csv(ROOT / "data/labeled/26_12-20/26_12-20.csv", encoding="utf-8")
matches = match_rows(pred.copy(), gt.copy())
matched = [(g, p) for g, p, _ in matches if p is not None][:5]

# Для каждого — read frame, найти соответствующий bbox, OCR на rotated crop
cap = cv2.VideoCapture(str(ROOT / "data/labeled/26_12-20/26_12-20.mp4"))
ocr = RapidOCR()

for g_idx, p_idx in matched:
    p = pred.iloc[p_idx]; g = gt.iloc[g_idx]
    ts = p["frame_timestamp"]
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
    ok, frame = cap.read()
    if not ok: continue
    rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # rotation rotated coords ← pred bbox в original. Нам нужно rotated.
    # Inverse rotate_bbox_back_to_original: было (x,y)_rot -> (W-y, x)_orig
    # rotated has shape (W_orig, H_orig). Forward: (x,y)_orig -> (y, W_orig - x)_rot
    W_orig = frame.shape[1]
    x1_o = p["x_min"]; y1_o = p["y_min"]; x2_o = p["x_max"]; y2_o = p["y_max"]
    # bbox в rotated:
    x1_r = y1_o; y1_r = W_orig - x2_o
    x2_r = y2_o; y2_r = W_orig - x1_o
    x1, y1, x2, y2 = map(int, (min(x1_r,x2_r), min(y1_r,y2_r), max(x1_r,x2_r), max(y1_r,y2_r)))
    # +25% padding
    bw, bh = x2-x1, y2-y1
    H_r, W_r = rotated.shape[:2]
    x1 = max(0, x1 - int(bw*0.25)); y1 = max(0, y1 - int(bh*0.25))
    x2 = min(W_r, x2 + int(bw*0.25)); y2 = min(H_r, y2 + int(bh*0.25))
    crop = rotated[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"\nidx_pred={p_idx} crop EMPTY"); continue
    crop_up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    try: result, _ = ocr(crop_up, use_angle_cls=True)
    except TypeError:
        result, _ = ocr(crop_up)
    print(f"\n{'='*70}")
    print(f"PRED idx={p_idx} <-> GT idx={g_idx} | ts={ts}ms")
    print(f"  GT  product_name = {str(g.get('product_name',''))[:80]!r}")
    print(f"  GT  price_card   = {g.get('price_card','')}  price_default = {g.get('price_default','')}")
    print(f"  GT  barcode      = {g.get('barcode','')}  id_sku = {g.get('id_sku','')}")
    print(f"  GT  print_datetime = {g.get('print_datetime','')}")
    print(f"\n  PRED product_name = {str(p.get('product_name',''))[:80]!r}")
    print(f"  PRED price_card   = {p.get('price_card','')}  price_default = {p.get('price_default','')}")
    print(f"  PRED barcode      = {p.get('barcode','')}  id_sku = {p.get('id_sku','')}")
    print()
    print(f"  RAW OCR ({len(result) if result else 0} entries):")
    if result:
        # сортируем по y
        entries = sorted(result, key=lambda x: (min(p[1] for p in x[0]), min(p[0] for p in x[0])))
        for bb, text, conf in entries:
            ys = [p[1] for p in bb]; xs = [p[0] for p in bb]
            h_ent = max(ys) - min(ys)
            print(f"    h={int(h_ent):3d}  conf={float(conf):.2f}  text={text!r}")
cap.release()
