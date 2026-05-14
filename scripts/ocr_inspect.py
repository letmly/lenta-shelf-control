"""Смотрим что OCR возвращает на нескольких GT-кропах.

Цель: понять структуру OCR-выхода для написания умного парсера.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ocr_inspect"
OUT.mkdir(parents=True, exist_ok=True)


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


# Берём 5 ценников из 43_15
csv = ROOT / "data/labeled/43_15/43_15.csv"
mp4 = ROOT / "data/labeled/43_15/43_15.mp4"
df = pd.read_csv(csv, encoding="utf-8")
for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
    df[c] = df[c].apply(parse_ru)
df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

ocr = RapidOCR()
cap = cv2.VideoCapture(str(mp4))

GT_FIELDS = ['product_name','price_default','price_card','discount_amount','barcode','id_sku','print_datetime']

for i in range(min(5, len(df))):
    r = df.iloc[i]
    ts = r["frame_timestamp"]
    bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
    ok, frame = cap.read()
    if not ok: continue
    b = pad(bbox, frame.shape, 0.5)
    crop = frame[b[1]:b[3], b[0]:b[2]]

    print(f"\n{'='*70}")
    print(f"idx={i}  ts={int(ts)}ms  bbox={bbox}  crop={crop.shape[1]}x{crop.shape[0]}")
    print(f"\nGT:")
    for f in GT_FIELDS:
        v = r.get(f, "")
        if not pd.isna(v) and str(v).strip() not in ("", "нет"):
            print(f"  {f:18s} = {v!r}")

    cv2.imwrite(str(OUT / f"idx{i:02d}_crop.jpg"), crop)
    result, _ = ocr(crop)
    print(f"\nOCR raw output ({len(result) if result else 0} entries):")
    if result:
        # сортируем по y_top, затем x_left
        items = []
        for entry in result:
            bb, text, conf = entry
            xs = [p[0] for p in bb]; ys = [p[1] for p in bb]
            x1 = min(xs); y1 = min(ys); x2 = max(xs); y2 = max(ys)
            items.append({"text": str(text), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "w": x2-x1, "h": y2-y1, "conf": float(conf)})
        items.sort(key=lambda x: (x["y1"], x["x1"]))
        for it in items:
            print(f"  [y={it['y1']:4.0f}-{it['y2']:4.0f} x={it['x1']:4.0f}-{it['x2']:4.0f}  "
                  f"size={int(it['w'])}x{int(it['h']):3d}  conf={it['conf']:.2f}]  {it['text']!r}")
cap.release()
print(f"\n[OK] crops -> {OUT}")
