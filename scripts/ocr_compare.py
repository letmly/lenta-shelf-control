"""Сравнение OCR качества: GT-bbox vs detector-bbox.

Для матченных GT-ценников:
    1. Возьмём GT-bbox, вырежем crop, прогоним OCR → сохраним результат.
    2. Возьмём наш detector-bbox (нашего trackа), вырежем crop, OCR → сохраним.
    3. Сравним: какой подход даёт больше текста.

Цель: понять, виноват ли DETECTOR (плохой bbox) или OCR (слабый на наших данных).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ocr_compare"
OUT.mkdir(parents=True, exist_ok=True)

video = "26_12-20"
gt_df = pd.read_csv(ROOT / f"data/labeled/{video}/{video}.csv", encoding="utf-8")
pred_df = pd.read_csv(ROOT / f"outputs/pipeline_v5/{video}.csv", encoding="utf-8")

def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None

for c in ("x_min","y_min","x_max","y_max","frame_timestamp"):
    gt_df[c] = gt_df[c].apply(parse_ru)
    pred_df[c] = pred_df[c].apply(parse_ru)
gt_df = gt_df.dropna(subset=["x_min","y_min","x_max","y_max","frame_timestamp"]).reset_index(drop=True)

cap = cv2.VideoCapture(str(ROOT / f"data/labeled/{video}/{video}.mp4"))
ocr = RapidOCR()

# берём первые 4 GT-ценника
for i in range(min(4, len(gt_df))):
    g = gt_df.iloc[i]
    print(f"\n{'='*80}")
    print(f"GT idx={i}  ts={int(g['frame_timestamp'])}ms  bc={g['barcode']}  product={str(g['product_name'])[:50]}")
    cap.set(cv2.CAP_PROP_POS_MSEC, float(g["frame_timestamp"]))
    ok, frame = cap.read()
    if not ok: continue

    # ----- GT-bbox crop -----
    gx1, gy1, gx2, gy2 = map(int, (g["x_min"], g["y_min"], g["x_max"], g["y_max"]))
    crop_gt = frame[gy1:gy2, gx1:gx2]
    # с padding 50% во все стороны
    bw, bh = gx2-gx1, gy2-gy1
    pgx1 = max(0, gx1 - int(bw*0.25)); pgy1 = max(0, gy1 - int(bh*0.25))
    pgx2 = min(frame.shape[1], gx2 + int(bw*0.25)); pgy2 = min(frame.shape[0], gy2 + int(bh*0.25))
    crop_gt_pad = frame[pgy1:pgy2, pgx1:pgx2]

    # OCR на 3 версиях: оригинал, +pad, +pad x3
    versions = {
        "GT_orig":     crop_gt,
        "GT_pad25":    crop_gt_pad,
        "GT_pad25_x3": cv2.resize(crop_gt_pad, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC),
    }
    # Сохраняем
    for name, img in versions.items():
        cv2.imwrite(str(OUT / f"idx{i:02d}_{name}.jpg"), img)
        try: result, _ = ocr(img)
        except: result = None
        n_entries = len(result) if result else 0
        texts = [r[1] for r in (result or [])]
        print(f"  [{name:14s}] size={img.shape[1]}x{img.shape[0]}  OCR entries={n_entries}  texts={texts[:8]}")

    # ----- наш detector-bbox через pred с минимальным IoU расстоянием -----
    # ищем pred-строку для этого GT по spatial-temporal матчингу
    same_ts = pred_df[abs(pred_df["frame_timestamp"] - g["frame_timestamp"]) < 1500]
    if len(same_ts) > 0:
        # ближайший по центру
        gcx, gcy = (gx1+gx2)/2, (gy1+gy2)/2
        same_ts = same_ts.copy()
        same_ts["dist"] = np.sqrt(((same_ts["x_min"]+same_ts["x_max"])/2 - gcx)**2 +
                                   ((same_ts["y_min"]+same_ts["y_max"])/2 - gcy)**2)
        best = same_ts.nsmallest(1, "dist").iloc[0]
        bx1, by1, bx2, by2 = map(int, (best["x_min"], best["y_min"], best["x_max"], best["y_max"]))
        crop_pred = frame[by1:by2, bx1:bx2]
        cv2.imwrite(str(OUT / f"idx{i:02d}_PRED.jpg"), crop_pred)
        try: result, _ = ocr(cv2.resize(crop_pred, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
        except: result = None
        n = len(result) if result else 0
        texts = [r[1] for r in (result or [])]
        print(f"  [PRED_x3       ] size={crop_pred.shape[1]}x{crop_pred.shape[0]} bbox=({bx1},{by1},{bx2},{by2})  OCR entries={n}  texts={texts[:8]}")
cap.release()
print(f"\n[OK] crops -> {OUT}")
