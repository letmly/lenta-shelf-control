"""DIAGNOSTIC: запускаем парсер на GT-bbox (идеальный detector).

Это даёт ВЕРХНЮЮ ГРАНИЦУ нашего OCR+parser stack без проблем детектора.
Если метрика тут хорошая → проблема в detector. Если плохая → в OCR/parser.

Padding +25% во все стороны (как мы видели в ocr_compare работает),
upscale x3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_v5 import parse_ocr, CSV_COLUMNS
from evaluate import match_rows, compare_fields

OUT = ROOT / "outputs" / "diag_gt_bbox"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def process_via_gt_bbox(video_name, filename_in_csv):
    csv_path = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4_path = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    out_csv = OUT / f"{video_name}.csv"
    df_gt = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min","y_min","x_max","y_max","frame_timestamp"):
        df_gt[c] = df_gt[c].apply(parse_ru)
    df_gt_v = df_gt.dropna(subset=["x_min","y_min","x_max","y_max","frame_timestamp"]).reset_index(drop=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened(): return None
    ocr = RapidOCR()
    rows = []
    for i, g in tqdm(df_gt_v.iterrows(), total=len(df_gt_v), desc=video_name):
        ts = g["frame_timestamp"]
        gx1, gy1, gx2, gy2 = int(g["x_min"]), int(g["y_min"]), int(g["x_max"]), int(g["y_max"])
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        bw, bh = gx2-gx1, gy2-gy1
        # padding +25% во все стороны
        H, W = frame.shape[:2]
        x1 = max(0, gx1 - int(bw * 0.25))
        y1 = max(0, gy1 - int(bh * 0.25))
        x2 = min(W, gx2 + int(bw * 0.25))
        y2 = min(H, gy2 + int(bh * 0.25))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30: continue
        # upscale x3
        crop_up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        try: result, _ = ocr(crop_up)
        except: result = None
        color = str(g.get("color", "red")).strip() or "red"
        fields = parse_ocr(result, color)
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(ts)
        fields["x_min"] = round(float(gx1), 1)
        fields["y_min"] = round(float(gy1), 1)
        fields["x_max"] = round(float(gx2), 1)
        fields["y_max"] = round(float(gy2), 1)
        rows.append(fields)
    cap.release()
    df_pred = pd.DataFrame(rows, columns=CSV_COLUMNS).fillna("")
    df_pred.to_csv(out_csv, index=False, encoding="utf-8")

    # evaluate
    matches = match_rows(df_pred.copy(), df_gt.copy())
    scores = []
    for gt_idx, p_idx, _ in matches:
        if p_idx is None: scores.append(0); continue
        s, _ = compare_fields(df_pred.iloc[p_idx], df_gt.iloc[gt_idx])
        scores.append(s)
    matched = sum(1 for m in matches if m[1] is not None)
    qual = sum(1 for s in scores if s >= 0.8)
    print(f"\n=== {video_name} ===")
    print(f"  pred={len(df_pred)}  gt={len(df_gt)}  matched={matched}  mean={sum(scores)/len(scores) if scores else 0:.0%}  qual>=80%={qual}/{len(df_gt)}")
    return {"name": video_name, "pred": len(df_pred), "gt": len(df_gt),
            "matched": matched, "qualified": qual,
            "mean_score": sum(scores)/len(scores) if scores else 0}


# Прогоним на 26_12-20 (главный)
results = [process_via_gt_bbox("26_12-20", "26_12-20.mp4")]
total_gt = sum(r["gt"] for r in results)
total_qual = sum(r["qualified"] for r in results)
print(f"\n=== TOTAL ===")
print(f"  Qualified >=80%:    {total_qual}/{total_gt} ({total_qual/total_gt:.0%})")
