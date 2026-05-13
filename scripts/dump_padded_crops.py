"""Сохраняет на диск S2_pad50 крепы первых N ценников из каждого видео —
глазами проверить, виден ли там QR/штрихкод вообще и что нам в принципе светит.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "padded_crops"
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
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


for name in ("26_12-20", "43_15"):
    csv = ROOT / f"data/labeled/{name}/{name}.csv"
    mp4 = ROOT / f"data/labeled/{name}/{name}.mp4"
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    samples = df.sample(8, random_state=42).reset_index(drop=True)
    cap = cv2.VideoCapture(str(mp4))
    print(f"\n=== {name} ===")
    for i, r in samples.iterrows():
        ts = r["frame_timestamp"]
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        b = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
        x1, y1, x2, y2 = pad(b, frame.shape, 0.5)
        crop = frame[y1:y2, x1:x2]
        # удвоим для лучшей читаемости в md превью
        crop2 = cv2.resize(crop, (min(crop.shape[1] * 2, 700), int(min(crop.shape[1] * 2, 700) * crop.shape[0] / crop.shape[1])))
        path = OUT / f"{name}_sample{i}_ts{int(ts)}_barcode{str(r['barcode']).strip()[:13]}.jpg"
        cv2.imwrite(str(path), crop2)
        print(f"  {path.name}  raw={x2-x1}x{y2-y1}px  barcode_gt={str(r['barcode']).strip()}")
    cap.release()
