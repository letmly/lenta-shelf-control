"""Просто: вытащить ВСЕ ценники из GT как картинки в одну папку.

Никакого стэкинга, никакой обработки. Один кадр на GT-таймстемпе, кроп с
padding +50%. Файлы пронумерованы для удобного просмотра.

Каждый файл назван так:
    <video>__idx<NN>__ts<MS>__bc<BARCODE>.jpg

В каждой папке есть _README.txt с GT-данными.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "all_gt_crops"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = ["26_12-20", "43_15"]
PAD_PCT = 0.5


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def pad(b, shape, pct=PAD_PCT):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


for vname in VIDEOS:
    csv = ROOT / f"data/labeled/{vname}/{vname}.csv"
    mp4 = ROOT / f"data/labeled/{vname}/{vname}.mp4"
    out_dir = OUT / vname
    out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        print(f"cannot open {mp4}"); continue

    readme_lines = [f"# Все GT-ценники из {vname}", "",
                    "Каждый файл = один ценник из CSV, кадр на GT-таймстемпе, кроп с padding +50%.",
                    "",
                    "| # | ts (ms) | bbox размер | barcode (GT) | название (GT) |",
                    "|---|---|---|---|---|"]

    print(f"\n=== {vname}.mp4: {len(df)} ценников ===")
    for i, r in df.iterrows():
        ts = r["frame_timestamp"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
        gt_bc = str(r.get("barcode", "")).strip().replace(".0", "")
        gt_name = str(r.get("product_name", "")).strip()[:60]

        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        b = pad(bbox, frame.shape, PAD_PCT)
        crop = frame[b[1]:b[3], b[0]:b[2]]
        if crop.size == 0: continue
        bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        fname = f"{vname}__idx{i:02d}__ts{int(ts):06d}__bc{gt_bc}.jpg"
        cv2.imwrite(str(out_dir / fname), crop)
        readme_lines.append(f"| {i:02d} | {int(ts)} | {bw}×{bh} | `{gt_bc}` | {gt_name} |")
    cap.release()
    (out_dir / "_README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"  -> {out_dir}")

print(f"\n[OK] all dumps -> {OUT}")
