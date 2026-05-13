"""QR/штрихкод rescue: пробуем разные стратегии декодирования.

Гипотезы:
    S1. baseline crop (tight bbox)                              — 0% (известно)
    S2. crop с padding +50% во все стороны
    S3. crop с padding +100%
    S4. полный кадр без crop (находим все коды)
    S5. S2 + препроцессинг (upscale 2x, grayscale, Otsu, adaptive)

Для каждой стратегии меряем:
    * сколько GT-ценников вообще удалось декодировать
    * сколько из них совпали с GT-штрихкодом (это — то что орги называют «первичным ключом»)
    * сколько распознали как QR vs как EAN13 vs прочие типы

Работаем только на 26_12-20 и 43_15 (для них видео соответствует разметке).
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "qr_rescue"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    (ROOT / "data/labeled/26_12-20/26_12-20.mp4", ROOT / "data/labeled/26_12-20/26_12-20.csv"),
    (ROOT / "data/labeled/43_15/43_15.mp4",        ROOT / "data/labeled/43_15/43_15.csv"),
]


# ----------------------------- helpers ----------------------------------------

def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def load_gt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype(str).str.strip()
    return df


def grab_frame(cap: cv2.VideoCapture, ts_ms: float):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
    ok, frame = cap.read()
    return frame if ok else None


def pad_bbox(bbox, frame_shape, pct: float):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def try_decode(img) -> list[dict]:
    """Возвращает список расшифрованных кодов с типом и payload."""
    if img is None or img.size == 0:
        return []
    res = pyzbar_decode(img)
    return [
        {
            "type": d.type,
            "data": d.data.decode("utf-8", errors="ignore"),
            "rect": (d.rect.left, d.rect.top, d.rect.left + d.rect.width, d.rect.top + d.rect.height),
        }
        for d in res
    ]


def preprocess_variants(crop):
    yield "raw", crop
    yield "2x", cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yield "gray", gray
    g2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    yield "2x_gray", g2
    _, otsu = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield "2x_otsu", otsu
    adap = cv2.adaptiveThreshold(g2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)
    yield "2x_adap", adap


def best_decoded(crop):
    """S5: пробуем все препроцессинги, возвращаем первый, который что-то декодировал."""
    for tag, variant in preprocess_variants(crop):
        decoded = try_decode(variant)
        if decoded:
            return tag, decoded
    return None, []


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------- strategies -------------------------------------

def run_strategies_for_video(video: Path, csv: Path) -> dict:
    df = load_gt(csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {"video": video.name, "error": "cannot open"}

    stats = {s: {"decoded": 0, "matched_barcode": 0, "types": Counter()} for s in
             ["S1_tight", "S2_pad50", "S3_pad100", "S4_full_frame", "S5_preproc"]}

    total = len(df)
    # для S4 (full-frame) — кэшируем декод по timestamp, тк один кадр содержит несколько ценников
    full_frame_cache = {}

    for _, r in tqdm(df.iterrows(), total=total, desc=video.name):
        ts = r["frame_timestamp"]
        gt_barcode = r["barcode"]
        frame = grab_frame(cap, ts)
        if frame is None: continue
        bbox_tight = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))

        # ---- S1: tight crop ----
        c = frame[bbox_tight[1]:bbox_tight[3], bbox_tight[0]:bbox_tight[2]]
        d1 = try_decode(c)
        if d1:
            stats["S1_tight"]["decoded"] += 1
            for x in d1: stats["S1_tight"]["types"][x["type"]] += 1
            if any(x["data"].strip() == gt_barcode for x in d1):
                stats["S1_tight"]["matched_barcode"] += 1

        # ---- S2: pad +50% ----
        b2 = pad_bbox(bbox_tight, frame.shape, 0.5)
        c2 = frame[b2[1]:b2[3], b2[0]:b2[2]]
        d2 = try_decode(c2)
        if d2:
            stats["S2_pad50"]["decoded"] += 1
            for x in d2: stats["S2_pad50"]["types"][x["type"]] += 1
            if any(x["data"].strip() == gt_barcode for x in d2):
                stats["S2_pad50"]["matched_barcode"] += 1

        # ---- S3: pad +100% ----
        b3 = pad_bbox(bbox_tight, frame.shape, 1.0)
        c3 = frame[b3[1]:b3[3], b3[0]:b3[2]]
        d3 = try_decode(c3)
        if d3:
            stats["S3_pad100"]["decoded"] += 1
            for x in d3: stats["S3_pad100"]["types"][x["type"]] += 1
            if any(x["data"].strip() == gt_barcode for x in d3):
                stats["S3_pad100"]["matched_barcode"] += 1

        # ---- S4: full frame (один декод на ts, кэшируем) ----
        if ts not in full_frame_cache:
            full_frame_cache[ts] = try_decode(frame)
        d4 = full_frame_cache[ts]
        # «Декодировано для этого GT-ценника» = хотя бы один код в кадре пересекается с tight bbox
        d4_for_this_tag = [x for x in d4 if bbox_iou(bbox_tight, x["rect"]) > 0.0]
        if d4_for_this_tag:
            stats["S4_full_frame"]["decoded"] += 1
            for x in d4_for_this_tag: stats["S4_full_frame"]["types"][x["type"]] += 1
            if any(x["data"].strip() == gt_barcode for x in d4_for_this_tag):
                stats["S4_full_frame"]["matched_barcode"] += 1

        # ---- S5: pad+50% + препроцессинг ----
        tag, d5 = best_decoded(c2)
        if d5:
            stats["S5_preproc"]["decoded"] += 1
            for x in d5: stats["S5_preproc"]["types"][x["type"]] += 1
            if any(x["data"].strip() == gt_barcode for x in d5):
                stats["S5_preproc"]["matched_barcode"] += 1

    cap.release()

    # нормализуем types в обычный dict для json
    out_stats = {}
    for k, v in stats.items():
        out_stats[k] = {
            "decoded": v["decoded"],
            "decoded_rate": v["decoded"] / total if total else 0,
            "matched_barcode": v["matched_barcode"],
            "matched_barcode_rate": v["matched_barcode"] / total if total else 0,
            "types": dict(v["types"]),
        }

    return {"video": video.name, "total_gt": total, "strategies": out_stats}


# ----------------------------- main -------------------------------------------

def main():
    t0 = time.time()
    report = {"started_at": t0, "videos": []}
    for video, csv in VIDEOS:
        if not video.exists() or not csv.exists():
            print(f"skip {video.name}")
            continue
        r = run_strategies_for_video(video, csv)
        report["videos"].append(r)
        print(f"\n=== {r['video']} (GT={r['total_gt']}) ===")
        for s, st in r["strategies"].items():
            print(f"  {s:18s} decoded={st['decoded']:3d} ({st['decoded_rate']:.0%})  "
                  f"barcode-match={st['matched_barcode']:3d} ({st['matched_barcode_rate']:.0%})  "
                  f"types={st['types']}")
    report["elapsed_sec"] = time.time() - t0
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT/'report.json'}")


if __name__ == "__main__":
    main()
