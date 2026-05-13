"""Proof-of-concept perspective rectification одного ценника.

Цель: взять idx02 из 26_12-20 (где QR глазами читается) и проверить:
    1. можем ли мы автоматически найти 4 угла ценника
    2. выпрямить homography'ей в прямоугольник
    3. декодировать QR на выпрямленной версии

Если на одном ценнике зайдёт — пишем универсальную версию.

Подходы к нахождению углов (применяем по очереди):
    A. HSV color mask (красный) → contour → approxPolyDP
    B. Adaptive threshold → contour → approxPolyDP
    C. Canny + Hough lines → пересечения (fallback)
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "rectify_poc"
OUT.mkdir(parents=True, exist_ok=True)

# Берём ВСЕ ценники из labeled (запускаем POC на всех чтобы не было cherry-pick)
VIDEOS = ["26_12-20", "43_15"]
PAD_PCT = 0.6  # чуть больше чем 0.5, чтобы захватить кончики ценника


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def pad(b, shape, pct):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def order_corners(pts):
    """Сортирует 4 точки как (tl, tr, br, bl)."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def find_quad_by_color(crop):
    """Находим красный ценник через HSV маску и approxPolyDP."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # красный — две зоны H
    m1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    mask = m1 | m2
    # морфология чтобы залатать дырки
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, mask
    # самый большой контур
    big = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(big)
    if area < 1000: return None, mask
    # approxPolyDP с разными эпсилон до получения 4 точек
    peri = cv2.arcLength(big, True)
    for eps_factor in (0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        approx = cv2.approxPolyDP(big, eps_factor * peri, True)
        if len(approx) == 4:
            return approx, mask
    # если не получилось 4 — вернём minAreaRect (повёрнутый прямоугольник)
    box = cv2.minAreaRect(big)
    box = cv2.boxPoints(box)
    return box.astype(np.int32).reshape(-1, 1, 2), mask


def warp_to_rect(img, quad, scale=2.0):
    """Применяет homography чтобы привести quad к прямоугольнику."""
    src = order_corners(quad)
    # размер целевого прямоугольника = по средним длинам сторон
    w1 = np.linalg.norm(src[0] - src[1])
    w2 = np.linalg.norm(src[3] - src[2])
    h1 = np.linalg.norm(src[0] - src[3])
    h2 = np.linalg.norm(src[1] - src[2])
    out_w = int(max(w1, w2) * scale)
    out_h = int(max(h1, h2) * scale)
    if out_w < 50 or out_h < 50: return None
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def decode_all(img) -> set:
    res = set()
    variants = [img,
                cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]
    for v in variants:
        try:
            for d in pyzbar_decode(v): res.add(d.data.decode("utf-8", errors="ignore").strip())
        except Exception: pass
        if HAS_ZXING:
            try:
                for r in zxingcpp.read_barcodes(v): res.add(r.text.strip())
            except Exception: pass
        if HAS_PYBOOF:
            try:
                gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if v.ndim == 3 else v
                pb_img = pb.ndarray_to_boof(gray)
                PB_QR.detect(pb_img)
                for det in PB_QR.detections:
                    if det.message: res.add(det.message.strip())
            except Exception: pass
    return res


def process(video_name: str):
    csv = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4 = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    out_dir = OUT / video_name
    out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    cap = cv2.VideoCapture(str(mp4))

    decoded_total = 0
    matched_total = 0
    quad_found = 0
    rows = []
    for i, r in df.iterrows():
        ts = r["frame_timestamp"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
        gt_bc = str(r.get("barcode", "")).strip().replace(".0", "")
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        b = pad(bbox, frame.shape, PAD_PCT)
        crop = frame[b[1]:b[3], b[0]:b[2]]
        if crop.size == 0: continue

        quad, mask = find_quad_by_color(crop)
        warped = None
        decoded = set()
        if quad is not None and len(quad) == 4:
            quad_found += 1
            warped = warp_to_rect(crop, quad.reshape(-1, 2), scale=2.5)
            if warped is not None:
                decoded = decode_all(warped)

        if decoded: decoded_total += 1
        matched = gt_bc in decoded
        if matched: matched_total += 1

        # сохраняем артефакты для первых 8 и для всех MATCH
        if i < 8 or matched:
            vis = crop.copy()
            if quad is not None:
                cv2.drawContours(vis, [quad.astype(np.int32).reshape(-1, 1, 2)], -1, (0, 255, 0), 4)
            tag = "MATCH" if matched else ("dec" if decoded else ("quad" if quad is not None else "noquad"))
            cv2.imwrite(str(out_dir / f"idx{i:02d}_GT{gt_bc}_{tag}_01_crop_with_quad.jpg"), vis)
            cv2.imwrite(str(out_dir / f"idx{i:02d}_GT{gt_bc}_{tag}_02_redmask.jpg"), mask)
            if warped is not None:
                cv2.imwrite(str(out_dir / f"idx{i:02d}_GT{gt_bc}_{tag}_03_warped.jpg"), warped)

        rows.append({"idx": int(i), "gt_barcode": gt_bc, "quad_found": quad is not None,
                     "decoded": list(decoded), "matched": matched})

    cap.release()
    total = len(df)
    print(f"\n=== {video_name} ===")
    print(f"  total GT:        {total}")
    print(f"  quad found:      {quad_found}  ({quad_found/total:.0%})")
    print(f"  decoded any QR:  {decoded_total}  ({decoded_total/total:.0%})")
    print(f"  matched GT-bc:   {matched_total}  ({matched_total/total:.0%})")
    # печатаем удачные matched
    matched_rows = [r for r in rows if r["matched"]]
    if matched_rows:
        print(f"  MATCHED ценники:")
        for r in matched_rows:
            print(f"    idx{r['idx']}  GT={r['gt_barcode']}  decoded={r['decoded']}")
    return {"video": video_name, "total": total, "quad_found": quad_found,
            "decoded_total": decoded_total, "matched_total": matched_total,
            "rows": rows}


import json
results = []
for v in VIDEOS:
    results.append(process(v))
(OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] {OUT/'report.json'}")
