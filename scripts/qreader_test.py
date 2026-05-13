"""Тест QReader — последняя серьёзная попытка декодировать QR.

QReader = YOLOv8-детектор QR + автоматический preprocessing chain + pyzbar.
Прогон на 100 GT-ценниках из 26_12-20 + 43_15.

Подаём в QReader несколько вариантов входа:
    A. Сырой кроп ценника (padded GT bbox)
    B. Warped кроп (rectified ценник)
    C. Warped увеличенный 2x
    D. Warped в grayscale + bilateral filter
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from qreader import QReader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "qreader_test"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = OUT / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

print("Initializing QReader (downloads YOLOv8 weights on first run)...")
qr = QReader(model_size="s", min_confidence=0.3, reencode_to="utf-8")
print("QReader initialized.")

VIDEOS = ["26_12-20", "43_15"]
PAD_PCT = 0.6


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
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1); diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def warp_full_tag(crop):
    """Возвращает warped изображение всего ценника или None."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, -5)
    k = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=3)
    cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    big = max(cs, key=cv2.contourArea)
    if cv2.contourArea(big) < 5000: return None
    peri = cv2.arcLength(big, True)
    quad = None
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(big, eps * peri, True)
        if len(approx) == 4:
            quad = approx; break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(big)).astype(np.int32).reshape(-1, 1, 2)
    src = order_corners(quad.reshape(-1, 2))
    w1 = np.linalg.norm(src[0] - src[1]); w2 = np.linalg.norm(src[3] - src[2])
    h1 = np.linalg.norm(src[0] - src[3]); h2 = np.linalg.norm(src[1] - src[2])
    out_w = int(max(w1, w2) * 2.5); out_h = int(max(h1, h2) * 2.5)
    if out_w < 50 or out_h < 50: return None
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(crop, M, (out_w, out_h))


def try_qreader(img):
    """Прогон QReader. Возвращает список декодированных строк."""
    try:
        decoded = qr.detect_and_decode(image=img)
        return [d for d in decoded if d]
    except Exception as e:
        return []


def run_video(video_name: str) -> dict:
    csv = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4 = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype(str).str.strip().str.replace(".0", "", regex=False)

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened(): return {}
    total = len(df)
    stats = {k: {"decoded": 0, "matched": 0} for k in
             ["A_raw_crop", "B_warped", "C_warped_2x", "D_warped_gray", "ANY"]}
    matched_rows = []

    for i, r in tqdm(df.iterrows(), total=total, desc=video_name):
        ts = r["frame_timestamp"]; gt_bc = r["barcode"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        b = pad(bbox, frame.shape, PAD_PCT)
        crop = frame[b[1]:b[3], b[0]:b[2]]
        if crop.size == 0: continue

        warped = warp_full_tag(crop)
        any_decoded = set()

        # A. raw crop
        codes = try_qreader(crop)
        if codes: stats["A_raw_crop"]["decoded"] += 1
        if gt_bc in codes: stats["A_raw_crop"]["matched"] += 1
        any_decoded.update(codes)

        if warped is not None:
            # B. warped
            codes = try_qreader(warped)
            if codes: stats["B_warped"]["decoded"] += 1
            if gt_bc in codes: stats["B_warped"]["matched"] += 1
            any_decoded.update(codes)
            # C. warped 2x
            big = cv2.resize(warped, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
            codes = try_qreader(big)
            if codes: stats["C_warped_2x"]["decoded"] += 1
            if gt_bc in codes: stats["C_warped_2x"]["matched"] += 1
            any_decoded.update(codes)
            # D. warped gray + bilateral
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 9, 75, 75)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            codes = try_qreader(gray3)
            if codes: stats["D_warped_gray"]["decoded"] += 1
            if gt_bc in codes: stats["D_warped_gray"]["matched"] += 1
            any_decoded.update(codes)

        # ANY
        if any_decoded: stats["ANY"]["decoded"] += 1
        if gt_bc in any_decoded:
            stats["ANY"]["matched"] += 1
            matched_rows.append({"idx": i, "GT": gt_bc, "decoded": list(any_decoded)})
            # сохраним MATCH-ценник
            cv2.imwrite(str(SAMPLES_DIR / f"{video_name}_idx{i:02d}_MATCH_GT{gt_bc}.jpg"),
                        warped if warped is not None else crop)
        elif any_decoded:
            # сохраним 'decoded but not matched' для первых нескольких
            if len(list(SAMPLES_DIR.glob(f"{video_name}_idx*_decmiss*"))) < 5:
                cv2.imwrite(str(SAMPLES_DIR / f"{video_name}_idx{i:02d}_decmiss_GT{gt_bc}.jpg"),
                            warped if warped is not None else crop)
    cap.release()

    print(f"\n=== {video_name} (total={total}) ===")
    for k, s in stats.items():
        print(f"  {k:18s} decoded={s['decoded']:3d} ({s['decoded']/total:.0%})  matched={s['matched']:3d} ({s['matched']/total:.0%})")
    if matched_rows:
        print(f"  MATCHED:")
        for r in matched_rows: print(f"    idx{r['idx']:02d}  GT={r['GT']}  decoded={r['decoded']}")
    return {"video": video_name, "total": total, "stats": stats, "matched": matched_rows}


def main():
    t0 = time.time()
    report = {"videos": []}
    for v in VIDEOS:
        report["videos"].append(run_video(v))
    elapsed = time.time() - t0
    report["elapsed_sec"] = elapsed
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] elapsed {elapsed:.1f}s  -> {OUT/'report.json'}")
    print(f"     samples -> {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
