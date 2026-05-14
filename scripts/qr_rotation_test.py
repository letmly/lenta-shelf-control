"""Финальная попытка QR-декода с учётом ротации видео.

Раньше мы пытались декодировать QR на сырых кропах. После того как
орг подтвердил поворот видео CCW 90°, попробуем заново:
    1. Берём GT-bbox (идеальный детектор)
    2. Padding +25%
    3. Rotate CCW 90° (откатываем ориентацию)
    4. Upscale x4
    5. Прогоняем 4 декодера: pyzbar, zxing-cpp, PyBoof, WeChat
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode

import zxingcpp

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

try:
    WECHAT = cv2.wechat_qrcode_WeChatQRCode()
    HAS_WECHAT = True
except Exception:
    HAS_WECHAT = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "qr_rotation_test"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def decode_all(img):
    res = set()
    for variant in [img,
                    cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img]:
        try:
            for d in pyzbar_decode(variant): res.add(d.data.decode("utf-8", errors="ignore"))
        except: pass
        try:
            for r in zxingcpp.read_barcodes(variant): res.add(r.text)
        except: pass
        if HAS_PYBOOF:
            try:
                gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if variant.ndim == 3 else variant
                pb_img = pb.ndarray_to_boof(gray)
                PB_QR.detect(pb_img)
                for d in PB_QR.detections:
                    if d.message: res.add(d.message)
            except: pass
        if HAS_WECHAT:
            try:
                decoded, _ = WECHAT.detectAndDecode(variant)
                for s in decoded:
                    if s: res.add(s)
            except: pass
    return res


# Прогон по всем 26_12-20 с rotation
csv_path = ROOT / "data/labeled/26_12-20/26_12-20.csv"
mp4_path = ROOT / "data/labeled/26_12-20/26_12-20.mp4"
df = pd.read_csv(csv_path, encoding="utf-8")
for c in ("x_min","y_min","x_max","y_max","frame_timestamp"):
    df[c] = df[c].apply(parse_ru)
df = df.dropna(subset=["x_min","y_min","x_max","y_max","frame_timestamp"]).reset_index(drop=True)
df["barcode"] = df["barcode"].astype(str).str.strip().str.replace(".0","",regex=False)

cap = cv2.VideoCapture(str(mp4_path))
total = 0; decoded_any = 0; matched_gt = 0; decoded_examples = []

print(f"Testing rotated QR decode on {len(df)} GT ценников...")
print(f"Decoders: pyzbar=YES  zxing=YES  pyboof={'YES' if HAS_PYBOOF else 'NO'}  wechat={'YES' if HAS_WECHAT else 'NO'}")
print()

for i, r in df.iterrows():
    ts = r["frame_timestamp"]; gt_bc = r["barcode"]
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
    ok, frame = cap.read()
    if not ok: continue
    gx1, gy1, gx2, gy2 = int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"])
    # padding +25%
    bw, bh = gx2-gx1, gy2-gy1
    H, W = frame.shape[:2]
    x1 = max(0, gx1 - int(bw*0.25)); y1 = max(0, gy1 - int(bh*0.25))
    x2 = min(W, gx2 + int(bw*0.25)); y2 = min(H, gy2 + int(bh*0.25))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: continue
    total += 1
    # ROTATE CCW 90° — критичный фикс
    crop_rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # upscale x4
    crop_big = cv2.resize(crop_rot, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    codes = decode_all(crop_big)
    if codes:
        decoded_any += 1
        if gt_bc in codes:
            matched_gt += 1
            print(f"  ✓ MATCH idx={i} bc={gt_bc}  decoded={[c[:30] for c in codes]}")
        elif len(decoded_examples) < 3:
            decoded_examples.append((i, gt_bc, list(codes)))

cap.release()

print()
print(f"=== RESULT ===")
print(f"Total tried:       {total}")
print(f"Decoded any QR:    {decoded_any}/{total}  ({decoded_any/total:.0%})")
print(f"Matched GT-bc:     {matched_gt}/{total}  ({matched_gt/total:.0%})")
if decoded_examples:
    print(f"Examples decoded-but-not-matched:")
    for i, gt, codes in decoded_examples:
        print(f"  idx={i}  GT={gt}  decoded={[c[:60] for c in codes]}")
