"""Применяем Real-ESRGAN на super-QR кадрах из multi-frame fusion.

Гипотеза: после ESRGAN x4 (240-->960 пикселей) декодеры смогут взять QR.
Берём первые N super-QR из fusion samples, прогоняем через ESRGAN,
пробуем декодировать.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pyzbar.pyzbar import decode as pyzbar_decode

import zxingcpp

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

# Real-ESRGAN
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_IN = ROOT / "outputs" / "qr_multiframe_fusion" / "samples"
OUT = ROOT / "outputs" / "esrgan_on_qr"
OUT.mkdir(parents=True, exist_ok=True)

# Качаем веса ESRGAN
print("Loading Real-ESRGAN x4 model...")
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
# Веса автоматически скачаются при первом запуске
WEIGHTS_PATH = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# Скачиваем веса вручную для надёжности
import urllib.request
weights_dir = ROOT / "_third_party" / "esrgan_weights"
weights_dir.mkdir(parents=True, exist_ok=True)
weights_file = weights_dir / "RealESRGAN_x4plus.pth"
if not weights_file.exists():
    print(f"Downloading weights to {weights_file}...")
    urllib.request.urlretrieve(WEIGHTS_PATH, str(weights_file))
    print("Downloaded.")

upsampler = RealESRGANer(
    scale=4,
    model_path=str(weights_file),
    model=model,
    tile=256,
    tile_pad=10,
    pre_pad=0,
    half=False,  # CPU не поддерживает FP16
    device=torch.device("cpu"),
)
print("ESRGAN ready.")


def decode_all(img):
    res = set()
    for v in [img,
              cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
              cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img]:
        try:
            for d in pyzbar_decode(v): res.add(d.data.decode("utf-8", errors="ignore"))
        except Exception: pass
        try:
            for r in zxingcpp.read_barcodes(v): res.add(r.text)
        except Exception: pass
        if HAS_PYBOOF:
            try:
                gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if v.ndim == 3 else v
                pb_img = pb.ndarray_to_boof(gray)
                PB_QR.detect(pb_img)
                for det in PB_QR.detections:
                    if det.message: res.add(det.message)
            except Exception: pass
    return res


# Берём первые 10 super-QR
samples = sorted(SAMPLES_IN.glob("*_n*_*_GT*.jpg"))[:10]
print(f"\nProcessing {len(samples)} super-QR samples through ESRGAN x4...")
print()

results = []
for s in samples:
    # парсим GT-bc из имени
    gt = s.stem.split("GT")[1]
    img = cv2.imread(str(s))
    if img is None: continue
    h, w = img.shape[:2]
    print(f"{s.name}: input {w}x{h}", end=" --> ")
    try:
        sr_img, _ = upsampler.enhance(img, outscale=4)
        sr_h, sr_w = sr_img.shape[:2]
        print(f"ESRGAN {sr_w}x{sr_h}", end=" ")
        # декодируем оригинал и SR
        codes_orig = decode_all(img)
        codes_sr = decode_all(sr_img)
        codes_orig_clean = [c for c in codes_orig if not isinstance(c, set)]
        codes_sr_clean = [c for c in codes_sr if not isinstance(c, set)]
        matched_orig = gt in codes_orig
        matched_sr = gt in codes_sr
        tag = "MATCH-SR!" if matched_sr else ("MATCH-orig" if matched_orig else
              ("dec-SR" if codes_sr else ("dec-orig" if codes_orig else "miss")))
        print(f"  decoded_orig={list(codes_orig)[:3]}  decoded_sr={list(codes_sr)[:3]}  [{tag}]")
        # сохраняем
        cv2.imwrite(str(OUT / f"{s.stem}_SR4x.jpg"), sr_img)
        results.append({
            "file": s.name, "gt": gt,
            "decoded_orig": list(codes_orig), "decoded_sr": list(codes_sr),
            "matched_orig": matched_orig, "matched_sr": matched_sr,
        })
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n=== Summary ===")
total = len(results)
matched_orig = sum(1 for r in results if r["matched_orig"])
matched_sr = sum(1 for r in results if r["matched_sr"])
dec_sr = sum(1 for r in results if r["decoded_sr"])
print(f"  Total processed:          {total}")
print(f"  Matched on original:      {matched_orig}/{total}")
print(f"  Matched on SR (x4):       {matched_sr}/{total}")
print(f"  Anything decoded on SR:   {dec_sr}/{total}")

import json
(OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] {OUT/'report.json'}")
print(f"     SR images: {OUT}/")
