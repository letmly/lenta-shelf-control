"""Глубокая диагностика QR на нескольких ценниках.

Шаги:
    1. Берём warped из rectify_v2 (где видим весь ценник)
    2. Вырезаем ТОЛЬКО зону QR (правый нижний угол ценника, ~30% от ширины)
    3. Super-zoom 4x с разными interpolation methods
    4. Apply preprocessing pipeline: denoise + unsharp + binarize
    5. Запускаем все декодеры
    6. ОТДЕЛЬНО: пытаемся найти QR finder patterns через PyBoof + OpenCV WeChat
       (даже если не декодируется — узнаем, видна ли структура)

Сохраняем ВСЕ промежуточные изображения для глазного контроля.
"""
from __future__ import annotations

import json
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

# WeChat QR detector
WECHAT = None
try:
    # модели нужно скачать вручную, но cv2.wechat_qrcode без них работает в read-only режиме
    WECHAT = cv2.wechat_qrcode_WeChatQRCode()
    HAS_WECHAT = True
except Exception as e:
    HAS_WECHAT = False

ROOT = Path(__file__).resolve().parent.parent
RECTIFY_DIR = ROOT / "outputs" / "rectify_v2" / "26_12-20"
OUT = ROOT / "outputs" / "qr_deep_dive"
OUT.mkdir(parents=True, exist_ok=True)

# Берём первые 10 ценников где А_adaptive нашёл quad
SAMPLES = []
for f in sorted(RECTIFY_DIR.glob("idx*_A_adaptive_warped.jpg"))[:10]:
    # парсим имя
    parts = f.stem.split("_")
    idx = int(parts[0].replace("idx", ""))
    gt = parts[1].replace("GT", "")
    SAMPLES.append((idx, gt, f))


def unsharp_mask(img, sigma=1.0, amount=1.5):
    """Unsharp mask — увеличивает резкость."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def extract_qr_region(warped):
    """Вырезаем нижний правый квадрант ценника — там обычно QR."""
    h, w = warped.shape[:2]
    # QR на ценнике Ленты — нижне-правый угол белой части
    # На ценнике: красная половина слева, белая справа. QR в нижней правой части белой
    # Возьмём правую половину + нижние 60%
    y1 = int(h * 0.30); y2 = h
    x1 = int(w * 0.50); x2 = w
    return warped[y1:y2, x1:x2]


def decode_all(img) -> dict:
    """Прогон через все декодеры. Возвращает {decoder_name: list of decoded strings}."""
    res = {"pyzbar": [], "zxing": [], "pyboof": [], "wechat": [],
           "pyboof_finder_patterns": 0}
    try:
        for d in pyzbar_decode(img):
            res["pyzbar"].append(d.data.decode("utf-8", errors="ignore"))
    except Exception: pass
    try:
        for r in zxingcpp.read_barcodes(img):
            res["zxing"].append(r.text)
    except Exception: pass
    if HAS_PYBOOF:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            pb_img = pb.ndarray_to_boof(gray)
            PB_QR.detect(pb_img)
            for det in PB_QR.detections:
                # finder pattern detection даже без полного декода
                res["pyboof_finder_patterns"] += 1
                if det.message: res["pyboof"].append(det.message)
        except Exception as e:
            res["pyboof_error"] = str(e)
    if HAS_WECHAT and WECHAT is not None:
        try:
            decoded, points = WECHAT.detectAndDecode(img)
            for s in decoded:
                if s: res["wechat"].append(s)
        except Exception: pass
    return res


def process_one(idx, gt, warped_path):
    out_dir = OUT / f"idx{idx:02d}_GT{gt}"
    out_dir.mkdir(exist_ok=True)
    warped = cv2.imread(str(warped_path))
    if warped is None: return None

    cv2.imwrite(str(out_dir / "01_warped.jpg"), warped)

    # ---- crop QR region ----
    qr_region = extract_qr_region(warped)
    cv2.imwrite(str(out_dir / "02_qr_region_raw.jpg"), qr_region)

    # ---- super-zoom 4x с разными interpolations ----
    interp_methods = {
        "lanczos": cv2.INTER_LANCZOS4,
        "cubic":   cv2.INTER_CUBIC,
        "linear":  cv2.INTER_LINEAR,
    }
    big_versions = {}
    for name, interp in interp_methods.items():
        big = cv2.resize(qr_region, None, fx=4, fy=4, interpolation=interp)
        big_versions[name] = big
        cv2.imwrite(str(out_dir / f"03_zoom4x_{name}.jpg"), big)

    # ---- preprocessing chain ----
    gray = cv2.cvtColor(big_versions["lanczos"], cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_dir / "04_gray.jpg"), gray)
    # denoise
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    cv2.imwrite(str(out_dir / "05_denoised.jpg"), denoised)
    # unsharp
    sharp = unsharp_mask(denoised, sigma=1.5, amount=2.0)
    cv2.imwrite(str(out_dir / "06_sharpened.jpg"), sharp)
    # otsu binarize
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(out_dir / "07_otsu_binary.jpg"), otsu)
    # adaptive binarize
    adap = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 5)
    cv2.imwrite(str(out_dir / "08_adaptive_binary.jpg"), adap)

    # ---- decode на КАЖДОМ из preprocessing вариантов ----
    test_imgs = {
        "raw_zoom4x_lanczos": big_versions["lanczos"],
        "raw_zoom4x_cubic":   big_versions["cubic"],
        "gray":               gray,
        "denoised":           denoised,
        "sharpened":          sharp,
        "otsu":               otsu,
        "adaptive":           adap,
    }
    results = {}
    finder_count = 0
    matched_any = False
    decoded_any = False
    for name, img in test_imgs.items():
        d = decode_all(img)
        results[name] = d
        finder_count = max(finder_count, d.get("pyboof_finder_patterns", 0))
        for k in ("pyzbar", "zxing", "pyboof", "wechat"):
            if d.get(k):
                decoded_any = True
                if gt in d[k]:
                    matched_any = True
    return {
        "idx": idx, "gt": gt,
        "finder_patterns_detected": finder_count,
        "decoded_any": decoded_any,
        "matched_gt": matched_any,
        "by_variant": results,
    }


report = []
for idx, gt, path in SAMPLES:
    print(f"processing idx{idx:02d} GT={gt}...")
    r = process_one(idx, gt, path)
    if r is not None:
        report.append(r)

# сводка
print(f"\n=== Сводка по {len(report)} ценникам ===")
total = len(report)
fp_detected = sum(1 for r in report if r["finder_patterns_detected"] > 0)
decoded = sum(1 for r in report if r["decoded_any"])
matched = sum(1 for r in report if r["matched_gt"])
print(f"  finder patterns обнаружены хотя бы на одном preprocessing: {fp_detected}/{total}")
print(f"  что-то декодировано:                                       {decoded}/{total}")
print(f"  декодировано именно GT штрихкод:                           {matched}/{total}")

(OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] full report -> {OUT/'report.json'}")
print(f"     artifacts per sample -> {OUT}/idx*_GT*/")
