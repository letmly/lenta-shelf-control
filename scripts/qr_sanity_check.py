"""Sanity check: генерируем синтетический QR на разных уровнях деградации,
проверяем что наш decoder pipeline в принципе работает.

Эксперимент:
    Берём QR с реальным payload Ленты (JSON со штрихкодом + цены).
    Постепенно ухудшаем: уменьшаем размер, добавляем motion blur, шум.
    На каждом уровне пробуем декодировать через все 4 декодера.
    Получаем «карту деградации» — на каком уровне декодер ломается.

Это даёт ОБЪЕКТИВНУЮ цифру: способны ли мы вообще декодировать
сильно повреждённые QR, или у нас баг в pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import qrcode

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
    WECHAT = None; HAS_WECHAT = False

OUT = Path(__file__).resolve().parent.parent / "outputs" / "qr_sanity"
OUT.mkdir(parents=True, exist_ok=True)

# Реалистичный payload Ленты (формат из ТЗ)
PAYLOAD = '{"b":"3552848940002","p1":2401.99,"p2":1631.99,"p3":1631.99,"p4":1631.99,"aP":1631.99,"aC":"AC42"}'

# Генерим QR в высоком разрешении
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=20, border=4)
qr.add_data(PAYLOAD)
qr.make(fit=True)
img_pil = qr.make_image(fill_color="black", back_color="white").convert("RGB")
clean = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
print(f"clean QR: {clean.shape}  payload len={len(PAYLOAD)}")
cv2.imwrite(str(OUT / "00_clean.jpg"), clean)


def decode_all(img) -> dict:
    res = {}
    try:
        r = [d.data.decode("utf-8", errors="ignore") for d in pyzbar_decode(img)]
        if r: res["pyzbar"] = r
    except Exception: pass
    try:
        r = [d.text for d in zxingcpp.read_barcodes(img)]
        if r: res["zxing"] = r
    except Exception: pass
    if HAS_PYBOOF:
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            pb_img = pb.ndarray_to_boof(gray)
            PB_QR.detect(pb_img)
            r = [det.message for det in PB_QR.detections if det.message]
            if r: res["pyboof"] = r
            res["pyboof_finder_count"] = len(PB_QR.detections)
        except Exception as e:
            res["pyboof_error"] = str(e)
    if HAS_WECHAT:
        try:
            decoded, _ = WECHAT.detectAndDecode(img)
            r = [s for s in decoded if s]
            if r: res["wechat"] = r
        except Exception: pass
    return res


def motion_blur_kernel(size: int, angle_deg: float = 0):
    """Ядро мотион-блюра."""
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0 / size
    # rotate
    M = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (size, size))
    k = k / k.sum()
    return k


# ---- эксперимент: разные уровни деградации ----
RESULTS = []
for target_qr_size_px in [200, 100, 80, 60, 50, 40, 30]:
    # отресайзим clean до target_qr_size_px
    sz = target_qr_size_px
    small = cv2.resize(clean, (sz, sz), interpolation=cv2.INTER_AREA)
    for blur_kernel in [0, 3, 5, 7, 9]:
        if blur_kernel == 0:
            blurred = small
        else:
            k = motion_blur_kernel(blur_kernel, angle_deg=15)
            blurred = cv2.filter2D(small, -1, k)
        # добавим небольшой gaussian noise
        noise = np.random.normal(0, 8, blurred.shape).astype(np.int16)
        noisy = np.clip(blurred.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        # для декода — увеличим 4х (это что мы и делали в pipeline)
        big = cv2.resize(noisy, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        res = decode_all(big)
        decoded = any(k in res for k in ["pyzbar", "zxing", "pyboof", "wechat"])
        matched = any("3552848940002" in str(v) for v in res.values() if isinstance(v, list))
        finder_count = res.get("pyboof_finder_count", 0)
        RESULTS.append({"size_px": target_qr_size_px, "blur": blur_kernel,
                        "decoded": decoded, "matched_bc": matched,
                        "finder_count": finder_count, "details": res})
        path = OUT / f"qr_size{sz}_blur{blur_kernel}_dec{int(decoded)}_match{int(matched)}.jpg"
        cv2.imwrite(str(path), big)

# Сводка таблицей
print("\n=== Sanity check matrix (synthetic blurred QR) ===")
print("\n   blur kernel:    0     3     5     7     9")
for sz in [200, 100, 80, 60, 50, 40, 30]:
    row = [f"QR {sz:>3}px:  "]
    for bk in [0, 3, 5, 7, 9]:
        r = next(x for x in RESULTS if x["size_px"] == sz and x["blur"] == bk)
        if r["matched_bc"]:
            row.append("  OK  ")
        elif r["decoded"]:
            row.append("  ?   ")
        elif r["finder_count"] > 0:
            row.append(" fp   ")
        else:
            row.append("  --  ")
    print(" ".join(row))
print("\nOK = декодирован и payload содержит наш barcode")
print("?  = декодирован, но не наш payload")
print("fp = finder pattern найден, но не задекодилось")
print("-- = ничего не найдено")

(OUT / "report.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] artifacts in {OUT}/")
