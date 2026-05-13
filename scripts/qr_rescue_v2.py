"""QR/штрихкод rescue v2: добавляем zxing-cpp + WeChat QR + time-window search.

Стратегии:
    A. pyzbar     — baseline (что было)
    B. zxing-cpp  — более мощный декодер
    C. WeChat QR  — нейросетевой детектор QR (из opencv-contrib)
    D. time-window search ±300 ms — берём sharpest кадр в окрестности GT-timestamp

Каждая стратегия применяется к pad+50% crop.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from tqdm import tqdm

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

# WeChat QR — требует opencv-contrib + 4 файла моделей
HAS_WECHAT = False
WECHAT_DET = None
try:
    # модели лежат в opencv-contrib пакете
    import os
    cv2_dir = Path(cv2.__file__).parent
    models_dir = cv2_dir / "data"
    # Попробуем дефолтные пути; если нет — отключим
    detect_proto = models_dir / "detect.prototxt"
    detect_model = models_dir / "detect.caffemodel"
    sr_proto = models_dir / "sr.prototxt"
    sr_model = models_dir / "sr.caffemodel"
    # cv2.wechat_qrcode_WeChatQRCode требует все четыре
    if all(p.exists() for p in (detect_proto, detect_model, sr_proto, sr_model)):
        WECHAT_DET = cv2.wechat_qrcode_WeChatQRCode(
            str(detect_proto), str(detect_model), str(sr_proto), str(sr_model)
        )
        HAS_WECHAT = True
except Exception as e:
    HAS_WECHAT = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "qr_rescue_v2"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    (ROOT / "data/labeled/26_12-20/26_12-20.mp4", ROOT / "data/labeled/26_12-20/26_12-20.csv"),
    (ROOT / "data/labeled/43_15/43_15.mp4",        ROOT / "data/labeled/43_15/43_15.csv"),
]


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def load_gt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype(str).str.strip().str.replace(".0", "", regex=False)
    return df


def pad(b, shape, pct=0.5):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def sharpness(img) -> float:
    """Laplacian variance — чем выше, тем чётче."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def decode_pyzbar(crop) -> list[str]:
    return [d.data.decode("utf-8", errors="ignore").strip() for d in pyzbar_decode(crop)]


def decode_zxing(crop) -> list[str]:
    if not HAS_ZXING: return []
    try:
        res = zxingcpp.read_barcodes(crop)
        return [r.text.strip() for r in res]
    except Exception:
        return []


def decode_wechat(crop) -> list[str]:
    if not HAS_WECHAT: return []
    try:
        decoded, _ = WECHAT_DET.detectAndDecode(crop)
        return [s.strip() for s in decoded if s]
    except Exception:
        return []


def grab_frame(cap, ts_ms):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
    ok, frame = cap.read()
    return frame if ok else None


def best_frame_in_window(cap, ts_center_ms, bbox, half_window_ms=300, step_ms=50):
    """Ищем самый чёткий кадр в [ts_center - half_window, ts_center + half_window]."""
    best = (None, -1.0, ts_center_ms)
    for dt in range(-half_window_ms, half_window_ms + 1, step_ms):
        ts = ts_center_ms + dt
        f = grab_frame(cap, ts)
        if f is None: continue
        # измеряем sharpness в bbox + небольшая зона вокруг
        x1, y1, x2, y2 = pad(bbox, f.shape, 0.5)
        roi = f[y1:y2, x1:x2]
        if roi.size == 0: continue
        s = sharpness(roi)
        if s > best[1]:
            best = (f, s, ts)
    return best


def run_video(video: Path, csv: Path) -> dict:
    df = load_gt(csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): return {"video": video.name, "error": "cannot open"}
    total = len(df)

    stats = {k: {"decoded": 0, "matched": 0, "samples": []} for k in
             ["A_pyzbar", "B_zxing", "C_wechat", "D_window_pyzbar", "D_window_zxing", "ANY"]}

    for _, r in tqdm(df.iterrows(), total=total, desc=video.name):
        ts_gt = r["frame_timestamp"]
        gt_bc = r["barcode"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))

        # обычный кадр на ts_gt
        frame = grab_frame(cap, ts_gt)
        if frame is None: continue
        x1, y1, x2, y2 = pad(bbox, frame.shape, 0.5)
        crop = frame[y1:y2, x1:x2]

        # ---- A: pyzbar ----
        a = decode_pyzbar(crop)
        if a:
            stats["A_pyzbar"]["decoded"] += 1
            if gt_bc in a:
                stats["A_pyzbar"]["matched"] += 1

        # ---- B: zxing ----
        b = decode_zxing(crop)
        if b:
            stats["B_zxing"]["decoded"] += 1
            if gt_bc in b:
                stats["B_zxing"]["matched"] += 1

        # ---- C: wechat ----
        c = decode_wechat(crop)
        if c:
            stats["C_wechat"]["decoded"] += 1
            if gt_bc in c:
                stats["C_wechat"]["matched"] += 1

        # ---- D: best frame in time window ----
        best_f, _sh, _ts = best_frame_in_window(cap, ts_gt, bbox)
        if best_f is not None:
            xx1, yy1, xx2, yy2 = pad(bbox, best_f.shape, 0.5)
            crop_w = best_f[yy1:yy2, xx1:xx2]
            dwp = decode_pyzbar(crop_w)
            if dwp:
                stats["D_window_pyzbar"]["decoded"] += 1
                if gt_bc in dwp: stats["D_window_pyzbar"]["matched"] += 1
            dwz = decode_zxing(crop_w)
            if dwz:
                stats["D_window_zxing"]["decoded"] += 1
                if gt_bc in dwz: stats["D_window_zxing"]["matched"] += 1

        # ---- ANY (union всех стратегий по match) ----
        any_decoded = bool(a or b or c or (dwp if best_f is not None else []) or (dwz if best_f is not None else []))
        any_matched = (gt_bc in a or gt_bc in b or gt_bc in c
                       or (best_f is not None and (gt_bc in dwp or gt_bc in dwz)))
        if any_decoded: stats["ANY"]["decoded"] += 1
        if any_matched: stats["ANY"]["matched"] += 1

    cap.release()

    out = {}
    for k, v in stats.items():
        out[k] = {
            "decoded": v["decoded"],
            "decoded_rate": v["decoded"] / total if total else 0,
            "matched": v["matched"],
            "matched_rate": v["matched"] / total if total else 0,
        }
    return {"video": video.name, "total": total, "stats": out}


def main():
    print(f"Available decoders: pyzbar=YES  zxing-cpp={'YES' if HAS_ZXING else 'NO'}  WeChat={'YES' if HAS_WECHAT else 'NO'}")
    report = {"started_at": time.time(), "decoders": {"pyzbar": True, "zxing": HAS_ZXING, "wechat": HAS_WECHAT}, "videos": []}
    for video, csv in VIDEOS:
        if not video.exists(): continue
        r = run_video(video, csv)
        report["videos"].append(r)
        print(f"\n=== {r['video']} (GT={r['total']}) ===")
        for k, st in r["stats"].items():
            print(f"  {k:18s} decoded={st['decoded']:3d} ({st['decoded_rate']:.0%})  matched-gt={st['matched']:3d} ({st['matched_rate']:.0%})")
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT/'report.json'}")


if __name__ == "__main__":
    main()
