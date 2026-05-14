"""Pipeline v9: v8 + QR-rescue с rotation+upscale.

Открытие 14.05: QR декодируется с rotation CCW 90° + upscale x4!
На 26_12-20 нашли 2/71 декодов с правильным GT-штрихкодом.
Payload формат URL-encoded: `barcode=XXX&price1=Y&price2=Z&price4=W&aP=V`

Что меняется vs v8:
    1. На best_crop делаем QR-rescue: rotate CCW + upscale x4 + decode
       через pyzbar + zxing-cpp.
    2. Парсим URL-encoded payload → заполняем 11 QR-полей.
    3. Если QR взялся — barcode = primary key для матчинга с GT.
"""
from __future__ import annotations

import argparse
import gc
import sys
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

import zxingcpp

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_v7 import (
    ALWAYS_NET, HSV_RANGES, detect_multicolor,
    iou, global_nms, apply_padding, sharpness, Track, Tracker,
)
from pipeline_v5 import parse_ocr, CSV_COLUMNS

# Маппинг QR-ключей → CSV-поля
QR_FIELD_MAP = {
    "barcode": "qr_code_barcode", "b": "qr_code_barcode",
    "price1": "price1_qr",        "p1": "price1_qr",
    "price2": "price2_qr",        "p2": "price2_qr",
    "price3": "price3_qr",        "p3": "price3_qr",
    "price4": "price4_qr",        "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count", "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price", "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count", "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price", "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr", "aP": "action_price_qr",
    "actionCode":  "action_code_qr",  "aC": "action_code_qr",
}


def parse_qr_payload(payload: str) -> dict:
    """URL-encoded `key=val&key=val` → {csv_field: value}."""
    fields = {}
    # пробуем как URL-encoded
    try:
        parsed = urllib.parse.parse_qs(payload, keep_blank_values=False)
        for k, vals in parsed.items():
            if k in QR_FIELD_MAP and vals:
                fields[QR_FIELD_MAP[k]] = vals[0]
        if fields: return fields
    except Exception: pass
    # fallback: ручной split &
    for token in payload.split("&"):
        if "=" in token:
            k, v = token.split("=", 1)
            if k in QR_FIELD_MAP and v:
                fields[QR_FIELD_MAP[k]] = v
    return fields


def try_decode_qr(crop) -> dict:
    """Пробуем декодировать QR. Возвращает dict с QR-полями (если декодилось)."""
    if crop is None or crop.size == 0: return {}
    # ROTATE CCW + upscale x4
    rotated = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    variants = [
        cv2.resize(rotated, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
        cv2.resize(rotated, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
        cv2.cvtColor(cv2.resize(rotated, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                     cv2.COLOR_BGR2GRAY),
    ]
    for v in variants:
        # pyzbar
        try:
            for d in pyzbar_decode(v):
                payload = d.data.decode("utf-8", errors="ignore")
                if "=" in payload and ("barcode" in payload or "b=" in payload):
                    parsed = parse_qr_payload(payload)
                    if parsed: return parsed
        except Exception: pass
        # zxing
        try:
            for d in zxingcpp.read_barcodes(v):
                if "=" in d.text and ("barcode" in d.text or "b=" in d.text):
                    parsed = parse_qr_payload(d.text)
                    if parsed: return parsed
        except Exception: pass
        # pyboof
        if HAS_PYBOOF:
            try:
                gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if v.ndim == 3 else v
                pb_img = pb.ndarray_to_boof(gray)
                PB_QR.detect(pb_img)
                for d in PB_QR.detections:
                    if d.message and "=" in d.message:
                        parsed = parse_qr_payload(d.message)
                        if parsed: return parsed
            except Exception: pass
    return {}


def process_video(video_path: Path, fps_sample=3, output_csv=None,
                  filename_in_csv=None, upscale=3.0, min_track_frames=3):
    output_csv = output_csv or Path(f"outputs/pipeline_v9/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== {video_path.name} (v9: rotation + QR-rescue) ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / fps_sample))

    tracker = Tracker(iou_th=0.25, max_missed_ms=600)
    fi = 0
    pbar = tqdm(total=n_frames, desc="frames")
    while True:
        ok, frame = cap.read()
        if not ok: break
        pbar.update(1)
        if fi % step == 0:
            ts_ms = fi * (1000.0 / src_fps)
            dets = detect_multicolor(frame)
            tracker.update(dets, ts_ms, frame)
        fi += 1
    pbar.close(); cap.release(); gc.collect()

    tracks = [t for t in tracker.tracks if t.n_frames >= min_track_frames and t.best_crop is not None]
    print(f"  raw={len(tracker.tracks)}  kept={len(tracks)}")

    ocr = RapidOCR()
    rows = []; skipped = 0; qr_decoded = 0

    # Открываем видео ОТДЕЛЬНО — для QR нужен расширенный crop по best_ts
    cap_qr = cv2.VideoCapture(str(video_path))

    for tr in tqdm(tracks, desc="OCR+QR"):
        # 1. OCR на rotated crop (best_crop уже padded +25%)
        crop_rotated = cv2.rotate(tr.best_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        crop_up = cv2.resize(crop_rotated, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try: result, _ = ocr(crop_up, use_angle_cls=True)
        except TypeError:
            try: result, _ = ocr(crop_up)
            except Exception: result = None
        except Exception: result = None

        # 2. QR-rescue с РАСШИРЕННЫМ crop (+75% вокруг bbox, не +25% как у OCR)
        # — QR часто за пределами тесного red-bbox
        qr_fields = {}
        try:
            cap_qr.set(cv2.CAP_PROP_POS_MSEC, float(tr.best_ts))
            ok_qr, fr_qr = cap_qr.read()
            if ok_qr:
                x1, y1, x2, y2 = tr.best_bbox
                bw, bh = x2 - x1, y2 - y1
                H, W = fr_qr.shape[:2]
                # padding +75% во все стороны
                ex1 = max(0, int(x1 - bw * 0.75))
                ey1 = max(0, int(y1 - bh * 0.75))
                ex2 = min(W, int(x2 + bw * 0.75))
                ey2 = min(H, int(y2 + bh * 0.75))
                wide_crop = fr_qr[ey1:ey2, ex1:ex2]
                qr_fields = try_decode_qr(wide_crop)
        except Exception:
            pass
        if qr_fields: qr_decoded += 1

        # parse OCR
        if result:
            fields = parse_ocr(result, tr.color)
        else:
            from pipeline_v5 import CSV_COLUMNS as CC
            fields = {col: "" for col in CC}
            fields["color"] = tr.color

        # FILTER: пустой OCR + пустой QR → выкидываем
        useful = any([fields.get("product_name"), fields.get("price_card"),
                      fields.get("discount_amount"), fields.get("barcode"),
                      qr_fields])
        if not useful: skipped += 1; continue

        # merge QR fields (приоритет над OCR где есть)
        for qf, qv in qr_fields.items():
            fields[qf] = str(qv)
        # если QR содержит barcode — используем его как primary
        if qr_fields.get("qr_code_barcode") and not fields.get("barcode"):
            fields["barcode"] = qr_fields["qr_code_barcode"]

        # дефолт 'нет' для always-нет полей (если QR не положил значение)
        for col in ALWAYS_NET:
            if not fields.get(col): fields[col] = "нет"

        x1, y1, x2, y2 = tr.best_bbox
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(tr.best_ts)
        fields["x_min"] = round(float(x1), 1); fields["y_min"] = round(float(y1), 1)
        fields["x_max"] = round(float(x2), 1); fields["y_max"] = round(float(y2), 1)
        rows.append(fields)
    cap_qr.release()
    print(f"  skipped={skipped}  QR decoded={qr_decoded}/{len(tracks)} ({qr_decoded/max(len(tracks),1):.0%})")
    df_out = pd.DataFrame(rows, columns=CSV_COLUMNS).fillna("")
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"  saved: {output_csv}  ({len(df_out)} rows)")
    return df_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fps", type=float, default=3)
    p.add_argument("--filename-in-csv", type=str, default=None)
    p.add_argument("--upscale", type=float, default=3.0)
    args = p.parse_args()
    process_video(args.video, fps_sample=args.fps, output_csv=args.output,
                  filename_in_csv=args.filename_in_csv, upscale=args.upscale)


if __name__ == "__main__":
    main()
