"""Pipeline v8: КРИТИЧНЫЙ ФИКС — поворот crop на 90° CCW перед OCR.

Орг подтвердил (14.05 18:37): "Съемки видео с этой камеры повёрнутой
на 90 градусов против часовой стрелки".

Что меняется vs v7:
    1. Перед OCR — поворот best_crop на 90° CCW (cv2.ROTATE_90_COUNTERCLOCKWISE).
       Бутылки/ценники/текст становятся в нормальной ориентации.
    2. RapidOCR с use_angle_cls=True (на случай если crop ещё под углом).
    3. Bbox координаты остаются в ИСХОДНОЙ системе (как GT) — детект+трекинг
       не меняем, только OCR-stage rotate'ит свой ввод.

Параметры камеры (от орга): 3840x2160, объектив 16/2.8mm, focal 2.8mm.
"""
from __future__ import annotations

import argparse
import gc
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_v7 import (
    ALWAYS_NET, HSV_RANGES, detect_multicolor,
    iou, global_nms, apply_padding, sharpness, Track, Tracker,
)
from pipeline_v5 import parse_ocr, CSV_COLUMNS


def process_video(video_path: Path, fps_sample=3, output_csv=None,
                  filename_in_csv=None, upscale=3.0, min_track_frames=3):
    output_csv = output_csv or Path(f"outputs/pipeline_v8/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== {video_path.name} (v8 + rotation CCW) ===")
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

    # RapidOCR с угловым классификатором
    ocr = RapidOCR()
    rows = []; skipped = 0
    for tr in tqdm(tracks, desc="OCR"):
        # КРИТИЧНО: поворачиваем crop на 90° CCW чтобы текст стал горизонтальным
        crop_rotated = cv2.rotate(tr.best_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        crop_up = cv2.resize(crop_rotated, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try: result, _ = ocr(crop_up, use_angle_cls=True)
        except TypeError:
            # на случай если параметр не поддерживается этой версией
            try: result, _ = ocr(crop_up)
            except Exception: result = None
        except Exception: result = None
        if not result: skipped += 1; continue
        fields = parse_ocr(result, tr.color)
        if not any([fields["product_name"], fields["price_card"],
                    fields["discount_amount"], fields["barcode"]]):
            skipped += 1; continue
        for col in ALWAYS_NET:
            if not fields.get(col): fields[col] = "нет"
        # bbox в ИСХОДНЫХ координатах (не повёрнутых) — для матчинга с GT
        x1, y1, x2, y2 = tr.best_bbox
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(tr.best_ts)
        fields["x_min"] = round(float(x1), 1); fields["y_min"] = round(float(y1), 1)
        fields["x_max"] = round(float(x2), 1); fields["y_max"] = round(float(y2), 1)
        rows.append(fields)
    print(f"  skipped={skipped}")
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
