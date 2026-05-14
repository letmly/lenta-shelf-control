"""Pipeline v7: padding +25% во все стороны (не extension +60% вниз) + дефолт 'нет'.

Изменения vs v5:
    1. Детектор: bbox без extension вниз, padding +25% во ВСЕ стороны.
       (В v5 extension вниз делал кроп с белой пустотой → OCR тупил.)
    2. Дефолт 'нет' для 7 always-нет полей.
    3. Парсер: используем v5 версию (с улучшенным product_name).
    4. Upscale x3 (как мы видели работает).
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
from pipeline_v5 import parse_ocr, CSV_COLUMNS

# 7 полей которые ВСЕГДА 'нет' для красных АПЦ ценников (из анализа GT 26_12-20)
ALWAYS_NET = [
    "price_discount",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

HSV_RANGES = {
    "red":    [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))],
    "yellow": [((18, 100, 100), (35, 255, 255))],
    "green":  [((40, 100, 60), (85, 255, 255))],
}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def detect_color(frame, hsv, ranges, color_name, min_area):
    """Детект bbox для одного цвета. БЕЗ extension вниз!
    Только сам цветовой блок. Padding делаем позже на кропе."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges: mask |= cv2.inRange(hsv, lo, hi)
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area: continue
        ar = w / h if h > 0 else 0
        if not (0.25 < ar < 4): continue
        boxes.append(([x, y, x + w, y + h], color_name))
    return boxes


def detect_multicolor(frame, min_area=4000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    all_boxes = []
    for color, ranges in HSV_RANGES.items():
        ma = min_area * 2 if color in ("green", "yellow") else min_area
        all_boxes.extend(detect_color(frame, hsv, ranges, color, ma))
    return global_nms(all_boxes, iou_th=0.3)


def global_nms(boxes, iou_th=0.3):
    if not boxes: return []
    boxes_sorted = sorted(boxes, key=lambda x: -((x[0][2]-x[0][0])*(x[0][3]-x[0][1])))
    keep = []
    for bc in boxes_sorted:
        if all(iou(bc[0], kc[0]) <= iou_th for kc in keep):
            keep.append(bc)
    return keep


def apply_padding(bbox, shape, pct=0.25):
    """+25% padding во все стороны от исходного bbox."""
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct), int(bh * pct)
    return [max(0, x1 - px), max(0, y1 - py),
            min(w, x2 + px), min(h, y2 + py)]


def sharpness(img):
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


@dataclass
class Track:
    track_id: int
    bbox: list
    color: str
    first_ts: float
    last_ts: float
    n_frames: int = 0
    best_ts: float = 0.0
    best_bbox: list = field(default_factory=list)  # bbox внутри best_crop_frame
    best_crop: np.ndarray = None
    best_sharpness: float = -1.0


class Tracker:
    def __init__(self, iou_th=0.25, max_missed_ms=600):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.iou_th = iou_th
        self.max_missed_ms = max_missed_ms

    def update(self, dets, ts_ms, frame):
        for t in self.tracks:
            if (ts_ms - t.last_ts) > self.max_missed_ms:
                continue
        used = set()
        for bbox, color in dets:
            best_t = None; best_iou = 0
            for t in self.tracks:
                if t.track_id in used: continue
                if (ts_ms - t.last_ts) > self.max_missed_ms: continue
                v = iou(bbox, t.bbox)
                if v > best_iou: best_iou = v; best_t = t
            if best_t and best_iou >= self.iou_th:
                best_t.bbox = bbox; best_t.last_ts = ts_ms; best_t.n_frames += 1
                self._maybe_update(best_t, ts_ms, bbox, frame); used.add(best_t.track_id)
            else:
                t = Track(self.next_id, bbox, color, ts_ms, ts_ms, n_frames=1)
                self._maybe_update(t, ts_ms, bbox, frame)
                self.tracks.append(t); self.next_id += 1

    def _maybe_update(self, track, ts_ms, bbox, frame):
        """Сохраняем PADDED crop (детекторный bbox + 25% padding)."""
        padded = apply_padding(bbox, frame.shape, pct=0.25)
        x1, y1, x2, y2 = padded
        if x2 <= x1 or y2 <= y1: return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30: return
        sh = sharpness(crop)
        if sh > track.best_sharpness:
            track.best_sharpness = sh; track.best_ts = ts_ms
            track.best_bbox = list(bbox)  # храним исходный bbox без padding (для CSV)
            track.best_crop = crop.copy()


def process_video(video_path: Path, fps_sample=3, output_csv=None,
                  filename_in_csv=None, upscale=3.0, min_track_frames=3):
    output_csv = output_csv or Path(f"outputs/pipeline_v7/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== {video_path.name} ===")
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
    rows = []; skipped = 0
    for tr in tqdm(tracks, desc="OCR"):
        crop_up = cv2.resize(tr.best_crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try: result, _ = ocr(crop_up)
        except Exception: result = None
        if not result: skipped += 1; continue
        fields = parse_ocr(result, tr.color)
        if not any([fields["product_name"], fields["price_card"],
                    fields["discount_amount"], fields["barcode"]]):
            skipped += 1; continue
        # ДЕФОЛТ 'нет' для always-нет полей
        for col in ALWAYS_NET:
            if not fields.get(col): fields[col] = "нет"
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
