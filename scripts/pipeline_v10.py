"""Pipeline v10: YOLO детектор на rotated frame.

Замена red-mask HSV → openfoodfacts/price-tag-detection YOLOv11x на rotated frame.
Test показал: matched 9/9 (100%) на 26_12-20 vs ~60% у red-mask.

⚠️ AGPL-3.0 — только для прототипа хакатона.
В проде Ленты заменяется на YOLOX-Nano (Apache 2.0) + fine-tune на pseudo-labels.

Архитектура:
    1. video.read() → frame (original orientation, ценники на боку)
    2. rotated = rotate(frame, CCW 90°)  ← теперь ценники нормально
    3. YOLO детектит на rotated → bbox в rotated coords
    4. Tracker работает в rotated coords
    5. best_crop из rotated frame
    6. OCR на best_crop (уже правильно ориентирован)
    7. QR-rescue на wide-crop из rotated frame
    8. Перед записью CSV: bbox координаты конвертируем обратно в original
       (для матчинга с GT который в original coords)
"""
from __future__ import annotations

import argparse
import gc
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm
from ultralytics import YOLO

import zxingcpp

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_v5 import parse_ocr, CSV_COLUMNS
from pipeline_v9 import QR_FIELD_MAP, parse_qr_payload, try_decode_qr
from pipeline_v7 import ALWAYS_NET

YOLO_WEIGHTS = ROOT / "_third_party/yolo_weights/weights/best.pt"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def nms(boxes, iou_th=0.5):
    """Простой NMS оставляющий самые большие."""
    if not boxes: return []
    sorted_b = sorted(boxes, key=lambda b: -((b[2]-b[0])*(b[3]-b[1])))
    keep = []
    for b in sorted_b:
        if all(iou(b, k) <= iou_th for k in keep):
            keep.append(b)
    return keep


def rotate_bbox_back_to_original(bbox, H_orig, W_orig):
    """rotated CCW 90° (на frame с shape (W_orig, H_orig)) → original (H_orig, W_orig).
    Inverse rotation: CW 90°. Mapping: (x,y)_rot → (W_orig - y, x)_orig
    bbox(x1,y1,x2,y2)_rot → corners (W_orig-y1, x1) and (W_orig-y2, x2)
    """
    x1, y1, x2, y2 = bbox
    ox1, oy1 = W_orig - y2, x1
    ox2, oy2 = W_orig - y1, x2
    return (ox1, oy1, ox2, oy2)


def detect_yolo(model, rotated_frame, conf=0.1, iou_th=0.5):
    """Запускаем YOLO на rotated frame, возвращаем bbox + 'red' как color."""
    results = model.predict(rotated_frame, conf=conf, iou=iou_th, imgsz=1920,
                             verbose=False, max_det=200)
    boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes.xyxy.cpu().numpy():
            boxes.append(list(box))
    return nms(boxes, iou_th=0.5)


def sharpness(img):
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def classify_color_hsv(crop):
    """Определяет цвет ценника по HSV-гистограмме."""
    if crop.size == 0: return "red"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # центральная зона
    h, w = hsv.shape[:2]
    center = hsv[h//4:h*3//4, w//4:w*3//4]
    if center.size == 0: return "red"
    masks = {
        "red":    cv2.inRange(center, (0, 80, 60), (10, 255, 255)) |
                  cv2.inRange(center, (170, 80, 60), (180, 255, 255)),
        "yellow": cv2.inRange(center, (18, 100, 100), (35, 255, 255)),
        "green":  cv2.inRange(center, (40, 100, 60), (85, 255, 255)),
    }
    counts = {k: int(np.count_nonzero(m)) for k, m in masks.items()}
    total = center.shape[0] * center.shape[1]
    # если ни один цвет не доминирует — считаем white
    best = max(counts.items(), key=lambda x: x[1])
    if best[1] / total < 0.15: return "white"
    return best[0]


@dataclass
class Track:
    track_id: int
    bbox: list  # в rotated coords
    color: str
    first_ts: float
    last_ts: float
    n_frames: int = 0
    best_ts: float = 0.0
    best_bbox_rotated: list = field(default_factory=list)
    best_crop: np.ndarray = None
    best_sharpness: float = -1.0


class Tracker:
    def __init__(self, iou_th=0.3, max_missed_ms=600):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.iou_th = iou_th
        self.max_missed_ms = max_missed_ms

    def update(self, dets, ts_ms, rotated_frame):
        used = set()
        for bbox in dets:
            best_t = None; best_iou = 0
            for t in self.tracks:
                if t.track_id in used: continue
                if (ts_ms - t.last_ts) > self.max_missed_ms: continue
                v = iou(bbox, t.bbox)
                if v > best_iou: best_iou = v; best_t = t
            if best_t and best_iou >= self.iou_th:
                best_t.bbox = bbox; best_t.last_ts = ts_ms; best_t.n_frames += 1
                self._maybe_update(best_t, ts_ms, bbox, rotated_frame); used.add(best_t.track_id)
            else:
                # crop для классификации цвета
                x1, y1, x2, y2 = map(int, bbox)
                ccrop = rotated_frame[y1:y2, x1:x2]
                color = classify_color_hsv(ccrop) if ccrop.size else "red"
                t = Track(self.next_id, bbox, color, ts_ms, ts_ms, n_frames=1)
                self._maybe_update(t, ts_ms, bbox, rotated_frame)
                self.tracks.append(t); self.next_id += 1

    def _maybe_update(self, track, ts_ms, bbox, frame):
        # padding +25% в rotated coords
        x1, y1, x2, y2 = bbox
        bw, bh = x2-x1, y2-y1
        H, W = frame.shape[:2]
        px1 = max(0, int(x1 - bw * 0.25))
        py1 = max(0, int(y1 - bh * 0.25))
        px2 = min(W, int(x2 + bw * 0.25))
        py2 = min(H, int(y2 + bh * 0.25))
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30: return
        sh = sharpness(crop)
        if sh > track.best_sharpness:
            track.best_sharpness = sh; track.best_ts = ts_ms
            track.best_bbox_rotated = list(bbox)
            track.best_crop = crop.copy()


def process_video(video_path: Path, fps_sample=3, output_csv=None,
                  filename_in_csv=None, upscale=3.0, min_track_frames=3,
                  yolo_conf=0.1):
    output_csv = output_csv or Path(f"outputs/pipeline_v10/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== {video_path.name} (v10: YOLO on rotated) ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / fps_sample))
    H_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    print(f"  loading YOLO...")
    model = YOLO(str(YOLO_WEIGHTS))

    tracker = Tracker(iou_th=0.3, max_missed_ms=600)
    fi = 0
    pbar = tqdm(total=n_frames, desc="frames")
    while True:
        ok, frame = cap.read()
        if not ok: break
        pbar.update(1)
        if fi % step == 0:
            ts_ms = fi * (1000.0 / src_fps)
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            dets = detect_yolo(model, rotated, conf=yolo_conf)
            tracker.update(dets, ts_ms, rotated)
        fi += 1
    pbar.close(); cap.release(); gc.collect()

    tracks = [t for t in tracker.tracks if t.n_frames >= min_track_frames and t.best_crop is not None]
    print(f"  raw={len(tracker.tracks)}  kept={len(tracks)}")

    # Видео открываем заново для wide-crop QR-rescue (нам нужны исходные кадры)
    cap_qr = cv2.VideoCapture(str(video_path))

    ocr = RapidOCR()
    rows = []; skipped = 0; qr_decoded = 0
    for tr in tqdm(tracks, desc="OCR+QR"):
        # OCR на best_crop (уже в rotated coords, ориентация правильная)
        crop_up = cv2.resize(tr.best_crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try: result, _ = ocr(crop_up, use_angle_cls=True)
        except TypeError:
            try: result, _ = ocr(crop_up)
            except Exception: result = None
        except Exception: result = None

        # QR-rescue: широкий crop в rotated coords
        qr_fields = {}
        try:
            cap_qr.set(cv2.CAP_PROP_POS_MSEC, float(tr.best_ts))
            ok_qr, fr_qr = cap_qr.read()
            if ok_qr:
                fr_rot = cv2.rotate(fr_qr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                x1, y1, x2, y2 = tr.best_bbox_rotated
                bw, bh = x2-x1, y2-y1
                H_r, W_r = fr_rot.shape[:2]
                ex1 = max(0, int(x1 - bw * 0.75))
                ey1 = max(0, int(y1 - bh * 0.75))
                ex2 = min(W_r, int(x2 + bw * 0.75))
                ey2 = min(H_r, int(y2 + bh * 0.75))
                wide = fr_rot[ey1:ey2, ex1:ex2]
                # QR в rotated frame не требует доп rotation (уже правильно)
                # Используем try_decode_qr, но без двойного rotate. Делаем напрямую.
                from pipeline_v9 import parse_qr_payload as _pq
                variants = [
                    cv2.resize(wide, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                    cv2.resize(wide, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                ]
                found_payload = None
                for v in variants:
                    if found_payload: break
                    try:
                        for d in pyzbar_decode(v):
                            payload = d.data.decode("utf-8", errors="ignore")
                            if "=" in payload and ("barcode" in payload or "b=" in payload):
                                qr_fields = _pq(payload)
                                if qr_fields: found_payload = payload; break
                    except Exception: pass
                    if found_payload: break
                    try:
                        for d in zxingcpp.read_barcodes(v):
                            if "=" in d.text and ("barcode" in d.text or "b=" in d.text):
                                qr_fields = _pq(d.text)
                                if qr_fields: found_payload = d.text; break
                    except Exception: pass
        except Exception:
            pass
        if qr_fields: qr_decoded += 1

        # parse OCR
        if result:
            fields = parse_ocr(result, tr.color)
        else:
            fields = {col: "" for col in CSV_COLUMNS}
            fields["color"] = tr.color

        # filter useless
        useful = any([fields.get("product_name"), fields.get("price_card"),
                      fields.get("discount_amount"), fields.get("barcode"),
                      qr_fields])
        if not useful: skipped += 1; continue

        # merge QR
        for qf, qv in qr_fields.items():
            fields[qf] = str(qv)
        if qr_fields.get("qr_code_barcode") and not fields.get("barcode"):
            fields["barcode"] = qr_fields["qr_code_barcode"]

        # default "нет"
        for col in ALWAYS_NET:
            if not fields.get(col): fields[col] = "нет"

        # bbox: rotated → original coordinates для CSV (GT в original)
        x1_r, y1_r, x2_r, y2_r = tr.best_bbox_rotated
        x1_o, y1_o, x2_o, y2_o = rotate_bbox_back_to_original(
            (x1_r, y1_r, x2_r, y2_r), H_orig, W_orig)
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(tr.best_ts)
        fields["x_min"] = round(float(x1_o), 1); fields["y_min"] = round(float(y1_o), 1)
        fields["x_max"] = round(float(x2_o), 1); fields["y_max"] = round(float(y2_o), 1)
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
    p.add_argument("--conf", type=float, default=0.1)
    args = p.parse_args()
    process_video(args.video, fps_sample=args.fps, output_csv=args.output,
                  filename_in_csv=args.filename_in_csv, upscale=args.upscale,
                  yolo_conf=args.conf)


if __name__ == "__main__":
    main()
