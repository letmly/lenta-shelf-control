"""End-to-end pipeline v1: видео → CSV.

Шаги:
    1. Sampling кадров с заданной FPS (по умолчанию 3 fps)
    2. RedMask детектор на каждом кадре
    3. NMS внутри кадра (убираем перекрывающиеся bbox-ы)
    4. Простой IoU-трекер между соседними кадрами
    5. Для каждого track: выбираем best frame по sharpness
    6. RapidOCR на best-frame crop
    7. Парсер OCR → CSV поля
    8. CSV writer в формате ТЗ
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

# CSV columns по ТЗ
CSV_COLUMNS = [
    "filename", "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols", "frame_timestamp",
    "x_min", "y_min", "x_max", "y_max",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]


# ----------------------------- detector ---------------------------------------

def detect_redmask(frame, min_area=5000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask = m1 | m2
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H = frame.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area: continue
        ar = w / h if h > 0 else 0
        if not (0.25 < ar < 4): continue
        # extend down 50%
        ext_h = int(h * 1.5)
        boxes.append([x, y, x + w, min(H, y + ext_h)])
    return boxes


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def nms(boxes, iou_th=0.3):
    """Простой NMS — оставляем самые большие, выкидываем перекрывающиеся."""
    if not boxes: return []
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = sorted(range(len(boxes)), key=lambda i: -areas[i])
    keep = []
    for i in idx:
        ok = True
        for j in keep:
            if iou(boxes[i], boxes[j]) > iou_th: ok = False; break
        if ok: keep.append(i)
    return [boxes[i] for i in keep]


# ----------------------------- tracker ----------------------------------------

@dataclass
class Track:
    track_id: int
    bbox: list  # последний bbox
    first_ts: float
    last_ts: float
    frames: list = field(default_factory=list)  # [(ts, bbox, frame_image)]
    is_active: bool = True


class SimpleIoUTracker:
    def __init__(self, iou_th=0.2, max_missed_ms=500):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.iou_th = iou_th
        self.max_missed_ms = max_missed_ms

    def update(self, detections, ts_ms, frame):
        """detections — list of bboxes на текущем кадре."""
        # деактивировать треки которые давно не видели
        for t in self.tracks:
            if t.is_active and (ts_ms - t.last_ts) > self.max_missed_ms:
                t.is_active = False

        # для каждого детектора найти best-matching активный трек
        used_tracks = set()
        for det in detections:
            best_t = None; best_iou = 0
            for t in self.tracks:
                if not t.is_active or t.track_id in used_tracks: continue
                v = iou(det, t.bbox)
                if v > best_iou:
                    best_iou = v; best_t = t
            if best_t and best_iou >= self.iou_th:
                best_t.bbox = det
                best_t.last_ts = ts_ms
                best_t.frames.append((ts_ms, det, frame))
                used_tracks.add(best_t.track_id)
            else:
                # новый трек
                t = Track(self.next_id, det, ts_ms, ts_ms,
                          frames=[(ts_ms, det, frame)])
                self.tracks.append(t)
                self.next_id += 1


# ----------------------------- best frame -------------------------------------

def sharpness(img):
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def best_frame_for_track(track: Track):
    best = max(track.frames, key=lambda x: sharpness(
        x[2][int(x[1][1]):int(x[1][3]), int(x[1][0]):int(x[1][2])]))
    return best  # (ts, bbox, full_frame)


# ----------------------------- OCR parser -------------------------------------

PRICE_RE = re.compile(r"(\d{1,4})[.,]?(\d{2})?")
DATE_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{2,4}(?:\s+\d{1,2}:\d{2})?)")
DISCOUNT_RE = re.compile(r"-?\d{1,2}\s*%")
SKU_RE = re.compile(r"^\d{10,12}$")
BARCODE_RE = re.compile(r"^\d{8,14}$")


def parse_ocr_to_fields(ocr_result) -> dict:
    """OCR-результат → словарь полей CSV.

    Грубая логика MVP:
    - крупный bbox с цифрами + '99' = price_card
    - '-XX%' = discount_amount
    - дата DD.MM.YYYY = print_datetime
    - длинная строка с латиницей/кириллицей = product_name
    - чисто цифры 10-13 знаков = barcode
    - чисто цифры 10-12 знаков и не похоже на barcode = id_sku
    """
    fields = {col: "" for col in CSV_COLUMNS}
    fields["color"] = "red"  # по нашему детектору
    if not ocr_result:
        return fields

    # сортируем по площади bbox убывая
    items = []
    for entry in ocr_result:
        bbox, text, conf = entry
        text = str(text).strip()
        # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
        w = max(xs) - min(xs); h = max(ys) - min(ys)
        items.append({"text": text, "w": w, "h": h, "area": w * h,
                      "y_center": (max(ys) + min(ys)) / 2, "conf": float(conf)})

    if not items: return fields

    # 1. discount_amount: первый текст содержащий % с цифрой
    for it in items:
        m = DISCOUNT_RE.search(it["text"])
        if m:
            fields["discount_amount"] = m.group(0).replace(" ", "").replace("%", "%")
            if not fields["discount_amount"].startswith("-"):
                fields["discount_amount"] = "-" + fields["discount_amount"]
            break

    # 2. print_datetime: первая строка содержащая дату
    for it in items:
        m = DATE_RE.search(it["text"])
        if m:
            fields["print_datetime"] = m.group(0).strip()
            break

    # 3. цена — самый крупный numeric-only text
    price_candidates = [it for it in items if re.fullmatch(r"\d{2,4}[.,]?\d{0,2}", it["text"])]
    if price_candidates:
        biggest = max(price_candidates, key=lambda x: x["area"])
        # нормализуем: запятая → точка
        fields["price_card"] = biggest["text"].replace(",", ".")

    # 4. barcode — цифры 8-14 знаков
    for it in items:
        if BARCODE_RE.match(it["text"]):
            fields["barcode"] = it["text"]
            break

    # 5. id_sku — цифры 10-12 (если не совпало с barcode)
    if not fields["id_sku"]:
        for it in items:
            if SKU_RE.match(it["text"]) and it["text"] != fields["barcode"]:
                fields["id_sku"] = it["text"]
                break

    # 6. product_name — все нецифровые строки склеиваем
    name_parts = []
    for it in items:
        # фильтр: не чистая цифра, не процент, не дата
        if re.fullmatch(r"[\d.,\s\-%]+", it["text"]): continue
        if DATE_RE.search(it["text"]): continue
        if len(it["text"]) < 2: continue
        name_parts.append(it["text"])
    if name_parts:
        fields["product_name"] = " ".join(name_parts[:8])

    return fields


# ----------------------------- main pipeline ----------------------------------

def process_video(video_path: Path, fps_sample=3, output_csv=None, video_filename_for_csv=None):
    output_csv = output_csv or Path(f"outputs/pipeline_v1/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if video_filename_for_csv is None:
        video_filename_for_csv = video_path.name

    print(f"\n=== Processing {video_path.name} ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("cannot open"); return None

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = max(1, int(src_fps / fps_sample))
    duration_ms = (n_frames / src_fps) * 1000
    print(f"  src fps={src_fps:.1f}  frames={n_frames}  duration={duration_ms/1000:.1f}s")
    print(f"  sampling every {step_frames}-th frame (~{fps_sample} fps effective)")

    tracker = SimpleIoUTracker(iou_th=0.2, max_missed_ms=int(1500 / fps_sample))

    # ---- проход по видео ----
    frame_idx = 0
    pbar = tqdm(total=n_frames, desc="frames")
    while True:
        ok, frame = cap.read()
        if not ok: break
        pbar.update(1)
        if frame_idx % step_frames == 0:
            ts_ms = frame_idx * (1000.0 / src_fps)
            dets = detect_redmask(frame)
            dets = nms(dets, iou_th=0.3)
            tracker.update(dets, ts_ms, frame.copy())
        frame_idx += 1
    pbar.close()
    cap.release()

    print(f"  total tracks: {len(tracker.tracks)}")

    # ---- OCR на каждом треке ----
    print(f"  initializing RapidOCR...")
    ocr = RapidOCR()
    rows = []
    for track in tqdm(tracker.tracks, desc="OCR"):
        if not track.frames: continue
        ts, bbox, frame = best_frame_for_track(track)
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: continue
        try:
            result, _ = ocr(crop)
        except Exception:
            result = None
        fields = parse_ocr_to_fields(result)
        fields["filename"] = video_filename_for_csv
        fields["frame_timestamp"] = int(ts)
        fields["x_min"] = x1; fields["y_min"] = y1
        fields["x_max"] = x2; fields["y_max"] = y2
        # QR-поля оставляем пусто
        rows.append(fields)

    # ---- запись CSV ----
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
    args = p.parse_args()
    process_video(args.video, fps_sample=args.fps, output_csv=args.output,
                  video_filename_for_csv=args.filename_in_csv)


if __name__ == "__main__":
    main()
