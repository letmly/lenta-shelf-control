"""End-to-end pipeline v3: multicolor детектор + sample-conform CSV-формат.

Что меняется vs v2:
    1. MULTICOLOR детектор: red, yellow, green, white — пишем реальный color в CSV.
    2. CSV writer строго по sample.csv формату:
       - десятичный = точка
       - bbox с десятичной частью (1023.4, не 1023)
       - 'нет' пишем явно (по умолчанию для незаполненных полей оставляем пусто
         — это значит "не распознано")
    3. Color-aware extension: красные/жёлтые акционные → extend вниз 60%,
       зелёные/белые МНЦ → без extension (квадратные).
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent

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

HSV_RANGES = {
    "red":    [((0, 80, 60), (10, 255, 255)), ((170, 80, 60), (180, 255, 255))],
    "yellow": [((18, 80, 80), (35, 255, 255))],
    "green":  [((40, 60, 60), (85, 255, 255))],
}


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


# ----------------------------- detector ---------------------------------------

def detect_color_mask(frame, hsv, ranges, color_name, min_area):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges: mask |= cv2.inRange(hsv, lo, hi)
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H = frame.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area: continue
        ar = w / h if h > 0 else 0
        if not (0.25 < ar < 4): continue
        # для красных горизонтальных АПЦ — extend вниз для захвата белой части
        if color_name == "red" and ar > 0.9:
            ext_h = int(h * 1.6)
            bb = [x, y, x + w, min(H, y + ext_h)]
        else:
            bb = [x, y, x + w, y + h]
        boxes.append((bb, color_name))
    return boxes


def detect_multicolor(frame, min_area=5000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    all_boxes = []
    for color, ranges in HSV_RANGES.items():
        ma = min_area * 2 if color == "green" else min_area
        all_boxes.extend(detect_color_mask(frame, hsv, ranges, color, ma))
    return nms_color(all_boxes, iou_th=0.3)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def nms_color(boxes, iou_th=0.3):
    if not boxes: return []
    # сортируем по площади убывая
    boxes_sorted = sorted(boxes, key=lambda x: -((x[0][2]-x[0][0])*(x[0][3]-x[0][1])))
    keep = []
    for bc in boxes_sorted:
        if all(iou(bc[0], kc[0]) <= iou_th for kc in keep):
            keep.append(bc)
    return keep


# ----------------------------- tracker ----------------------------------------

@dataclass
class Track:
    track_id: int
    bbox: list
    color: str
    first_ts: float
    last_ts: float
    frames: list = field(default_factory=list)  # [(ts, bbox, frame_img)]
    is_active: bool = True


class Tracker:
    def __init__(self, iou_th=0.25, max_missed_ms=600):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.iou_th = iou_th
        self.max_missed_ms = max_missed_ms

    def update(self, dets, ts_ms, frame):
        for t in self.tracks:
            if t.is_active and (ts_ms - t.last_ts) > self.max_missed_ms:
                t.is_active = False
        used = set()
        for bbox, color in dets:
            best_t = None; best_iou = 0
            for t in self.tracks:
                if not t.is_active or t.track_id in used: continue
                if t.color != color: continue  # матчим только тот же цвет
                v = iou(bbox, t.bbox)
                if v > best_iou: best_iou = v; best_t = t
            if best_t and best_iou >= self.iou_th:
                best_t.bbox = bbox; best_t.last_ts = ts_ms
                best_t.frames.append((ts_ms, bbox, frame))
                used.add(best_t.track_id)
            else:
                t = Track(self.next_id, bbox, color, ts_ms, ts_ms,
                          frames=[(ts_ms, bbox, frame)])
                self.tracks.append(t); self.next_id += 1


def sharpness(img):
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def best_frame(track: Track):
    return max(track.frames, key=lambda x: sharpness(
        x[2][int(x[1][1]):int(x[1][3]), int(x[1][0]):int(x[1][2])]))


# ----------------------------- OCR parser -------------------------------------

DISCOUNT_RE = re.compile(r"-?(\d{1,2})\s*%")
DATE_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})(?:\s+(\d{1,2}:\d{2}))?")
TIME_RE = re.compile(r"(\d{1,2}:\d{2})")
PURE_PRICE_INT_RE = re.compile(r"^\d{2,4}$")
TWO_DIGIT_RE = re.compile(r"^\d{2}$")
BARCODE_RE = re.compile(r"^\d{8,14}$")
ID_SKU_RE = re.compile(r"^\d{10,12}$")
WORD_RE = re.compile(r"[A-Za-zА-Яа-я]")


def _items_from_ocr(ocr_result):
    items = []
    if not ocr_result: return items
    for bb, text, conf in ocr_result:
        xs = [p[0] for p in bb]; ys = [p[1] for p in bb]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        items.append({"text": str(text).strip(),
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "w": x2-x1, "h": y2-y1,
                      "cx": (x1+x2)/2, "cy": (y1+y2)/2,
                      "conf": float(conf)})
    return items


def _find_price_with_cents(items, exclude_text=None, prefer_largest=True):
    """Ищет цену: целая часть = крупное число + копейки = 2-цифры рядом."""
    candidates = [it for it in items if PURE_PRICE_INT_RE.fullmatch(it["text"])
                  and it["text"] != exclude_text]
    if not candidates: return ""
    if prefer_largest:
        candidates.sort(key=lambda x: -x["h"])
    big = candidates[0]
    whole = big["text"]
    # ищем копейки 2-цифры рядом, выше середины big (superscript), правее
    for it in items:
        if it is big: continue
        if not TWO_DIGIT_RE.fullmatch(it["text"]): continue
        if it["h"] >= big["h"] * 0.85: continue
        # x: должно быть рядом
        if abs(it["cx"] - big["x2"]) > big["w"] * 0.6: continue
        # y: должно быть в районе big (выше нижней границы)
        if abs(it["cy"] - big["cy"]) > big["h"] * 1.2: continue
        return f"{whole}.{it['text']}"
    return whole


def parse_ocr(ocr_result, color: str) -> dict:
    fields = {col: "" for col in CSV_COLUMNS}
    fields["color"] = color
    items = _items_from_ocr(ocr_result)
    if not items: return fields

    # discount %
    for it in items:
        m = DISCOUNT_RE.search(it["text"])
        if m:
            fields["discount_amount"] = f"-{m.group(1)}%"
            break

    # date + time
    date_str = ""; time_str = ""
    for it in items:
        m = DATE_RE.search(it["text"])
        if m:
            date_str = m.group(1)
            if m.group(2): time_str = m.group(2)
            break
    if date_str and not time_str:
        for it in items:
            m = TIME_RE.search(it["text"])
            if m: time_str = m.group(1); break
    if date_str:
        fields["print_datetime"] = f"{date_str} {time_str}" if time_str else date_str

    # price_card — самая крупная цена
    fields["price_card"] = _find_price_with_cents(items)

    # price_default — вторая по размеру (исключив price_card)
    pc_whole = fields["price_card"].split(".")[0] if fields["price_card"] else None
    fields["price_default"] = _find_price_with_cents(items, exclude_text=pc_whole)

    # barcode — длинное чисто-цифровое
    bcs = [it for it in items if BARCODE_RE.fullmatch(it["text"])]
    if bcs:
        bcs.sort(key=lambda x: (-len(x["text"]), -x["conf"]))
        fields["barcode"] = bcs[0]["text"]

    # id_sku
    for it in items:
        if ID_SKU_RE.fullmatch(it["text"]) and it["text"] != fields["barcode"]:
            fields["id_sku"] = it["text"]
            break

    # product_name
    used_texts = {fields.get(k, "") for k in ("discount_amount", "print_datetime", "barcode", "id_sku")}
    name_parts = []
    for it in items:
        if it["text"] in used_texts: continue
        if not WORD_RE.search(it["text"]): continue
        if DATE_RE.search(it["text"]): continue
        digits = sum(c.isdigit() for c in it["text"])
        letters = sum(c.isalpha() for c in it["text"])
        if digits > letters: continue
        if len(it["text"]) < 2: continue
        name_parts.append((it["y1"], it["x1"], it["text"]))
    name_parts.sort()
    if name_parts:
        fields["product_name"] = " ".join(p[2] for p in name_parts[:10])

    return fields


# ----------------------------- pipeline ---------------------------------------

def process_video(video_path: Path, fps_sample=3, output_csv=None, filename_in_csv=None, upscale=2.0):
    output_csv = output_csv or Path(f"outputs/pipeline_v3/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== Processing {video_path.name} ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / fps_sample))
    print(f"  src fps={src_fps:.1f}  frames={n_frames}  step={step}")

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
            tracker.update(dets, ts_ms, frame.copy())
        fi += 1
    pbar.close()
    cap.release()

    tracks = [t for t in tracker.tracks if len(t.frames) >= 2]
    print(f"  tracks raw: {len(tracker.tracks)} -> filtered (>=2 frames): {len(tracks)}")

    print(f"  initializing RapidOCR + upscale x{upscale}...")
    ocr = RapidOCR()
    rows = []
    for tr in tqdm(tracks, desc="OCR"):
        ts, bbox, frame = best_frame(tr)
        x1, y1, x2, y2 = bbox
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20: continue
        crop_up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        try: result, _ = ocr(crop_up)
        except Exception: result = None
        fields = parse_ocr(result, tr.color)
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(ts)
        fields["x_min"] = round(float(x1), 1); fields["y_min"] = round(float(y1), 1)
        fields["x_max"] = round(float(x2), 1); fields["y_max"] = round(float(y2), 1)
        rows.append(fields)

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
    p.add_argument("--upscale", type=float, default=2.0)
    args = p.parse_args()
    process_video(args.video, fps_sample=args.fps, output_csv=args.output,
                  filename_in_csv=args.filename_in_csv, upscale=args.upscale)


if __name__ == "__main__":
    main()
