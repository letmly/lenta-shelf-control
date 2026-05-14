"""End-to-end pipeline v2: с улучшенным OCR и парсером.

Что меняется vs v1:
    1. UPSCALE кропа x2 перед OCR (главная проблема v1 — OCR не работал на мелких).
    2. Tracker: IoU 0.4, мин длительность 3 кадра, max_missed 300ms.
    3. Парсер цен: склеивает "1631" + "99" superscript → "1631.99".
    4. Парсер product_name: только кириллица/латиница без цифр/процентов.
    5. Парсер barcode: 13-значные числа, фильтр id_sku.
    6. Парсер даты: regex по полному формату DD.MM.YYYY HH:MM.
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


# ----------------------------- detector ---------------------------------------

def detect_redmask(frame, min_area=5000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask = m1 | m2
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H = frame.shape[0]
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area: continue
        ar = w / h if h > 0 else 0
        if not (0.25 < ar < 4): continue
        # extend вниз на 50%, чтобы захватить белую часть с QR и штрихкодом
        ext_h = int(h * 1.6)
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
    if not boxes: return []
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    idx = sorted(range(len(boxes)), key=lambda i: -areas[i])
    keep = []
    for i in idx:
        if all(iou(boxes[i], boxes[j]) <= iou_th for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep]


# ----------------------------- tracker ----------------------------------------

@dataclass
class Track:
    track_id: int
    bbox: list
    first_ts: float
    last_ts: float
    frames: list = field(default_factory=list)
    is_active: bool = True


class SimpleTracker:
    def __init__(self, iou_th=0.4, max_missed_ms=300):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.iou_th = iou_th
        self.max_missed_ms = max_missed_ms

    def update(self, dets, ts_ms, frame):
        for t in self.tracks:
            if t.is_active and (ts_ms - t.last_ts) > self.max_missed_ms:
                t.is_active = False
        used = set()
        for det in dets:
            best_t = None; best_iou = 0
            for t in self.tracks:
                if not t.is_active or t.track_id in used: continue
                v = iou(det, t.bbox)
                if v > best_iou: best_iou = v; best_t = t
            if best_t and best_iou >= self.iou_th:
                best_t.bbox = det; best_t.last_ts = ts_ms
                best_t.frames.append((ts_ms, det, frame))
                used.add(best_t.track_id)
            else:
                t = Track(self.next_id, det, ts_ms, ts_ms, frames=[(ts_ms, det, frame)])
                self.tracks.append(t); self.next_id += 1


# ----------------------------- best frame -------------------------------------

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
PURE_INT_RE = re.compile(r"^\d+$")
PURE_PRICE_INT_RE = re.compile(r"^\d{2,4}$")
TWO_DIGIT_RE = re.compile(r"^\d{2}$")
BARCODE_RE = re.compile(r"^\d{8,14}$")
ID_SKU_RE = re.compile(r"^\d{10,12}$")
WORD_RE = re.compile(r"[A-Za-zА-Яа-я]")


def _items_from_ocr(ocr_result):
    items = []
    if not ocr_result: return items
    for entry in ocr_result:
        bb, text, conf = entry
        xs = [p[0] for p in bb]; ys = [p[1] for p in bb]
        x1, y1 = min(xs), min(ys); x2, y2 = max(xs), max(ys)
        items.append({
            "text": str(text).strip(),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "area": (x2 - x1) * (y2 - y1),
            "conf": float(conf),
        })
    return items


def _find_price_card(items):
    """price_card = самая КРУПНАЯ цена (число 2-4 цифры).
    Если рядом справа сверху есть 2-цифровое число → склеиваем как копейки.
    """
    candidates = [it for it in items if PURE_PRICE_INT_RE.fullmatch(it["text"])]
    if not candidates: return ""
    # самый высокий (по высоте bbox = размер шрифта)
    big = max(candidates, key=lambda x: x["h"])
    whole = big["text"]
    # ищем "99" рядом справа сверху (superscript) — bbox должен быть выше центра big и правее
    cents = None
    for it in items:
        if it is big: continue
        if not TWO_DIGIT_RE.fullmatch(it["text"]): continue
        # должно быть мельче big
        if it["h"] >= big["h"] * 0.7: continue
        # рядом по x (правее или внутри bbox big)
        if it["x1"] < big["x2"] - big["w"] * 0.3: continue
        if it["x1"] > big["x2"] + big["w"] * 0.5: continue
        # выше или на уровне средины big
        if it["cy"] > big["cy"]: continue
        cents = it["text"]
        break
    if cents:
        return f"{whole}.{cents}"
    return whole


def _find_price_default(items, exclude_text=None):
    """price_default = вторая по размеру цена (зачёркнутая обычно сверху)."""
    candidates = [it for it in items if PURE_PRICE_INT_RE.fullmatch(it["text"])
                  and it["text"] != exclude_text]
    if len(candidates) < 1: return ""
    # сортируем по высоте
    candidates.sort(key=lambda x: -x["h"])
    # берём второй кандидат если первый ещё не взят как card; иначе первый
    # exclude_text — текст price_card; первый из оставшихся
    big = candidates[0]
    whole = big["text"]
    cents = None
    for it in items:
        if it is big: continue
        if not TWO_DIGIT_RE.fullmatch(it["text"]): continue
        if it["h"] >= big["h"] * 0.9: continue
        if abs(it["cy"] - big["cy"]) > big["h"] * 1.2: continue
        if abs(it["cx"] - big["cx"]) > big["w"] * 1.5: continue
        # cents правее
        if it["x1"] < big["x2"] - big["w"] * 0.3: continue
        cents = it["text"]; break
    if cents:
        return f"{whole}.{cents}"
    return whole


def _find_discount(items):
    for it in items:
        m = DISCOUNT_RE.search(it["text"])
        if m:
            val = m.group(1)
            return f"-{val}%"
    return ""


def _find_date(items):
    """ищем дату в OCR-результатах. Можем встретить '04.01.2026' и '2:00' в разных bbox."""
    date_str = None; time_str = None
    for it in items:
        m = DATE_RE.search(it["text"])
        if m:
            date_str = m.group(1)
            if m.group(2):
                time_str = m.group(2)
                return f"{date_str} {time_str}"
    # ищем время отдельно
    if date_str:
        for it in items:
            m = re.search(r"(\d{1,2}:\d{2})", it["text"])
            if m:
                time_str = m.group(1)
                return f"{date_str} {time_str}"
        return date_str
    return ""


def _find_barcode(items):
    """Самая длинная цифровая последовательность (13 знаков EAN-13 / 8 знаков EAN-8)."""
    candidates = [it for it in items if BARCODE_RE.fullmatch(it["text"])]
    if not candidates: return ""
    # предпочитаем 13-значные
    by_len = sorted(candidates, key=lambda x: (-len(x["text"]), -x["conf"]))
    return by_len[0]["text"]


def _find_id_sku(items, exclude=None):
    """id_sku обычно 10-12 цифр, мельче чем штрихкод."""
    candidates = [it for it in items if ID_SKU_RE.fullmatch(it["text"]) and it["text"] != exclude]
    if not candidates: return ""
    return sorted(candidates, key=lambda x: -x["conf"])[0]["text"]


def _find_product_name(items, used_texts):
    """Склеиваем многострочное название из текстов с буквами."""
    parts = []
    for it in items:
        if it["text"] in used_texts: continue
        if not WORD_RE.search(it["text"]): continue  # нет букв — пропустим
        if DATE_RE.search(it["text"]): continue
        if DISCOUNT_RE.search(it["text"]): continue
        if len(it["text"]) < 2: continue
        # отбрасываем строки слишком цифровые
        digits = sum(c.isdigit() for c in it["text"])
        letters = sum(c.isalpha() for c in it["text"])
        if digits > letters: continue
        parts.append(it)
    # сортируем по y (сверху вниз)
    parts.sort(key=lambda x: x["y1"])
    return " ".join(p["text"] for p in parts[:10])


def parse_ocr(ocr_result) -> dict:
    fields = {col: "" for col in CSV_COLUMNS}
    fields["color"] = "red"
    items = _items_from_ocr(ocr_result)
    if not items: return fields

    fields["discount_amount"] = _find_discount(items)
    fields["print_datetime"] = _find_date(items)
    fields["price_card"] = _find_price_card(items)
    fields["price_default"] = _find_price_default(items, exclude_text=fields["price_card"].split(".")[0] if fields["price_card"] else None)
    fields["barcode"] = _find_barcode(items)
    fields["id_sku"] = _find_id_sku(items, exclude=fields["barcode"])

    # product name — все буквенные строки, не использованные ранее
    used = set()
    for it in items:
        for f in ("discount_amount", "print_datetime", "barcode", "id_sku"):
            if fields[f] and fields[f] in it["text"]:
                used.add(it["text"])
    fields["product_name"] = _find_product_name(items, used)

    return fields


# ----------------------------- pipeline ---------------------------------------

def process_video(video_path: Path, fps_sample=3, output_csv=None, filename_in_csv=None, upscale=2.0):
    output_csv = output_csv or Path(f"outputs/pipeline_v2/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== Processing {video_path.name} ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / fps_sample))
    print(f"  src fps={src_fps:.1f}  frames={n_frames}  step={step}")

    tracker = SimpleTracker(iou_th=0.25, max_missed_ms=600)
    frame_idx = 0
    pbar = tqdm(total=n_frames, desc="frames")
    while True:
        ok, frame = cap.read()
        if not ok: break
        pbar.update(1)
        if frame_idx % step == 0:
            ts_ms = frame_idx * (1000.0 / src_fps)
            dets = nms(detect_redmask(frame), iou_th=0.3)
            tracker.update(dets, ts_ms, frame.copy())
        frame_idx += 1
    pbar.close()
    cap.release()

    # фильтр: треки минимум 2 кадра
    tracks = [t for t in tracker.tracks if len(t.frames) >= 2]
    print(f"  raw tracks: {len(tracker.tracks)} -> filtered >=2 frames: {len(tracks)}")

    print(f"  initializing RapidOCR + upscale x{upscale}...")
    ocr = RapidOCR()
    rows = []
    for tr in tqdm(tracks, desc="OCR"):
        ts, bbox, frame = best_frame(tr)
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20: continue
        # UPSCALE
        if upscale != 1.0:
            crop_up = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        else:
            crop_up = crop
        try:
            result, _ = ocr(crop_up)
        except Exception:
            result = None
        fields = parse_ocr(result)
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(ts)
        fields["x_min"] = x1; fields["y_min"] = y1
        fields["x_max"] = x2; fields["y_max"] = y2
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
