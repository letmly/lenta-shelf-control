"""Pipeline v5: фокусные правки парсера по результатам debug на v4.

Изменения vs v4:
    1. product_name: убран жёсткий фильтр letters > digits, добавлены 1-словные.
    2. price_card vs price_default различаются по высоте шрифта:
       - самый высокий = price_card
       - 70-95% высоты от него = price_default
    3. Склейка копеек "99" может быть и СПРАВА (не только superscript).
    4. barcode: если 13-значного нет — пробуем склеивать 2 соседних цифровых
       сегмента OCR в одну 13-цифровую строку (по сумме длин == 13).
    5. Min area детектора 4000 (было 5000) — захватываем мельче ценники.
    6. Upscale OCR x3 (было x2).
"""
from __future__ import annotations

import argparse
import gc
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
    "yellow": [((18, 100, 100), (35, 255, 255))],
    "green":  [((40, 100, 60), (85, 255, 255))],
}


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


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
        if color_name == "red" and ar > 0.9:
            ext_h = int(h * 1.6)
            bb = [x, y, x + w, min(H, y + ext_h)]
        else:
            bb = [x, y, x + w, y + h]
        boxes.append((bb, color_name))
    return boxes


def detect_multicolor(frame, min_area=4000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    all_boxes = []
    for color, ranges in HSV_RANGES.items():
        ma = min_area * 2 if color in ("green", "yellow") else min_area
        all_boxes.extend(detect_color_mask(frame, hsv, ranges, color, ma))
    return global_nms(all_boxes, iou_th=0.3)


def global_nms(boxes, iou_th=0.3):
    if not boxes: return []
    boxes_sorted = sorted(boxes, key=lambda x: -((x[0][2]-x[0][0])*(x[0][3]-x[0][1])))
    keep = []
    for bc in boxes_sorted:
        if all(iou(bc[0], kc[0]) <= iou_th for kc in keep):
            keep.append(bc)
    return keep


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
    is_active: bool = True
    n_frames: int = 0
    best_ts: float = 0.0
    best_bbox: list = field(default_factory=list)
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
            if t.is_active and (ts_ms - t.last_ts) > self.max_missed_ms:
                t.is_active = False
        used = set()
        for bbox, color in dets:
            best_t = None; best_iou = 0
            for t in self.tracks:
                if not t.is_active or t.track_id in used: continue
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
        x1, y1, x2, y2 = map(int, bbox)
        if x2 <= x1 or y2 <= y1: return
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20: return
        sh = sharpness(crop)
        if sh > track.best_sharpness:
            track.best_sharpness = sh; track.best_ts = ts_ms
            track.best_bbox = list(bbox); track.best_crop = crop.copy()


# ----------------------------- IMPROVED parser --------------------------------

DISCOUNT_RE = re.compile(r"-?(\d{1,2})\s*%")
DATE_RE = re.compile(r"(\d{1,2}\.\d{1,2}\.\d{4})(?:\s+(\d{1,2}:\d{2}))?")
TIME_RE = re.compile(r"(\d{1,2}:\d{2})")
PURE_PRICE_INT_RE = re.compile(r"^\d{2,4}$")
TWO_DIGIT_RE = re.compile(r"^\d{2}$")
DIGITS_RE = re.compile(r"^\d+$")
BARCODE_RE = re.compile(r"^\d{8,14}$")
ID_SKU_RE = re.compile(r"^\d{9,13}$")  # расширил с 10-12 до 9-13
WORD_RE = re.compile(r"[A-Za-zА-Яа-я]")
CODE_RE = re.compile(r"\d{1,3}_\d{4,8}")  # формат '01_025019'
SPECIAL_SYM_RE = re.compile(r"^[ШКЛшкл]$")


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


def _find_cents_for(big, items):
    """Ищем 2-цифровые копейки рядом с big (или сверху-справа, или просто справа)."""
    for it in items:
        if it is big: continue
        if not TWO_DIGIT_RE.fullmatch(it["text"]): continue
        # размер: меньше big (но не слишком)
        if it["h"] >= big["h"] * 0.95: continue
        if it["h"] < big["h"] * 0.15: continue
        # x: рядом справа от big или внутри его правой половины
        x_ok = (it["x1"] >= big["cx"]) and (it["x1"] <= big["x2"] + big["w"] * 0.7)
        if not x_ok: continue
        # y: пересекается с big по вертикали (внутри big по высоте)
        y_ok = (it["cy"] >= big["y1"] - big["h"] * 0.2) and (it["cy"] <= big["y2"] + big["h"] * 0.2)
        if not y_ok: continue
        return it["text"]
    return None


def _find_price_card_and_default(items):
    """price_card = самая КРУПНАЯ цена. price_default = вторая (>=60% высоты card)."""
    candidates = [it for it in items if PURE_PRICE_INT_RE.fullmatch(it["text"])]
    if not candidates: return "", ""
    # сортируем по высоте
    candidates.sort(key=lambda x: -x["h"])
    big_card = candidates[0]
    card_cents = _find_cents_for(big_card, items)
    price_card = f"{big_card['text']}.{card_cents}" if card_cents else big_card["text"]

    # price_default: следующий по высоте >=60% card и не пересекается с card
    big_def = None
    for it in candidates[1:]:
        if it["h"] < big_card["h"] * 0.5: continue
        if it["h"] > big_card["h"] * 1.05: continue
        if abs(it["cx"] - big_card["cx"]) < big_card["w"] * 0.5 and \
           abs(it["cy"] - big_card["cy"]) < big_card["h"] * 0.5:
            continue  # перекрывается с big_card
        big_def = it; break
    price_default = ""
    if big_def:
        def_cents = _find_cents_for(big_def, items)
        price_default = f"{big_def['text']}.{def_cents}" if def_cents else big_def["text"]
    return price_card, price_default


def _find_barcode(items):
    """Сначала ищем готовое 8-14 цифр. Если нет — пробуем склеить 2 соседних."""
    direct = [it for it in items if BARCODE_RE.fullmatch(it["text"])]
    if direct:
        direct.sort(key=lambda x: (-len(x["text"]), -x["conf"]))
        return direct[0]["text"]
    # склейка: 2 соседних цифровых блока, сумма длин = 13 или 8
    digit_blocks = [it for it in items if DIGITS_RE.fullmatch(it["text"]) and 4 <= len(it["text"]) <= 12]
    digit_blocks.sort(key=lambda x: (x["y1"], x["x1"]))
    for i, a in enumerate(digit_blocks):
        for j, b in enumerate(digit_blocks):
            if i == j: continue
            # должны быть на одной горизонтали (по y близко)
            if abs(a["cy"] - b["cy"]) > a["h"] * 0.7: continue
            # b справа от a, и достаточно близко
            if b["x1"] < a["x2"]: continue
            if b["x1"] - a["x2"] > a["w"]: continue
            merged = a["text"] + b["text"]
            if len(merged) in (8, 13, 12, 13, 14):
                return merged
    return ""


def _find_id_sku(items, exclude):
    for it in items:
        if ID_SKU_RE.fullmatch(it["text"]) and it["text"] != exclude:
            return it["text"]
    return ""


def parse_ocr(ocr_result, color: str) -> dict:
    fields = {col: "" for col in CSV_COLUMNS}
    fields["color"] = color
    items = _items_from_ocr(ocr_result)
    if not items: return fields

    # discount
    for it in items:
        m = DISCOUNT_RE.search(it["text"])
        if m: fields["discount_amount"] = f"-{m.group(1)}%"; break

    # date + time
    date_str = ""; time_str = ""
    for it in items:
        m = DATE_RE.search(it["text"])
        if m: date_str = m.group(1); time_str = m.group(2) or ""; break
    if date_str and not time_str:
        for it in items:
            m = TIME_RE.search(it["text"])
            if m: time_str = m.group(1); break
    if date_str:
        fields["print_datetime"] = f"{date_str} {time_str}" if time_str else date_str

    # prices
    pc, pd = _find_price_card_and_default(items)
    fields["price_card"] = pc; fields["price_default"] = pd

    # barcode (с склейкой)
    fields["barcode"] = _find_barcode(items)
    fields["id_sku"] = _find_id_sku(items, exclude=fields["barcode"])

    # code: формат '01_025019' (см. sample.csv)
    for it in items:
        m = CODE_RE.search(it["text"])
        if m:
            fields["code"] = m.group(0); break

    # special_symbols: один символ Ш/К/Л
    for it in items:
        if SPECIAL_SYM_RE.fullmatch(it["text"]):
            fields["special_symbols"] = it["text"].upper(); break

    # product_name — БОЛЕЕ МЯГКИЙ фильтр
    used = {fields[k] for k in ("discount_amount","print_datetime","barcode","id_sku") if fields[k]}
    name_parts = []
    for it in items:
        if it["text"] in used: continue
        # содержит буквы — кандидат
        if not WORD_RE.search(it["text"]): continue
        # пропускаем если это дата/время/процент
        if DATE_RE.search(it["text"]) or TIME_RE.search(it["text"]): continue
        if DISCOUNT_RE.search(it["text"]): continue
        if len(it["text"]) < 2: continue
        # МЯГКИЙ фильтр: хотя бы 2 буквы
        letters = sum(c.isalpha() for c in it["text"])
        if letters < 2: continue
        name_parts.append((it["y1"], it["x1"], it["text"]))
    name_parts.sort()
    if name_parts:
        fields["product_name"] = " ".join(p[2] for p in name_parts[:12])
    return fields


# ----------------------------- pipeline ---------------------------------------

def process_video(video_path: Path, fps_sample=3, output_csv=None,
                  filename_in_csv=None, upscale=3.0, min_track_frames=3):
    output_csv = output_csv or Path(f"outputs/pipeline_v5/{video_path.stem}.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if filename_in_csv is None: filename_in_csv = video_path.name

    print(f"\n=== Processing {video_path.name} ===")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(src_fps / fps_sample))
    print(f"  src fps={src_fps:.1f} step={step}")

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
    print(f"  raw={len(tracker.tracks)} -> kept={len(tracks)}")
    print(f"  init OCR + upscale x{upscale}...")
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
        x1, y1, x2, y2 = tr.best_bbox
        fields["filename"] = filename_in_csv
        fields["frame_timestamp"] = int(tr.best_ts)
        fields["x_min"] = round(float(x1), 1); fields["y_min"] = round(float(y1), 1)
        fields["x_max"] = round(float(x2), 1); fields["y_max"] = round(float(y2), 1)
        rows.append(fields)
    print(f"  skipped: {skipped}")
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
