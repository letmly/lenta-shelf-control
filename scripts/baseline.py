"""Baseline-замеры на размеченных видео (Д1).

Цель — получить 3 цифры до начала любых обучений:

    1. YOLO recall — сколько GT-ценников ловит pretrained YOLOv8n
       (без fine-tune) при IoU >= 0.3 с GT.
    2. QR success rate — на скольких GT-кропах pyzbar декодирует QR.
    3. Сколько ценников всего в GT, какое их распределение по размерам/timestamp.

Запуск:
    python scripts/baseline.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "labeled"
OUT = ROOT / "outputs" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = [
    DATA / "25_12-20" / "25_12-20.mp4",
    DATA / "26_12-20" / "26_12-20.mp4",
    DATA / "43_15"    / "43_15.mp4",
]

# IoU порог для зачёта детекции
IOU_TH = 0.3
# Сколько GT-ценников максимально сэмплируем на видео (для скорости)
SAMPLE_PER_VIDEO = 30


# ----------------------------- helpers ----------------------------------------

def parse_ru_number(s: str) -> float | None:
    """Парсит '2063,1' → 2063.1; пустую строку → None."""
    if not isinstance(s, str) or not s.strip():
        return None
    s = s.replace(",", ".").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def load_gt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    for col in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[col] = df[col].apply(lambda v: parse_ru_number(str(v)))
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    return df


def iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1); inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2); inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def grab_frame(cap: cv2.VideoCapture, ts_ms: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
    ok, frame = cap.read()
    return frame if ok else None


# ----------------------------- main steps -------------------------------------

def step_qr(df: pd.DataFrame, video_path: Path) -> dict:
    """Декодирует QR на GT-кропах. Возвращает success rate и сэмплы."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "error": "cannot open"}

    rows = df.sample(min(SAMPLE_PER_VIDEO, len(df)), random_state=42)
    success = 0; tried = 0; samples = []

    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"QR {video_path.name}"):
        frame = grab_frame(cap, r["frame_timestamp"])
        if frame is None:
            continue
        x1, y1, x2, y2 = map(int, (r["x_min"], r["y_min"], r["x_max"], r["y_max"]))
        h, w = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = frame[y1:y2, x1:x2]
        tried += 1
        decoded = pyzbar_decode(crop)
        # Иногда помогает upscale + grayscale
        if not decoded:
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            decoded = pyzbar_decode(gray)
        if decoded:
            success += 1
            samples.append({
                "ts_ms": r["frame_timestamp"],
                "barcode_gt": str(r.get("barcode", "")),
                "qr_payload_head": decoded[0].data.decode("utf-8", errors="ignore")[:200],
                "qr_types": [d.type for d in decoded],
            })

    cap.release()
    return {
        "video": video_path.name,
        "qr_attempted": tried,
        "qr_decoded": success,
        "qr_rate": success / tried if tried else 0.0,
        "samples": samples[:5],
    }


def step_yolo_recall(df: pd.DataFrame, video_path: Path) -> dict:
    """Считает recall pretrained YOLO по GT-bbox.

    Pretrained YOLOv8n обучен на COCO (80 классов, ценников нет).
    Мы запускаем его без фильтра классов — берём ВСЕ предсказания
    с conf >= 0.05 и проверяем IoU с GT. Это даёт «теоретический потолок»
    того, что YOLO в принципе хоть как-то находит на ценнике.
    После Д2 (fine-tune на price_tag) ожидаем resorb >> baseline.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"video": video_path.name, "error": "ultralytics not installed"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video": video_path.name, "error": "cannot open"}

    model = YOLO("yolov8n.pt")
    rows = df.sample(min(SAMPLE_PER_VIDEO, len(df)), random_state=42)
    matched = 0
    tried = 0
    avg_best_iou = []

    for _, r in tqdm(rows.iterrows(), total=len(rows), desc=f"YOLO {video_path.name}"):
        frame = grab_frame(cap, r["frame_timestamp"])
        if frame is None:
            continue
        tried += 1
        results = model.predict(frame, conf=0.05, verbose=False, max_det=200)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.zeros((0, 4))
        gt_box = (r["x_min"], r["y_min"], r["x_max"], r["y_max"])
        if len(boxes) == 0:
            avg_best_iou.append(0.0)
            continue
        ious = np.array([iou(gt_box, tuple(b)) for b in boxes])
        best = float(ious.max())
        avg_best_iou.append(best)
        if best >= IOU_TH:
            matched += 1

    cap.release()
    return {
        "video": video_path.name,
        "yolo_attempted": tried,
        "yolo_matched": matched,
        "yolo_recall@IoU>=0.3": matched / tried if tried else 0.0,
        "mean_best_iou": float(np.mean(avg_best_iou)) if avg_best_iou else 0.0,
        "note": "pretrained YOLOv8n (COCO), без fine-tune — ожидаемо низко",
    }


def step_dataset_stats(df: pd.DataFrame) -> dict:
    bw = df["x_max"] - df["x_min"]
    bh = df["y_max"] - df["y_min"]
    return {
        "rows": int(len(df)),
        "unique_filenames": int(df["filename"].nunique()),
        "bbox_w": {"min": float(bw.min()), "median": float(bw.median()), "max": float(bw.max())},
        "bbox_h": {"min": float(bh.min()), "median": float(bh.median()), "max": float(bh.max())},
        "ts_ms": {"min": float(df["frame_timestamp"].min()), "max": float(df["frame_timestamp"].max())},
        "color_distribution": dict(Counter(df["color"].astype(str))),
    }


# ----------------------------- runner -----------------------------------------

def main(skip_yolo: bool = False) -> None:
    report = {"videos": [], "started_at": time.time()}
    for video in VIDEOS:
        gt_csv = video.with_suffix(".csv")
        if not video.exists():
            print(f"⚠️  Видео не найдено: {video} — пропускаю")
            continue
        if not gt_csv.exists():
            print(f"⚠️  GT CSV не найден: {gt_csv} — пропускаю")
            continue
        print(f"\n=== {video.name} ===")
        df = load_gt(gt_csv)
        stats = step_dataset_stats(df)
        qr = step_qr(df, video)
        if skip_yolo:
            yolo = {"skipped": True}
        else:
            yolo = step_yolo_recall(df, video)
        report["videos"].append({"name": video.name, "stats": stats, "qr": qr, "yolo": yolo})
        # промежуточный лог
        print(json.dumps({"stats": stats, "qr": {k: v for k, v in qr.items() if k != "samples"}, "yolo": yolo}, ensure_ascii=False, indent=2))

    report["finished_at"] = time.time()
    out_path = OUT / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Отчёт сохранён: {out_path}")

    # summary
    print("\n=== SUMMARY ===")
    for v in report["videos"]:
        qr = v["qr"]
        yolo = v["yolo"]
        qr_rate = qr.get("qr_rate", 0.0)
        yolo_rec = yolo.get("yolo_recall@IoU>=0.3", "—")
        print(f"  {v['name']:20s}  QR: {qr_rate:.0%} ({qr.get('qr_decoded',0)}/{qr.get('qr_attempted',0)})  "
              f"YOLO recall@.3: {yolo_rec if isinstance(yolo_rec, str) else f'{yolo_rec:.0%}'}")


if __name__ == "__main__":
    skip_yolo = "--no-yolo" in sys.argv
    main(skip_yolo=skip_yolo)
