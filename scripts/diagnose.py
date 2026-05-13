"""Диагностика: сохраняем GT-кропы на диск и проверяем pyzbar.

Цель: понять, почему baseline дал QR success rate = 0%.
Гипотезы:
    H_A. GT bbox/timestamps не соответствуют реальному видеофайлу (filename в CSV
         для 25/26 видео указывает на "<video>/N.mp4" — возможно нарезка).
    H_B. pyzbar не работает / libzbar сломан.
    H_C. Кропы слишком маленькие или QR неконтрастный.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "diagnose"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ru(s) -> float | None:
    if not isinstance(s, str): s = str(s)
    s = s.replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def test_pyzbar_on_synthetic() -> dict:
    """Генерируем простой ШК и проверяем, что pyzbar его декодирует."""
    # белое поле + чёрные полосы — простой Code128-подобный
    # вместо генерации возьмём готовый: ставим точки EAN-13 паттерна
    # Проще — сразу попробуем зашитую картинку: pyzbar.zbarinst
    # Здесь делаем тривиальный тест: чёрно-белые полосы, должны не декодироваться,
    # просто проверяем что функция отрабатывает без креша
    img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    for x in range(0, 400, 10):
        img[20:180, x:x+5] = 0
    res = pyzbar_decode(img)
    return {"pyzbar_call_ok": True, "decoded_from_dummy": len(res)}


def save_crops_per_video(csv_path: Path, video_path: Path, n: int = 6) -> dict:
    df = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(lambda v: parse_ru(v))
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "video not opened"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_ms = (total_frames / fps * 1000) if fps else 0
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    samples = df.sample(min(n, len(df)), random_state=42)
    saved = []
    decoded_total = 0
    for i, (_, r) in enumerate(samples.iterrows()):
        ts = r["frame_timestamp"]
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok:
            continue
        # save full frame with bbox drawn
        fx1, fy1, fx2, fy2 = map(int, (r["x_min"], r["y_min"], r["x_max"], r["y_max"]))
        annotated = frame.copy()
        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 0), 4)
        cv2.putText(annotated, f"ts={int(ts)}", (fx1, max(fy1 - 10, 30)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        frame_path = OUT / f"{video_path.stem}_sample{i}_frame.jpg"
        cv2.imwrite(str(frame_path), cv2.resize(annotated, (1280, 720)))
        # save crop
        h, w = frame.shape[:2]
        x1 = max(0, fx1); y1 = max(0, fy1); x2 = min(w, fx2); y2 = min(h, fy2)
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
            crop_path = OUT / f"{video_path.stem}_sample{i}_crop.jpg"
            cv2.imwrite(str(crop_path), crop)
            # pyzbar attempts
            attempts = []
            d1 = pyzbar_decode(crop)
            attempts.append(("raw", len(d1)))
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            d2 = pyzbar_decode(big)
            attempts.append(("2x", len(d2)))
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            d3 = pyzbar_decode(gray)
            attempts.append(("2x_gray", len(d3)))
            # Otsu
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            d4 = pyzbar_decode(otsu)
            attempts.append(("2x_otsu", len(d4)))
            best = max(d1 or d2 or d3 or d4 or [], key=lambda x: len(x.data) if x else 0, default=None)
            decoded = best.data.decode("utf-8", errors="ignore") if best else None
            if best: decoded_total += 1
            saved.append({
                "sample": i,
                "ts_ms": float(ts),
                "bbox": [fx1, fy1, fx2, fy2],
                "crop_size": [int(x2 - x1), int(y2 - y1)],
                "attempts": dict(attempts),
                "decoded": (decoded[:120] if decoded else None),
                "gt_barcode": str(r.get("barcode", "")),
                "gt_filename_field": str(r.get("filename", "")),
            })

    cap.release()
    return {
        "video": video_path.name,
        "video_res": [w_video, h_video],
        "video_duration_ms": duration_ms,
        "video_fps": fps,
        "csv_unique_filename_values": list(df["filename"].astype(str).unique()),
        "decoded_in_diagnostic": decoded_total,
        "samples": saved,
    }


def main() -> None:
    report = {"pyzbar_self_test": test_pyzbar_on_synthetic(), "videos": []}

    for name in ("25_12-20", "26_12-20", "43_15"):
        csv = ROOT / f"data/labeled/{name}/{name}.csv"
        mp4 = ROOT / f"data/labeled/{name}/{name}.mp4"
        if not (csv.exists() and mp4.exists()):
            print(f"skip {name}")
            continue
        r = save_crops_per_video(csv, mp4, n=6)
        report["videos"].append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "samples"}, ensure_ascii=False, indent=2))
        for s in r.get("samples", []):
            print(f"  sample {s['sample']}  crop={s['crop_size']}  decoded={'YES' if s['decoded'] else 'no'}  "
                  f"gt_barcode={s['gt_barcode'][:13]}  gt_file={s['gt_filename_field']}")

    out = OUT / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] crops + report -> {OUT}")


if __name__ == "__main__":
    main()
