"""Проверка: какому видео реально соответствует 25_12-20.csv —
labeled/25_12-20/25_12-20.mp4 или unlabeled/25_12-20.mp4?

Берём первые 4 GT-bbox из 25_12-20.csv, рисуем bbox на обоих видео
и сохраняем рядом для визуального сравнения.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data/labeled/25_12-20/25_12-20.csv"
V_LABELED = ROOT / "data/labeled/25_12-20/25_12-20.mp4"
V_UNLABELED = ROOT / "data/unlabeled/25_12-20.mp4"
OUT = ROOT / "outputs/match_check"
OUT.mkdir(parents=True, exist_ok=True)


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def annotate(video: Path, rows, tag: str):
    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1) * 1000
    print(f"\n{tag}: {video.name}  {w}x{h}  ~{dur:.0f}ms")
    for i, r in enumerate(rows):
        ts = r["frame_timestamp"]
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok:
            print(f"  sample {i}: cannot read at ts={ts}")
            continue
        x1, y1, x2, y2 = map(int, (r["x_min"], r["y_min"], r["x_max"], r["y_max"]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
        cv2.putText(frame, f"ts={int(ts)}ms", (max(x1, 30), max(y1 - 20, 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
        out = OUT / f"{tag}_sample{i}.jpg"
        cv2.imwrite(str(out), cv2.resize(frame, (1280, 720)))
        print(f"  sample {i}: ts={ts:.0f}ms bbox=({x1},{y1},{x2},{y2}) -> {out.name}")
    cap.release()


def main():
    df = pd.read_csv(CSV, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    # 4 case'а с разными timestamps
    rows = df.sample(4, random_state=7).to_dict("records")
    print("GT records picked:")
    for r in rows:
        print(f"  ts={r['frame_timestamp']:.0f}ms bbox=({r['x_min']:.0f},{r['y_min']:.0f},{r['x_max']:.0f},{r['y_max']:.0f}) barcode={r.get('barcode')}")
    annotate(V_LABELED, rows, "LABELED-folder")
    annotate(V_UNLABELED, rows, "UNLABELED-folder")
    print(f"\nСравни {OUT}\\LABELED-* vs {OUT}\\UNLABELED-* —")
    print("на каком наборе bbox реально попадает на ценник, тот и есть «настоящий» 2.mp4")


if __name__ == "__main__":
    main()
