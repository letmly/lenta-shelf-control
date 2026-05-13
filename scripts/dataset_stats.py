"""Характеризация датасета: каких ценников у нас сколько и какие из них «лёгкие».

Считаем для каждого GT-ценника:
    - размер bbox (площадь в пикселях, доля от 4K кадра)
    - расстояние центра bbox от центра кадра (важно для fisheye — края сильно дисторсированы)
    - расстояние от края кадра (если bbox прижат к границе — может быть обрезан)
    - aspect ratio (квадратные ценники легче декодируются, сильно вытянутые = под углом)

И выдаём «бакеты»:
    EASY      bbox > 50000 px², в центре (расстояние < 30% диагонали), aspect ratio ~ 1:1.5
    MEDIUM    bbox 20000-50000 px², ИЛИ в среднем поясе
    HARD      bbox < 20000 px², ИЛИ край фишая (расстояние > 40% диагонали), ИЛИ extreme aspect
    UNUSABLE  bbox прижат к краю (< 50 px от границы), bbox < 5000 px²
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "dataset_stats"
OUT.mkdir(parents=True, exist_ok=True)

W4K, H4K = 3840, 2160
CENTER = np.array([W4K / 2, H4K / 2])
DIAG = np.sqrt(W4K**2 + H4K**2)

VIDEOS = ["26_12-20", "43_15"]  # 25 пропускаем — GT не соответствует видео


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def categorize(area, dist_norm, edge_min, aspect):
    if edge_min < 50 or area < 5000:
        return "UNUSABLE"
    if area < 20000 or dist_norm > 0.4 or aspect > 3 or aspect < 0.4:
        return "HARD"
    if area < 50000 or dist_norm > 0.25:
        return "MEDIUM"
    return "EASY"


def main():
    overall = []
    by_video = {}

    for vname in VIDEOS:
        df = pd.read_csv(ROOT / f"data/labeled/{vname}/{vname}.csv", encoding="utf-8")
        for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
            df[c] = df[c].apply(parse_ru)
        df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max"]).reset_index(drop=True)

        rows = []
        for _, r in df.iterrows():
            x1, y1, x2, y2 = r["x_min"], r["y_min"], r["x_max"], r["y_max"]
            w, h = x2 - x1, y2 - y1
            area = w * h
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist_center = np.sqrt((cx - CENTER[0])**2 + (cy - CENTER[1])**2)
            dist_norm = dist_center / (DIAG / 2)  # 0=центр, 1=угол кадра
            edge_min = min(x1, y1, W4K - x2, H4K - y2)  # дист. до ближайшего края
            aspect = w / h if h > 0 else 1
            cat = categorize(area, dist_norm, edge_min, aspect)
            rows.append({
                "bbox_w": w, "bbox_h": h, "area": area,
                "dist_norm_from_center": dist_norm,
                "edge_min_px": edge_min, "aspect_w_h": aspect,
                "category": cat,
                "barcode": str(r.get("barcode", "")).strip(),
            })
            overall.append({**rows[-1], "video": vname})

        cats = Counter(x["category"] for x in rows)
        by_video[vname] = {
            "total": len(rows),
            "categories": dict(cats),
            "rates": {k: v / len(rows) for k, v in cats.items()},
            "area_median": float(np.median([x["area"] for x in rows])),
            "area_min": float(np.min([x["area"] for x in rows])),
            "area_max": float(np.max([x["area"] for x in rows])),
            "edge_proximity_violations": sum(1 for x in rows if x["edge_min_px"] < 50),
            "fisheye_edge_count": sum(1 for x in rows if x["dist_norm_from_center"] > 0.4),
        }

    # общая статистика
    overall_cats = Counter(x["category"] for x in overall)
    total = len(overall)
    summary = {
        "videos": by_video,
        "overall": {
            "total": total,
            "categories": dict(overall_cats),
            "rates": {k: f"{v/total:.0%}" for k, v in overall_cats.items()},
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "rows.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")

    # человекочитаемый отчёт
    lines = ["# Характеризация датасета", ""]
    lines.append(f"**Всего ценников (26+43): {total}**")
    lines.append("")
    lines.append("## По категориям сложности")
    lines.append("")
    lines.append("| Категория | Сколько | Доля | Что значит |")
    lines.append("|---|---|---|---|")
    desc = {
        "EASY":     "крупный, в центре кадра, нормальное соотношение сторон — должны браться легко",
        "MEDIUM":   "средний размер ИЛИ в среднем поясе кадра — берутся при хороших декодерах",
        "HARD":     "мелкий (< 20K px²) ИЛИ на краю фишая (>40% от центра) ИЛИ сильно перекошен — низкие шансы single-frame",
        "UNUSABLE": "прижат к краю кадра (< 50px от границы) ИЛИ микро (<5K px²) — физически нечитаем",
    }
    for cat in ("EASY", "MEDIUM", "HARD", "UNUSABLE"):
        n = overall_cats.get(cat, 0)
        lines.append(f"| **{cat}** | {n} | {n/total:.0%} | {desc[cat]} |")
    lines.append("")
    lines.append("## Реалистичный потолок метрики")
    lines.append("")
    lines.append("Если решаем только EASY+MEDIUM = " + f"{sum(overall_cats.get(c,0) for c in ('EASY','MEDIUM'))/total:.0%} ценников.")
    lines.append("При single-frame OCR на качественных кропах ожидаем decoding success ~70-85%.")
    lines.append("→ Реалистичная верхняя граница нашей метрики: ~50-65%.")
    lines.append("→ Чтобы выйти выше — нужно качественно покрывать HARD категорию (fisheye undistort, multi-frame fusion с perspective rectify).")
    lines.append("")
    for vname, st in by_video.items():
        lines.append(f"## {vname}.mp4")
        lines.append(f"- Всего: {st['total']}")
        lines.append(f"- Категории: {st['categories']}")
        lines.append(f"- Доли: { {k: f'{v:.0%}' for k,v in st['rates'].items()} }")
        lines.append(f"- Площадь bbox: min={st['area_min']:.0f}  median={st['area_median']:.0f}  max={st['area_max']:.0f} px²")
        lines.append(f"- Прижатых к краю (< 50px): {st['edge_proximity_violations']}")
        lines.append(f"- На краю фишая (> 40% от центра): {st['fisheye_edge_count']}")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] report → {OUT/'report.md'}")


if __name__ == "__main__":
    main()
