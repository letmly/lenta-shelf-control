"""Прогон pipeline_v3 на всех 5 размеченных видео + evaluate.

Цель: получить полную метрику по всему датасету.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_v3 import process_video
from evaluate import match_rows, compare_fields, FIELDS_TO_COMPARE

VIDEOS = [
    ("25_12-20", "25_12-20/2.mp4"),  # GT для другого сегмента, но прогон есть
    ("25_2-10",  "25_2-10"),
    ("26_12-20", "26_12-20.mp4"),
    ("43_15",    "43_15.mp4"),
    ("49_5",     "49_5.mp4"),
]

results = {}
for name, filename_in_csv in VIDEOS:
    video = ROOT / f"data/labeled/{name}/{name}.mp4"
    if not video.exists():
        print(f"skip {name} (no video)")
        continue
    out_csv = ROOT / f"outputs/pipeline_v3/{name}.csv"
    df_pred = process_video(video, fps_sample=3, output_csv=out_csv,
                             filename_in_csv=filename_in_csv, upscale=2.0)
    gt_csv = ROOT / f"data/labeled/{name}/{name}.csv"
    df_gt = pd.read_csv(gt_csv, encoding="utf-8")
    matches = match_rows(df_pred.copy(), df_gt.copy())
    scores = []
    for gt_idx, p_idx, mtype in matches:
        if p_idx is None: scores.append(0); continue
        s, _ = compare_fields(df_pred.iloc[p_idx], df_gt.iloc[gt_idx])
        scores.append(s)
    matched = sum(1 for m in matches if m[1] is not None)
    qualified = sum(1 for s in scores if s >= 0.8)
    mean_score = sum(scores)/len(scores) if scores else 0
    results[name] = {
        "gt_total": len(df_gt),
        "matched": matched,
        "mean_score": mean_score,
        "qualified_>=80%": qualified,
    }
    print(f"\n  >>> {name}: matched={matched}/{len(df_gt)} ({matched/len(df_gt):.0%}), "
          f"mean_score={mean_score:.0%}, qualified={qualified}/{len(df_gt)} ({qualified/len(df_gt):.0%})")

print("\n\n=== TOTAL ===")
tot_gt = sum(r["gt_total"] for r in results.values())
tot_matched = sum(r["matched"] for r in results.values())
tot_qual = sum(r["qualified_>=80%"] for r in results.values())
print(f"GT total:               {tot_gt}")
print(f"Matched (any):          {tot_matched}/{tot_gt}  ({tot_matched/tot_gt:.0%})")
print(f"Qualified (>=80% полей): {tot_qual}/{tot_gt}  ({tot_qual/tot_gt:.0%})")

(ROOT / "outputs/pipeline_v3/summary.json").write_text(
    json.dumps({"per_video": results,
                "total_gt": tot_gt,
                "total_matched": tot_matched,
                "total_qualified": tot_qual,
                "main_metric": tot_qual / tot_gt if tot_gt else 0},
               ensure_ascii=False, indent=2), encoding="utf-8")
