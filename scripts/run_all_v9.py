"""Прогон pipeline_v9 на всех 5 видео + summary."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_v9 import process_video
from evaluate import match_rows, compare_fields

VIDEOS = [
    ("25_12-20", "25_12-20/2.mp4"),
    ("25_2-10",  "25_2-10"),
    ("26_12-20", "26_12-20.mp4"),
    ("43_15",    "43_15.mp4"),
    ("49_5",     "49_5.mp4"),
]

results = {}
for name, fn in VIDEOS:
    video = ROOT / f"data/labeled/{name}/{name}.mp4"
    if not video.exists(): continue
    out_csv = ROOT / f"outputs/pipeline_v9/{name}.csv"
    df_pred = process_video(video, fps_sample=3, output_csv=out_csv,
                             filename_in_csv=fn, upscale=3.0)
    if df_pred is None or len(df_pred) == 0:
        results[name] = {"gt":0, "pred":0, "matched":0, "mean_score":0, "qualified_80":0}
        continue
    df_gt = pd.read_csv(ROOT / f"data/labeled/{name}/{name}.csv", encoding="utf-8")
    matches = match_rows(df_pred.copy(), df_gt.copy())
    scores = []
    for gt_idx, p_idx, _ in matches:
        if p_idx is None: scores.append(0); continue
        s, _ = compare_fields(df_pred.iloc[p_idx], df_gt.iloc[gt_idx])
        scores.append(s)
    matched = sum(1 for m in matches if m[1] is not None)
    qual = sum(1 for s in scores if s >= 0.8)
    mean = sum(scores)/len(scores) if scores else 0
    results[name] = {"gt":len(df_gt), "pred":len(df_pred), "matched":matched,
                     "mean_score":round(mean,3), "qualified_80":qual}
    print(f"  >>> {name}: pred={len(df_pred)} gt={len(df_gt)} matched={matched} mean={mean:.0%} qual={qual}")

print("\n=== TOTAL ===")
tot_gt = sum(r["gt"] for r in results.values())
tot_pred = sum(r["pred"] for r in results.values())
tot_matched = sum(r["matched"] for r in results.values())
tot_qual = sum(r["qualified_80"] for r in results.values())
print(f"GT: {tot_gt}  Pred: {tot_pred}  Matched: {tot_matched}/{tot_gt} ({tot_matched/tot_gt:.0%})  Qual: {tot_qual}/{tot_gt} ({tot_qual/tot_gt:.0%})")

(ROOT / "outputs/pipeline_v9/summary.json").write_text(
    json.dumps({"per_video": results, "total_gt": tot_gt, "total_pred": tot_pred,
                "total_matched": tot_matched, "total_qualified": tot_qual,
                "main_metric": tot_qual / tot_gt if tot_gt else 0},
               ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] summary saved")
