"""Прогон pipeline_v4 на всех 5 размеченных видео + evaluate.

Memory-efficient версия: треки хранят только best_crop, не полные кадры.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_v4 import process_video
from evaluate import match_rows, compare_fields

VIDEOS = [
    ("25_12-20", "25_12-20/2.mp4"),
    ("25_2-10",  "25_2-10"),
    ("26_12-20", "26_12-20.mp4"),
    ("43_15",    "43_15.mp4"),
    ("49_5",     "49_5.mp4"),
]

results = {}
for name, filename_in_csv in VIDEOS:
    video = ROOT / f"data/labeled/{name}/{name}.mp4"
    if not video.exists():
        print(f"skip {name}"); continue
    out_csv = ROOT / f"outputs/pipeline_v4/{name}.csv"
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
    pred_n = len(df_pred); gt_n = len(df_gt)
    results[name] = {"gt": gt_n, "pred": pred_n, "matched": matched,
                     "mean_score": round(mean_score, 3),
                     "qualified_80": qualified}
    print(f"\n  >>> {name}: pred={pred_n}, gt={gt_n}, matched={matched}/{gt_n} ({matched/gt_n:.0%}), "
          f"mean={mean_score:.0%}, qual={qualified}/{gt_n} ({qualified/gt_n:.0%})")

print("\n\n=== TOTAL ===")
tot_gt = sum(r["gt"] for r in results.values())
tot_pred = sum(r["pred"] for r in results.values())
tot_matched = sum(r["matched"] for r in results.values())
tot_qual = sum(r["qualified_80"] for r in results.values())
print(f"GT total:               {tot_gt}")
print(f"Pred total:             {tot_pred}  (over-detection: {tot_pred/tot_gt:.1f}x)")
print(f"Matched:                {tot_matched}/{tot_gt}  ({tot_matched/tot_gt:.0%})")
print(f"Qualified (>=80%):      {tot_qual}/{tot_gt}  ({tot_qual/tot_gt:.0%})  <-- MAIN")

(ROOT / "outputs/pipeline_v4/summary.json").write_text(
    json.dumps({"per_video": results,
                "total_gt": tot_gt, "total_pred": tot_pred,
                "total_matched": tot_matched, "total_qualified": tot_qual,
                "main_metric": tot_qual / tot_gt if tot_gt else 0},
               ensure_ascii=False, indent=2), encoding="utf-8")
