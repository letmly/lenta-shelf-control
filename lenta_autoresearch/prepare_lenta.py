"""FIXED eval harness для autoresearch. DO NOT MODIFY this file.

Содержит:
- SUBSET_VIDEO, SUBSET_GT — фиксированный тестовый набор
- evaluate_pipeline(pred_csv_path) — единственный способ замерить эксперимент

Output из evaluate_pipeline:
    {
        "matched": int,           # детекция: pred ↔ GT
        "matched_pct": float,     # = matched / |GT|
        "mean_score": float,      # средний score по полям на matched ценниках
        "qualified": int,         # ценников ≥80% полей правильно
        "qualified_pct": float,   # = qualified / |GT|
        "n_gt": int,
        "n_pred": int,
    }

Метрика для autoresearch: maximize qualified_pct (главная), tie-break by mean_score.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# === FIXED SUBSET ===
SUBSET_VIDEO = ROOT / "data" / "labeled" / "26_12-20" / "26_12-20.mp4"
SUBSET_GT = ROOT / "data" / "labeled" / "26_12-20" / "26_12-20.csv"
SUBSET_NAME = "26_12-20"  # для filename в pred CSV

# === EVAL via evaluate.py ===
import evaluate as ev


def evaluate_pipeline(pred_csv_path: str | Path) -> dict:
    """Compute metrics on pred CSV vs fixed GT subset."""
    pred_path = Path(pred_csv_path)
    if not pred_path.exists():
        return {"error": f"pred CSV not found: {pred_path}", "matched": 0, "qualified": 0, "n_gt": 0, "n_pred": 0,
                "matched_pct": 0.0, "mean_score": 0.0, "qualified_pct": 0.0}
    pred_df = pd.read_csv(pred_path, encoding="utf-8")
    gt_df = pd.read_csv(SUBSET_GT, encoding="utf-8")
    n_gt = len(gt_df); n_pred = len(pred_df)
    if n_pred == 0:
        return {"matched": 0, "qualified": 0, "n_gt": n_gt, "n_pred": 0,
                "matched_pct": 0.0, "mean_score": 0.0, "qualified_pct": 0.0}

    matches = ev.match_rows(pred_df.copy(), gt_df.copy())
    scores = []
    for gi, pi, _ in matches:
        if pi is None:
            scores.append(0.0); continue
        s, _ = ev.compare_fields(pred_df.iloc[pi], gt_df.iloc[gi])
        scores.append(s)

    matched = sum(1 for _, p, _ in matches if p is not None)
    qualified = sum(1 for s in scores if s >= 0.8)
    mean = sum(scores) / len(scores) if scores else 0.0

    return {
        "matched": matched,
        "matched_pct": matched / n_gt if n_gt else 0.0,
        "mean_score": mean,
        "qualified": qualified,
        "qualified_pct": qualified / n_gt if n_gt else 0.0,
        "n_gt": n_gt,
        "n_pred": n_pred,
    }


def print_summary(metrics: dict):
    """Final summary в формате, который grep-ится."""
    print("---")
    print(f"matched:       {metrics['matched']}/{metrics['n_gt']} ({metrics['matched_pct']:.4f})")
    print(f"mean_score:    {metrics['mean_score']:.4f}")
    print(f"qualified:     {metrics['qualified']}/{metrics['n_gt']} ({metrics['qualified_pct']:.4f})")
    print(f"n_pred:        {metrics['n_pred']}")
    print(f"main_metric:   {metrics['qualified_pct']:.6f}")
    print(f"tie_break:     {metrics['mean_score']:.6f}")


if __name__ == "__main__":
    # Smoke test: проверяем что данные на месте + evaluate работает
    print(f"SUBSET_VIDEO: {SUBSET_VIDEO} (exists={SUBSET_VIDEO.exists()})")
    print(f"SUBSET_GT:    {SUBSET_GT} (exists={SUBSET_GT.exists()})")
    if SUBSET_GT.exists():
        df = pd.read_csv(SUBSET_GT, encoding="utf-8")
        print(f"GT rows: {len(df)}")
    # try evaluate on existing v10 pred (sanity)
    v10_pred = ROOT / "outputs" / "pipeline_v10" / "26_12-20.csv"
    if v10_pred.exists():
        print(f"\nSanity: eval on existing pipeline_v10 pred...")
        m = evaluate_pipeline(v10_pred)
        print_summary(m)
