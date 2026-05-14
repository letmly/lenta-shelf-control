"""Для матченных строк показать GT vs Pred, чтобы понять что парсер ломает."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import match_rows, FIELDS_TO_COMPARE, normalize

video = "26_12-20"
pred = pd.read_csv(ROOT / f"outputs/pipeline_v4/{video}.csv", encoding="utf-8")
gt = pd.read_csv(ROOT / f"data/labeled/{video}/{video}.csv", encoding="utf-8")
matches = match_rows(pred.copy(), gt.copy())

# Покажем только matched (не None)
matched = [(g, p, mt) for g, p, mt in matches if p is not None]
print(f"Matched: {len(matched)} pairs\n")
for i, (g_idx, p_idx, mtype) in enumerate(matched[:6]):
    g = gt.iloc[g_idx]; pr = pred.iloc[p_idx]
    print(f"\n{'='*80}")
    print(f"Pair {i+1}: GT idx={g_idx} <-> Pred idx={p_idx}  ({mtype})")
    print(f"  ts: gt={g['frame_timestamp']}  pred={pr['frame_timestamp']}")
    print(f"  bbox gt:  ({g['x_min']}, {g['y_min']}, {g['x_max']}, {g['y_max']})")
    print(f"  bbox pred:({pr['x_min']}, {pr['y_min']}, {pr['x_max']}, {pr['y_max']})")
    for f in ["product_name", "price_default", "price_card", "discount_amount",
              "barcode", "id_sku", "print_datetime", "color"]:
        g_norm = normalize(g.get(f, "")); p_norm = normalize(pr.get(f, ""))
        match = "OK  " if g_norm == p_norm and g_norm else ("--  " if not g_norm and not p_norm else "MISS")
        print(f"  [{match}] {f:18s}  gt={str(g.get(f, ''))[:40]!r:<45} pred={str(pr.get(f, ''))[:40]!r}")
