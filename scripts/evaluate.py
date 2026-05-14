"""Метрика согласно ТЗ + чату оргов:

  1. Матчинг строк наш_CSV ↔ GT_CSV:
     a. ПЕРВИЧНЫЙ КЛЮЧ: barcode (если у нас непустой и совпал с GT — match)
     b. ВТОРИЧНЫЙ: spatial-temporal:
        - |frame_timestamp - GT_ts| <= 1500 мс  (допуск из чата орга)
        - IoU(bbox, GT_bbox) >= 0.3

  2. Для каждой матченной пары — сколько % полей распознано верно
     (с нормализацией: запятая→точка, lowercase, strip)

  3. Финальная метрика = доля GT-ценников с score >= 80%

Запуск:
    uv run python scripts/evaluate.py --pred outputs/pipeline_v1/43_15.csv \\
                                       --gt   data/labeled/43_15/43_15.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Поля для сравнения (КРОМЕ ключевых: frame_timestamp, bbox, filename)
FIELDS_TO_COMPARE = [
    "product_name", "price_default", "price_card", "price_discount",
    "barcode", "discount_amount", "id_sku", "print_datetime", "code",
    "additional_info", "color", "special_symbols",
    "qr_code_barcode", "price1_qr", "price2_qr", "price3_qr", "price4_qr",
    "wholesale_level_1_count", "wholesale_level_1_price",
    "wholesale_level_2_count", "wholesale_level_2_price",
    "action_price_qr", "action_code_qr",
]


def parse_ru(v):
    if pd.isna(v): return None
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def normalize(v) -> str:
    """Нормализация для сравнения (для точных полей: цены, штрихкоды, ts)."""
    if pd.isna(v): return ""
    s = str(v).strip().lower()
    s = s.replace(",", ".").replace(" ", "")
    return s


def normalize_text(v) -> str:
    """Нормализация для fuzzy match на текстовых полях (product_name, etc)."""
    if pd.isna(v): return ""
    s = str(v).strip().lower()
    # убираем пунктуацию, лишние пробелы
    import re
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Fuzzy threshold для текстовых полей (Levenshtein-like ratio)
FUZZY_FIELDS = {"product_name", "additional_info", "code", "special_symbols"}
FUZZY_THRESHOLD = 0.5  # 50% similarity = match — мягкий порог как у орга


def fuzzy_ratio(a: str, b: str) -> float:
    """Простой fuzzy ratio через python-Levenshtein. 0..1."""
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        # fallback: совпадение токенов
        ta = set(a.split()); tb = set(b.split())
        if not ta or not tb: return 0.0
        return len(ta & tb) / max(len(ta), len(tb))


def normalize_price(v) -> str:
    """Цена: '1199,99' / '1199.99' / '1199' → 1199.99 (если есть копейки, иначе 1199)."""
    if pd.isna(v): return ""
    s = str(v).strip().replace(",", ".")
    try:
        f = float(s)
        # если копеек нет или они 0, оставляем целое число
        if f == int(f): return str(int(f))
        return f"{f:.2f}"
    except ValueError:
        return s.lower()


def normalize_barcode_or_sku(v) -> str:
    """Штрихкод/SKU: только цифры."""
    if pd.isna(v): return ""
    s = str(v).strip()
    # уберём .0 в конце (pandas float)
    if s.endswith(".0"): s = s[:-2]
    # только цифры
    digits = "".join(c for c in s if c.isdigit())
    return digits


def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0


def compare_fields(pred_row, gt_row, fuzzy: bool = True):
    """% полей распознано верно с учётом fuzzy match для текстовых полей.

    fuzzy=True: для текстовых полей (product_name, code, ...) применяем
                fuzzy_ratio >= FUZZY_THRESHOLD.
                Для цен/штрихкодов — нормализованное равенство.
    """
    PRICE_FIELDS = {"price_default", "price_card", "price_discount",
                    "price1_qr", "price2_qr", "price3_qr", "price4_qr",
                    "wholesale_level_1_price", "wholesale_level_2_price",
                    "action_price_qr"}
    BARCODE_FIELDS = {"barcode", "id_sku", "qr_code_barcode"}

    correct = 0; total = 0
    per_field = {}
    for f in FIELDS_TO_COMPARE:
        gt_raw = gt_row.get(f, "")
        pred_raw = pred_row.get(f, "")
        # быстрая проверка на пустые
        if (pd.isna(gt_raw) or str(gt_raw).strip() == "") and \
           (pd.isna(pred_raw) or str(pred_raw).strip() == ""):
            continue

        # для цен нормализуем как числа
        if f in PRICE_FIELDS:
            gt_n = normalize_price(gt_raw); pred_n = normalize_price(pred_raw)
            match = gt_n == pred_n
        elif f in BARCODE_FIELDS:
            gt_n = normalize_barcode_or_sku(gt_raw); pred_n = normalize_barcode_or_sku(pred_raw)
            match = gt_n == pred_n
        elif fuzzy and f in FUZZY_FIELDS:
            gt_t = normalize_text(gt_raw); pred_t = normalize_text(pred_raw)
            ratio = fuzzy_ratio(gt_t, pred_t)
            match = ratio >= FUZZY_THRESHOLD
            gt_n = gt_t; pred_n = pred_t
        else:
            gt_n = normalize(gt_raw); pred_n = normalize(pred_raw)
            match = gt_n == pred_n

        # "нет" в pred и реальное значение в gt — НЕ match (мы написали "поля нет", а оно есть)
        if pred_n == "нет" and gt_n not in ("", "нет"):
            match = False
        if gt_n == "нет" and pred_n not in ("", "нет"):
            match = False

        total += 1
        if match: correct += 1
        per_field[f] = {"gt": gt_n, "pred": pred_n, "match": match}
    score = correct / total if total > 0 else 0
    return score, per_field


def match_rows(pred_df: pd.DataFrame, gt_df: pd.DataFrame):
    """Матчим строки pred к строкам gt.
    Возвращает list[(gt_idx, pred_idx or None, match_type)]
    """
    used_pred = set()
    matches = []
    # парсим bbox/ts из GT
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        gt_df[c] = gt_df[c].apply(parse_ru)
        pred_df[c] = pred_df[c].apply(parse_ru)
    gt_df["barcode_norm"] = gt_df["barcode"].apply(normalize)
    pred_df["barcode_norm"] = pred_df["barcode"].apply(normalize)

    # этап 1: по barcode
    for gt_idx, gt in gt_df.iterrows():
        if not gt["barcode_norm"]: continue
        for p_idx, p in pred_df.iterrows():
            if p_idx in used_pred: continue
            if p["barcode_norm"] and p["barcode_norm"] == gt["barcode_norm"]:
                matches.append((gt_idx, p_idx, "barcode"))
                used_pred.add(p_idx)
                break

    matched_gt = {m[0] for m in matches}

    # этап 2: spatial-temporal
    for gt_idx, gt in gt_df.iterrows():
        if gt_idx in matched_gt: continue
        if any(pd.isna(gt[c]) for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp")):
            continue
        gt_box = (gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"])
        gt_ts = gt["frame_timestamp"]
        best_p = None; best_score = 0
        for p_idx, p in pred_df.iterrows():
            if p_idx in used_pred: continue
            if any(pd.isna(p[c]) for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp")):
                continue
            dt = abs(gt_ts - p["frame_timestamp"])
            if dt > 1500: continue
            p_box = (p["x_min"], p["y_min"], p["x_max"], p["y_max"])
            v = iou(gt_box, p_box)
            if v >= 0.3 and v > best_score:
                best_score = v; best_p = p_idx
        if best_p is not None:
            matches.append((gt_idx, best_p, f"spatial-temporal (IoU={best_score:.2f})"))
            used_pred.add(best_p)
        else:
            matches.append((gt_idx, None, "NO MATCH"))
    return matches


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    args = p.parse_args()

    pred = pd.read_csv(args.pred, encoding="utf-8")
    gt = pd.read_csv(args.gt, encoding="utf-8")
    print(f"Pred rows: {len(pred)}  GT rows: {len(gt)}")

    matches = match_rows(pred, gt)

    matched = sum(1 for m in matches if m[1] is not None)
    no_match = sum(1 for m in matches if m[1] is None)
    print(f"\nGT total:                  {len(gt)}")
    print(f"GT matched to pred:        {matched}/{len(gt)}  ({matched/len(gt):.0%})")
    print(f"  by barcode:              {sum(1 for m in matches if m[2]=='barcode')}")
    print(f"  by spatial-temporal:     {sum(1 for m in matches if m[2].startswith('spatial'))}")
    print(f"GT NOT matched:            {no_match}")

    # для матченных — считаем score
    field_scores = []
    field_breakdown = {f: {"correct": 0, "total": 0} for f in FIELDS_TO_COMPARE}
    for gt_idx, p_idx, mtype in matches:
        if p_idx is None:
            field_scores.append(0.0); continue
        score, per_field = compare_fields(pred.iloc[p_idx], gt.iloc[gt_idx])
        field_scores.append(score)
        for f, info in per_field.items():
            field_breakdown[f]["total"] += 1
            if info["match"]: field_breakdown[f]["correct"] += 1

    qualified = sum(1 for s in field_scores if s >= 0.8)
    print(f"\nField-level score (matched only):")
    print(f"  Mean score (matched):    {sum(field_scores)/len(field_scores):.0%}" if field_scores else "0%")
    print(f"  >= 80% qualified:        {qualified}/{len(gt)}  ({qualified/len(gt):.0%})  <-- GLAVNAYA METRIKA")

    print(f"\nПо полям (среди матченных):")
    for f, st in field_breakdown.items():
        if st["total"] == 0: continue
        print(f"  {f:24s} {st['correct']:3d}/{st['total']:3d}  ({st['correct']/st['total']:.0%})")


if __name__ == "__main__":
    main()
