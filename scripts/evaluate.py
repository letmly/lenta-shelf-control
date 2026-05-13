"""Метрика согласно ТЗ: доля ценников с ≥80 % правильно распознанных полей.

Шаги:
    1. Загрузить pred.csv и gt.csv.
    2. Матчить строки по IoU bbox + |Δ frame_timestamp| ≤ 1000 мс.
    3. Для каждой матченной пары — считать % полей verbatim/fuzzy match.
    4. Считать долю строк, у которых score ≥ 80%.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    args = parser.parse_args()

    raise SystemExit("Stub. Реализация в Д4.")


if __name__ == "__main__":
    main()
