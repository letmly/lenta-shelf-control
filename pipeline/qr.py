"""QR-декодер + парсер payload по полям ТЗ.

Ключи в QR (см. docs/01-task.md):
    barcode/b                 → qr_code_barcode
    price1/p1 .. price4/p4    → price{1..4}_qr
    wholesaleLevel1Count/wL1C → wholesale_level_1_count
    wholesaleLevel1Price/wL1P → wholesale_level_1_price
    wholesaleLevel2Count/wL2C → wholesale_level_2_count
    wholesaleLevel2Price/wL2P → wholesale_level_2_price
    actionPrice/aP            → action_price_qr
    actionCode/aC             → action_code_qr
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

KEY_ALIASES = {
    "barcode": "qr_code_barcode", "b": "qr_code_barcode",
    "price1": "price1_qr", "p1": "price1_qr",
    "price2": "price2_qr", "p2": "price2_qr",
    "price3": "price3_qr", "p3": "price3_qr",
    "price4": "price4_qr", "p4": "price4_qr",
    "wholesaleLevel1Count": "wholesale_level_1_count", "wL1C": "wholesale_level_1_count",
    "wholesaleLevel1Price": "wholesale_level_1_price", "wL1P": "wholesale_level_1_price",
    "wholesaleLevel2Count": "wholesale_level_2_count", "wL2C": "wholesale_level_2_count",
    "wholesaleLevel2Price": "wholesale_level_2_price", "wL2P": "wholesale_level_2_price",
    "actionPrice": "action_price_qr", "aP": "action_price_qr",
    "actionCode": "action_code_qr", "aC": "action_code_qr",
}


@dataclass
class QRPayload:
    fields: dict[str, str]


def decode_qr(crop) -> str | None:
    """Декодирует QR из изображения. Возвращает строку payload или None."""
    raise NotImplementedError("Д2: подключить pyzbar.pyzbar.decode")


def parse_payload(payload: str) -> QRPayload:
    """Парсит payload QR. Поддерживает JSON и k=v;k=v форматы."""
    fields: dict[str, str] = {}
    payload = payload.strip()
    # JSON?
    try:
        obj = json.loads(payload)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in KEY_ALIASES:
                    fields[KEY_ALIASES[k]] = str(v)
            return QRPayload(fields)
    except (json.JSONDecodeError, TypeError):
        pass
    # k=v;k=v или k:v|k:v
    for token in re.split(r"[;|&\n]+", payload):
        if "=" in token:
            k, v = token.split("=", 1)
        elif ":" in token:
            k, v = token.split(":", 1)
        else:
            continue
        k = k.strip()
        if k in KEY_ALIASES:
            fields[KEY_ALIASES[k]] = v.strip()
    return QRPayload(fields)
