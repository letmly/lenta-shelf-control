"""
Pydantic schema для Lenta Tech price tag pipeline.

Соответствует SPEC §7: 29 полей CSV, каждая строка = один уникальный обнаруженный
ценник на видео.

Семантика пропусков (КРИТИЧНО для метрики):
    - Поле НЕТ на ценнике        → значение "нет"
    - Поле есть, но НЕ РАСПОЗНАНО → пустая строка ""

Три основных класса:
    VLMExtraction    - structured output для PaddleOCR-VL-1.5 / Qwen3-VL retry
    QRPayload        - парсинг JSON из QR-кода (поддержка обоих именований полей)
    PriceTagRecord   - финальная запись с 29 полями для CSV

Pydantic 2.13+, Python 3.11+.
"""

from __future__ import annotations

import csv
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


# ============================================================================
# Constants
# ============================================================================

NOT_PRESENT: Literal["нет"] = "нет"
"""Sentinel: поля нет на ценнике."""

NOT_RECOGNIZED: Literal[""] = ""
"""Sentinel: поле есть, но не распознано."""

# Всего распознаваемых полей (без служебных filename/bbox/timestamp).
# Используется для расчёта recognition_accuracy.
TOTAL_EXTRACTABLE_FIELDS: int = 23
"""12 текстовых с ценника + 11 из QR."""


# ============================================================================
# Classic OCR output (PP-OCRv5 East-Slavic)
# ============================================================================

class OCRLine(BaseModel):
    """
    Одна распознанная текстовая строка от PP-OCRv5 East-Slavic.
    Это plain OCR output — без структуры, без понимания ролей полей.

    PP-OCRv5 East-Slavic (Apache 2.0, ~5M params, RKNN-friendly) — это classic
    OCR pipeline: DBNet++ detector + SVTR/CRNN recognizer. Возвращает плоский
    список (bbox, text, confidence). НЕ VLM.

    Из списка OCRLine pipeline собирает VLMExtraction через rule-based parser
    (regex для цен/EAN-13, spatial heuristics для определения какая цена
    `price_default` vs `price_card`).

    PaddleOCR-VL-1.5 — это ОТДЕЛЬНАЯ модель, полноценный VLM (NaViT + ERNIE-4.5,
    0.9B params), он возвращает VLMExtraction напрямую, без OCRLine как
    промежуточного представления.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., description="Распознанный текст строки")
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int] = Field(
        ..., description="(x_min, y_min, x_max, y_max) относительно crop'а ценника"
    )


# ============================================================================
# Enums
# ============================================================================

class TagColor(StrEnum):
    """
    Цвет ценника. Lenta использует категориальные цвета для типов выкладки.
    Определяется HSV-классификацией dominant color crop'а.
    """
    WHITE = "white"        # стандартный
    YELLOW = "yellow"      # акция
    RED = "red"            # скидка / распродажа
    GREEN = "green"        # промо / органика
    ORANGE = "orange"      # сезонное
    BLUE = "blue"
    PINK = "pink"
    OTHER = "other"


# ============================================================================
# Regex patterns
# ============================================================================

PRICE_RE = re.compile(r"^\d{1,7}([.,]\d{1,2})?$")
EAN13_RE = re.compile(r"^\d{13}$")
DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(:\d{2})?$")
SKU_RE = re.compile(r"^[A-Za-z0-9\-_/]+$")


# ============================================================================
# Field status helpers
# ============================================================================

def is_empty(v: Any) -> bool:
    """True если поле не распознано (пусто)."""
    return v == "" or v is None


def is_not_present(v: Any) -> bool:
    """True если поля нет на ценнике."""
    return v == NOT_PRESENT


def is_meaningful(v: Any) -> bool:
    """True если поле имеет реальное распознанное значение."""
    return not is_empty(v) and not is_not_present(v)


def normalize_price(v: str) -> str:
    """Нормализация цены: '29,99' → '29.99', обрезка пробелов."""
    return v.replace(",", ".").strip()


def parse_price(v: Any) -> float | None:
    """Безопасный парсинг цены. None если не получается распарсить."""
    if not is_meaningful(v):
        return None
    try:
        return float(normalize_price(str(v)))
    except (ValueError, TypeError):
        return None


def validate_ean13_checksum(barcode: str) -> bool:
    """Проверка контрольной суммы EAN-13 (последняя цифра)."""
    if not EAN13_RE.match(barcode):
        return False
    digits = [int(d) for d in barcode]
    checksum_sum = sum(
        digits[i] * (1 if i % 2 == 0 else 3) for i in range(12)
    )
    expected_check = (10 - checksum_sum % 10) % 10
    return digits[12] == expected_check


# ============================================================================
# VLM structured output
# ============================================================================

class VLMExtraction(BaseModel):
    """
    JSON schema для structured output PaddleOCR-VL-1.5 / Qwen3-VL-8B retry.
    Передаётся в vLLM через XGrammar `guided_json` (требует xgrammar >= 0.1.32).

    Содержит только OCR-извлекаемые поля. Служебные (filename, bbox, timestamp)
    добавляются в pipeline post-processing.

    Семантика значений:
        ""        — VLM не смог распознать поле
        "нет"     — VLM уверен что поля нет на ценнике
        "<value>" — распознанное значение
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    product_name: str = Field(default="", description="Наименование товара")
    price_default: str = Field(default="", description="Цена без карты")
    price_card: str = Field(default="", description="Цена по карте лояльности")
    price_discount: str = Field(default="", description="Цена по акции")
    barcode: str = Field(default="", description="Штрихкод EAN-13 на ценнике")
    discount_amount: str = Field(default="", description="Размер скидки (% или ₽)")
    id_sku: str = Field(default="", description="Артикул / SKU")
    print_datetime: str = Field(default="", description="Дата и время печати")
    code: str = Field(default="", description="Код зоны выкладки")
    additional_info: str = Field(default="", description="Дополнительная информация")
    special_symbols: str = Field(default="", description="Тип выкладки / спец-символы")

    @field_validator(
        "price_default", "price_card", "price_discount",
        mode="after",
    )
    @classmethod
    def normalize_prices(cls, v: str) -> str:
        return normalize_price(v) if is_meaningful(v) else v


# ============================================================================
# QR code payload
# ============================================================================

class QRPayload(BaseModel):
    """
    Структура JSON внутри QR-кода ценника Ленты.

    Поддерживает оба именования полей:
        полное (barcode, price1, wholesaleLevel1Count, ...)
        короткое (b, p1, wL1C, ...)

    Создаётся из распарсенного JSON-содержимого QR (zxing-cpp + json.loads).
    Если QR не декодировался — экземпляр НЕ создаётся, передаётся None в pipeline.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="ignore",  # игнорируем неизвестные поля (timestamp, signature и т.п.)
    )

    barcode: str | None = Field(
        default=None,
        validation_alias=AliasChoices("barcode", "b"),
    )
    price1: float | None = Field(
        default=None,
        validation_alias=AliasChoices("price1", "p1"),
    )
    price2: float | None = Field(
        default=None,
        validation_alias=AliasChoices("price2", "p2"),
    )
    price3: float | None = Field(
        default=None,
        validation_alias=AliasChoices("price3", "p3"),
    )
    price4: float | None = Field(
        default=None,
        validation_alias=AliasChoices("price4", "p4"),
    )
    wholesale_level_1_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices("wholesaleLevel1Count", "wL1C"),
    )
    wholesale_level_1_price: float | None = Field(
        default=None,
        validation_alias=AliasChoices("wholesaleLevel1Price", "wL1P"),
    )
    wholesale_level_2_count: int | None = Field(
        default=None,
        validation_alias=AliasChoices("wholesaleLevel2Count", "wL2C"),
    )
    wholesale_level_2_price: float | None = Field(
        default=None,
        validation_alias=AliasChoices("wholesaleLevel2Price", "wL2P"),
    )
    action_price: float | None = Field(
        default=None,
        validation_alias=AliasChoices("actionPrice", "aP"),
    )
    action_code: str | None = Field(
        default=None,
        validation_alias=AliasChoices("actionCode", "aC"),
    )


# ============================================================================
# Main record (29 полей CSV)
# ============================================================================

class PriceTagRecord(BaseModel):
    """
    Один распознанный ценник = одна строка выходного CSV.

    Все поля распределены на 3 категории:
        1. Служебные (filename, frame_timestamp, x_min/y_min/x_max/y_max) — 6 шт
        2. Распознаваемые с ценника (OCR/VLM + color) — 12 шт
        3. Распознаваемые из QR — 11 шт
        Итого: 29 полей.

    Семантика всех нелужебных строковых полей:
        ""    — не распознано
        "нет" — нет на ценнике
        иначе — распознанное значение
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
        validate_assignment=True,
    )

    # ===== Group 1a: служебные (6 полей) =====

    filename: str = Field(..., description="Имя видеофайла")
    frame_timestamp: int = Field(
        ..., ge=0, description="Время в миллисекундах от начала видео"
    )
    x_min: int = Field(..., ge=0, description="Bbox: левый верх по горизонтали (px)")
    y_min: int = Field(..., ge=0, description="Bbox: левый верх по вертикали (px)")
    x_max: int = Field(..., ge=0, description="Bbox: правый низ по горизонтали (px)")
    y_max: int = Field(..., ge=0, description="Bbox: правый низ по вертикали (px)")

    # ===== Group 1b: данные с ценника, OCR/VLM (12 полей) =====

    product_name: str = Field(default="", description="Наименование товара")
    price_default: str = Field(default="", description="Цена без карты")
    price_card: str = Field(default="", description="Цена по карте")
    price_discount: str = Field(default="", description="Цена по акции")
    barcode: str = Field(default="", description="Штрихкод (EAN-13)")
    discount_amount: str = Field(default="", description="Размер скидки")
    id_sku: str = Field(default="", description="Артикул / SKU")
    print_datetime: str = Field(default="", description="Дата и время печати")
    code: str = Field(default="", description="Код зоны выкладки")
    additional_info: str = Field(default="", description="Дополнительная информация")
    color: str = Field(default="", description="Цвет ценника (HSV classifier)")
    special_symbols: str = Field(default="", description="Тип выкладки")

    # ===== Group 2: данные из QR-кода (11 полей) =====

    qr_code_barcode: str = Field(default="", description="QR: barcode / b")
    price1_qr: str = Field(default="", description="QR: price1 / p1")
    price2_qr: str = Field(default="", description="QR: price2 / p2")
    price3_qr: str = Field(default="", description="QR: price3 / p3")
    price4_qr: str = Field(default="", description="QR: price4 / p4")
    wholesale_level_1_count: str = Field(
        default="", description="QR: wholesaleLevel1Count / wL1C"
    )
    wholesale_level_1_price: str = Field(
        default="", description="QR: wholesaleLevel1Price / wL1P"
    )
    wholesale_level_2_count: str = Field(
        default="", description="QR: wholesaleLevel2Count / wL2C"
    )
    wholesale_level_2_price: str = Field(
        default="", description="QR: wholesaleLevel2Price / wL2P"
    )
    action_price_qr: str = Field(default="", description="QR: actionPrice / aP")
    action_code_qr: str = Field(default="", description="QR: actionCode / aC")

    # ===== Field-level validators =====

    @field_validator(
        "price_default", "price_card", "price_discount",
        "price1_qr", "price2_qr", "price3_qr", "price4_qr",
        "wholesale_level_1_price", "wholesale_level_2_price",
        "action_price_qr",
        mode="after",
    )
    @classmethod
    def _normalize_prices(cls, v: str) -> str:
        """Все ценовые поля: ',' → '.', обрезка пробелов."""
        return normalize_price(v) if is_meaningful(v) else v

    # ===== Cross-field validators =====

    @model_validator(mode="after")
    def _validate_bbox(self) -> PriceTagRecord:
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max ({self.x_max}) must be > x_min ({self.x_min})"
            )
        if self.y_max <= self.y_min:
            raise ValueError(
                f"y_max ({self.y_max}) must be > y_min ({self.y_min})"
            )
        return self

    # ===== Computed properties =====

    @computed_field  # type: ignore[prop-decorator]
    @property
    def barcode_checksum_valid(self) -> bool | None:
        """True если EAN-13 checksum валидный. None если поле не распознано / нет."""
        if is_meaningful(self.barcode):
            return validate_ean13_checksum(self.barcode)
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_default_value(self) -> float | None:
        return parse_price(self.price_default)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_card_value(self) -> float | None:
        return parse_price(self.price_card)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def price_discount_value(self) -> float | None:
        return parse_price(self.price_discount)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def prices_consistent(self) -> bool | None:
        """
        Бизнес-проверка: price_card ≤ price_default.
        None если хотя бы одна из цен не распарсилась.
        """
        d = self.price_default_value
        c = self.price_card_value
        if d is None or c is None:
            return None
        return c <= d

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recognized_fields_count(self) -> int:
        """
        Сколько полей реально несут информацию (распознано ИЛИ явно "нет").
        Используется для расчёта метрики хака.
        Служебные поля (filename, bbox, timestamp) не считаются.
        """
        extractable = [
            # 12 с ценника
            self.product_name, self.price_default, self.price_card,
            self.price_discount, self.barcode, self.discount_amount,
            self.id_sku, self.print_datetime, self.code,
            self.additional_info, self.color, self.special_symbols,
            # 11 из QR
            self.qr_code_barcode, self.price1_qr, self.price2_qr,
            self.price3_qr, self.price4_qr,
            self.wholesale_level_1_count, self.wholesale_level_1_price,
            self.wholesale_level_2_count, self.wholesale_level_2_price,
            self.action_price_qr, self.action_code_qr,
        ]
        return sum(
            1 for f in extractable if is_meaningful(f) or is_not_present(f)
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def recognition_completeness(self) -> float:
        """
        Доля заполненных полей (распознано + явно 'нет') от 23 извлекаемых.

        ВАЖНО: это НЕ та метрика, по которой жюри оценивают (там сверяется с
        ground truth). Это approximate complement для дебага в Langfuse.
        """
        return self.recognized_fields_count / TOTAL_EXTRACTABLE_FIELDS

    # ===== Сериализация =====

    def to_csv_dict(self) -> dict[str, str]:
        """
        Сериализация в dict под CSV-writer.
        Порядок ключей соответствует CSV_COLUMNS (SPEC §7).
        Все значения — строки (включая int-поля).
        """
        return {col: str(getattr(self, col)) for col in CSV_COLUMNS}


# ============================================================================
# CSV column order (SPEC §7)
# ============================================================================

CSV_COLUMNS: list[str] = [
    # Group 1: данные с ценника (18 полей)
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    # Group 2: QR-код (11 полей)
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]

assert len(CSV_COLUMNS) == 29, f"Expected 29 columns, got {len(CSV_COLUMNS)}"


# ============================================================================
# Builder helpers
# ============================================================================

def _qr_field_value(value: Any) -> str:
    """
    Helper для конвертации QR-поля Python-значения → строка CSV.
        None  → "нет"  (QR декодировался, но поля в JSON не было)
        else  → str(value)
    """
    if value is None:
        return NOT_PRESENT
    return str(value)


def build_record(
    *,
    filename: str,
    bbox: tuple[int, int, int, int],
    frame_timestamp_ms: int,
    vlm: VLMExtraction,
    color: str,
    qr: QRPayload | None = None,
) -> PriceTagRecord:
    """
    Финальная сборка PriceTagRecord из частей pipeline.

    Args:
        filename: имя видеофайла
        bbox: (x_min, y_min, x_max, y_max) в пикселях
        frame_timestamp_ms: время в мс от начала видео
        vlm: результат PaddleOCR-VL-1.5 / Qwen3-VL retry (structured JSON)
        color: HSV-классифицированный цвет ('white', 'yellow', ...)
        qr: распарсенный QRPayload или None если QR не декодирован

    QR-семантика:
        qr is None         → все QR-поля становятся "" (не распознано — QR не считан)
        qr.<field> is None → "нет" (QR декодирован, но поля в JSON нет)
        qr.<field> = value → str(value)
    """
    x_min, y_min, x_max, y_max = bbox

    if qr is None:
        # QR не считан — все 11 полей "" (пусто)
        qr_fields: dict[str, str] = {
            "qr_code_barcode": NOT_RECOGNIZED,
            "price1_qr": NOT_RECOGNIZED,
            "price2_qr": NOT_RECOGNIZED,
            "price3_qr": NOT_RECOGNIZED,
            "price4_qr": NOT_RECOGNIZED,
            "wholesale_level_1_count": NOT_RECOGNIZED,
            "wholesale_level_1_price": NOT_RECOGNIZED,
            "wholesale_level_2_count": NOT_RECOGNIZED,
            "wholesale_level_2_price": NOT_RECOGNIZED,
            "action_price_qr": NOT_RECOGNIZED,
            "action_code_qr": NOT_RECOGNIZED,
        }
    else:
        # QR считан — отсутствующие поля → "нет"
        qr_fields = {
            "qr_code_barcode": _qr_field_value(qr.barcode),
            "price1_qr": _qr_field_value(qr.price1),
            "price2_qr": _qr_field_value(qr.price2),
            "price3_qr": _qr_field_value(qr.price3),
            "price4_qr": _qr_field_value(qr.price4),
            "wholesale_level_1_count": _qr_field_value(qr.wholesale_level_1_count),
            "wholesale_level_1_price": _qr_field_value(qr.wholesale_level_1_price),
            "wholesale_level_2_count": _qr_field_value(qr.wholesale_level_2_count),
            "wholesale_level_2_price": _qr_field_value(qr.wholesale_level_2_price),
            "action_price_qr": _qr_field_value(qr.action_price),
            "action_code_qr": _qr_field_value(qr.action_code),
        }

    return PriceTagRecord(
        filename=filename,
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        frame_timestamp=frame_timestamp_ms,
        color=color,
        # из VLM extraction
        product_name=vlm.product_name,
        price_default=vlm.price_default,
        price_card=vlm.price_card,
        price_discount=vlm.price_discount,
        barcode=vlm.barcode,
        discount_amount=vlm.discount_amount,
        id_sku=vlm.id_sku,
        print_datetime=vlm.print_datetime,
        code=vlm.code,
        additional_info=vlm.additional_info,
        special_symbols=vlm.special_symbols,
        # из QR
        **qr_fields,
    )


# ============================================================================
# CSV writer
# ============================================================================

def write_csv(records: list[PriceTagRecord], output_path: str | Path) -> None:
    """
    Запись records в CSV согласно SPEC §7.
    Используется кодировка utf-8 с BOM для совместимости с Excel.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_csv_dict())


# ============================================================================
# Example usage / smoke test
# ============================================================================

if __name__ == "__main__":
    # Пример 1: ценник с распознанным QR и всеми полями
    vlm_full = VLMExtraction(
        product_name="Молоко Простоквашино 3.2% 930мл",
        price_default="89,99",
        price_card="74.99",
        price_discount="нет",  # на ценнике нет акционной цены
        barcode="4607034567892",
        discount_amount="нет",
        id_sku="3829461",
        print_datetime="2026-05-10 08:15",
        code="A12-03",
        additional_info="нет",
        special_symbols="нет",
    )

    qr_full = QRPayload.model_validate({
        "barcode": "4607034567892",
        "p1": 89.99,
        "p2": 74.99,
        "actionPrice": None,  # явно нет в QR
        "wholesaleLevel1Count": 6,
        "wholesaleLevel1Price": 70.00,
    })

    record1 = build_record(
        filename="aisle_dairy_01.mp4",
        bbox=(412, 580, 614, 730),
        frame_timestamp_ms=12_450,
        vlm=vlm_full,
        color=TagColor.WHITE.value,
        qr=qr_full,
    )

    print("=== Record 1 ===")
    print(f"  EAN-13 checksum valid: {record1.barcode_checksum_valid}")
    print(f"  Prices consistent (card ≤ default): {record1.prices_consistent}")
    print(f"  Recognized fields: {record1.recognized_fields_count} / {TOTAL_EXTRACTABLE_FIELDS}")
    print(f"  Completeness: {record1.recognition_completeness:.1%}")

    # Пример 2: ценник без QR (только OCR-распознавание)
    vlm_partial = VLMExtraction(
        product_name="Мёд цветочный 250г",
        price_default="320",
        price_card="",  # не распознано
        # остальные поля по дефолту ""
    )

    record2 = build_record(
        filename="aisle_honey_02.mp4",
        bbox=(120, 340, 285, 472),
        frame_timestamp_ms=34_120,
        vlm=vlm_partial,
        color=TagColor.YELLOW.value,
        qr=None,  # QR не декодирован
    )

    print("\n=== Record 2 ===")
    print(f"  Recognized fields: {record2.recognized_fields_count} / {TOTAL_EXTRACTABLE_FIELDS}")
    print(f"  QR fields all empty: {record2.qr_code_barcode == ''}")

    # Записываем в CSV
    out_path = Path("/tmp/sample_output.csv")
    write_csv([record1, record2], out_path)
    print(f"\nCSV written to: {out_path}")
    print(f"CSV columns count: {len(CSV_COLUMNS)}")
