"""Пайплайн распознавания ценников.

Модули:
    detector       — YOLO детектор ценников
    tracker        — ByteTrack-обёртка, дедуп между кадрами
    qr             — декодер QR-кода + парсер payload
    color          — HSV-классификатор цвета ценника
    template       — классификатор шаблона ценника (1..7)
    ocr            — обёртка PaddleOCR
    parser         — маппинг OCR-боксов в CSV-поля по template ROI
    aggregator     — голосование по полям между кадрами одного track_id
    pipeline       — оркестратор: видео → DataFrame
"""
