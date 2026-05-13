# 📋 Задачи (backlog для kanban-доски)

> Создать в Notion базу с полями: **Title, Status (Todo/Doing/Review/Done/Blocked), Day, Assignee, Priority (P0/P1/P2), Estimated h, Tags**. Группировка по Day, представления — Kanban + Timeline.

---

## Д1 — Setup и EDA (сегодня)

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T1.1 | Создать private repo + push v0 | PM | 1ч | P0 | infra |
| T1.2 | Загрузить видео в shared-storage, обновить data/README.md + fetch_data.py | PM | 1ч | P0 | infra |
| T1.3 | EDA: распределение bbox sizes, дубли по timestamp, разбор `filename` подпутей | ML | 3ч | P0 | data |
| T1.4 | Кропы по bbox → визуально подтвердить шаблоны 1..7 | ML | 2ч | P1 | data |
| T1.5 | Прогнать pretrained YOLOv8n на 1 кадре каждого видео → baseline recall | ML | 1ч | P0 | model |
| T1.6 | Скачать пример QR из размеченных ценников → проверить декодирование pyzbar | BE | 1ч | P0 | qr |
| T1.7 | Поднять Notion workspace + перенести задачи | PM | 1ч | P0 | infra |
| T1.8 | Прикинуть структуру презентации (оглавление слайдов) | PM | 1ч | P2 | pitch |

## Д2 — Детекция, QR, цвет

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T2.1 | Сконвертировать CSV-bbox в YOLO format (txt) | ML | 1ч | P0 | data |
| T2.2 | Fine-tune YOLOv8n на наших ценниках (epochs=50) | ML | 4ч | P0 | model |
| T2.3 | Реализовать pipeline/qr.py: decode_qr() через pyzbar | BE | 2ч | P0 | qr |
| T2.4 | Реализовать pipeline/color.py: HSV-классификатор | BE | 1ч | P1 | model |
| T2.5 | Стенд QR-success-rate: прогнать pyzbar на всех 154 GT-ценниках | BE | 2ч | P0 | qr, metric |
| T2.6 | UI-набросок: Gradio с фейковым CSV на выходе | FE | 2ч | P1 | ui |
| T2.7 | Подобрать draft-цветовую палитру и обложку презы | PM | 2ч | P2 | pitch |

## Д3 — Tracker, OCR, parser

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T3.1 | Подключить ByteTrack из ultralytics, прогон на видео | ML | 2ч | P0 | tracker |
| T3.2 | Лучший кадр трека (sharpness × area), сохраняем для OCR | ML | 1ч | P0 | tracker |
| T3.3 | Реализовать pipeline/ocr.py: PaddleOCR(ru) обёртка | BE | 2ч | P0 | ocr |
| T3.4 | Заполнить TEMPLATE_ROI для шаблонов 1..7 (по pptx) | BE | 3ч | P0 | parser |
| T3.5 | Реализовать parser.parse: ROI → CSV-поля | BE | 3ч | P0 | parser |
| T3.6 | Реализовать pipeline.process_video end-to-end (без агрегатора) | ML | 2ч | P0 | pipeline |
| T3.7 | UI: подключить process_video, streaming прогресс | FE | 2ч | P1 | ui |

## Д4 — End-to-end + метрика

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T4.1 | scripts/evaluate.py: матчинг по IoU+timestamp + per-field score | BE | 3ч | P0 | metric |
| T4.2 | Полный прогон на 43_15.mp4, сверка с GT | ML | 1ч | P0 | metric |
| T4.3 | Прогон на 25/26 12-20, агрегированная цифра | ML | 1ч | P0 | metric |
| T4.4 | Aggregator: голосование по полям | BE | 2ч | P1 | pipeline |
| T4.5 | Разбор ошибок: топ-10 «слабых» ценников | ML | 2ч | P0 | analysis |
| T4.6 | План улучшений на Д5 на основе разбора | ML | 1ч | P0 | plan |

## Д5 — Итерации + UI

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T5.1 | Улучшения OCR/parser по ошибкам Д4 (приоритет — цены, дата) | BE | 4ч | P0 | parser |
| T5.2 | Опц.: Super-Resolution перед QR, если H1 нужна | ML | 3ч | P1 | model |
| T5.3 | UI: полировка, drag-n-drop, прогрессбар, скачивание CSV | FE | 3ч | P0 | ui |
| T5.4 | Метрика на unlabeled видео (3 шт.) — sanity-check | ML | 2ч | P1 | metric |
| T5.5 | Draft презентации до раздела «архитектура» | PM | 3ч | P0 | pitch |

## Д6 — Деплой + презентация

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T6.1 | Dockerfile + HF Space repo | FE | 3ч | P0 | infra |
| T6.2 | Деплой Gradio на HF Spaces, smoke-тест | FE | 2ч | P0 | infra |
| T6.3 | Полная презентация со скринами UI + метриками | PM | 4ч | P0 | pitch |
| T6.4 | Записать demo-видео (если потребуется) | PM | 1ч | P1 | pitch |
| T6.5 | README — финальная полировка | PM | 1ч | P1 | docs |

## Д7 — Финал

| # | Title | Lead | Est | Priority | Tags |
|---|---|---|---|---|---|
| T7.1 | Репетиция питча | все | 2ч | P0 | pitch |
| T7.2 | Финальный прогон на контрольном видео | ML | 2ч | P0 | metric |
| T7.3 | Сабмит формы + презентация + ссылки | PM | 1ч | P0 | submit |

---

**Тэги для labels:** `infra, data, model, ocr, qr, parser, tracker, pipeline, metric, ui, pitch, docs, submit, analysis`
