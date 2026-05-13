# 👥 Команда и роли

> Распределение под 4-х человек (1 неделя). Каждый — owner своего блока, но кросс-помощь в обе стороны обязательна.

## Состав

| Роль | Имя | Зона ответственности | Owns | Контакт |
|---|---|---|---|---|
| **ML/CV lead** | TBD | Детектор, трекинг, эксперименты, EDA | `pipeline/detector.py`, `pipeline/tracker.py`, `notebooks/*` | |
| **ML/Backend** | TBD | OCR, парсер шаблонов, QR, метрика, оркестратор | `pipeline/ocr.py`, `pipeline/parser.py`, `pipeline/qr.py`, `scripts/evaluate.py` | |
| **Frontend/Deploy** | TBD | Gradio UI, демо, деплой, Docker | `ui/`, `Dockerfile`, HF Space repo | |
| **PM/Design** | TBD | Координация, презентация, нарратив, README, Notion-доска | `docs/`, презентация, Notion | |

## Распределение по дням

### Д1 (сегодня)
- ML lead → T1.3 (EDA), T1.4 (кропы по шаблонам), T1.5 (baseline YOLO)
- ML/Backend → T1.6 (тест pyzbar), помощь T1.3
- Frontend → знакомится с Gradio, шаблон UI
- PM → T1.1 (репо), T1.2 (видео в облако), T1.7 (Notion), T1.8 (план презы)

### Д2
- ML lead → T2.1 (CSV→YOLO), T2.2 (fine-tune)
- Backend → T2.3 (QR), T2.4 (color), T2.5 (QR success rate)
- Frontend → T2.6 (Gradio шаблон)
- PM → T2.7 (превью презы), пишет drafty docs/

### Д3
- ML lead → T3.1 (ByteTrack), T3.2 (best frame), T3.6 (pipeline)
- Backend → T3.3 (OCR), T3.4 (ROI), T3.5 (parser)
- Frontend → T3.7 (UI подключение)
- PM → продолжает презу

### Д4
- ML lead → T4.2, T4.3 (прогон), T4.5 (разбор ошибок)
- Backend → T4.1 (метрика), T4.4 (aggregator)
- Frontend → стабилизация UI
- PM → собирает первые цифры в презу

### Д5
- ML lead → T5.2 (если нужно), T5.4
- Backend → T5.1 (улучшения)
- Frontend → T5.3 (UI полировка)
- PM → T5.5 (драфт презы)

### Д6
- Frontend → T6.1, T6.2 (Docker + HF deploy)
- ML/Backend → подстраховка деплоя, fallback на Render
- PM → T6.3, T6.4, T6.5

### Д7
- Все → T7.1 (репетиция), T7.2 (финальный прогон), T7.3 (сабмит)

## Правила работы

1. **Ежедневный 15-мин синк в 19:00 МСК:** что сделал, что мешает, что завтра.
2. **Все merge через PR** (даже соло). Один член команды апрувит.
3. **Бранчинг:** `feat/<task-id>`, `fix/<task-id>`. Main защищён.
4. **Если задача буксует > 2× оценки** → эскалация в синк, переразбиваем или зовём напарника.
5. **Презентация** обновляется параллельно с кодом, не «в последний день».
