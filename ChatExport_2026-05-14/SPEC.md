# Lenta Tech «Полка под контролем» — спецификация проекта

**Версия:** 1.0
**Дата составления:** 12 мая 2026
**Дедлайн submission:** 11 июня 2026
**Заказчик:** ООО «Лента Тех»
**Оператор хакатона:** Changellenge

---

## 1. Контекст и постановка задачи

Lenta Tech организует хакатон **Lenta Tech Life Hack** с задачей разработки CV-решения для автоматического распознавания ценников с видеопотока робота, движущегося вдоль торговых стеллажей в реальных условиях съёмки.

**Бизнес-контекст.** В 2025 году «Группа Лента» уже внедрила стационарные камеры для видеоаналитики наличия товара в 8 супермаркетах Москвы (категории: безалкогольные напитки, вода, молочная продукция), сэкономив до 40% рабочего времени сотрудников на выкладке. Следующий шаг — переход от стационарных камер к мобильным источникам данных через робота, движущегося вдоль стеллажей.

**Цель проекта.** Превратить видеопоток робота в структурированные данные (CSV) с полями ценника и QR-кода, через работающий публичный пользовательский интерфейс, с обоснованной возможностью масштабирования на реальные магазины (>6000 магазинов сети).

---

## 2. Deliverables (формальные требования)

| # | Требование | Обязательность |
|---|---|---|
| D1 | Pipeline-решение для распознавания ценников с видео | Обязательно |
| D2 | CSV согласно схеме (см. §7) | Обязательно |
| D3 | Работающий пользовательский интерфейс (загрузка видео + скачивание CSV + интерактивный просмотр) | Обязательно |
| D4 | GitHub репозиторий с полным исходным кодом | Обязательно |
| D5 | README.md с описанием архитектуры решения | Обязательно |
| D6 | Инструкция, достаточная для воспроизведения, по локальному развёртыванию | Обязательно |
| D7 | Презентация (структура — см. §13) | Обязательно |
| D8 | Публичная ссылка на задеплоенное решение без авторизации | **Бонус** (даёт преимущество при оценке) |

---

## 3. Метрика и критерии оценки

### 3.1. Основная метрика

> **Доля ценников с контрольного видео, точность распознавания для которых составляет не менее 80%, от общего числа ценников.**

Это **per-tag метрика**, не per-field. Каждый ценник — атомарный датапоинт.

**Алгоритм (наша интерпретация, требует подтверждения):**
1. Для каждого ценника считается `accuracy = (правильных полей) / (всего применимых полей)`
2. Если `accuracy ≥ 0.80` → ценник засчитан
3. Финальная метрика = `(засчитанных ценников) / (всех ценников на контрольном видео)`

### 3.2. Косвенные критерии оценки

| Критерий | Тип | Влияние |
|---|---|---|
| Корректность распознавания и максимум извлечённых полей | Прямой | Главный |
| CSV в понятном структурированном формате | Прямой | Обязательно |
| Публичный деплой без авторизации | Бонусный | Дополнительное преимущество |
| Разумная архитектура, локальный контур | Прямой | Качественная оценка |
| Масштабируемость на разные стеллажи и форматы ценников | Прямой | Качественная оценка |
| Работа в условиях ограниченных ресурсов | Бонусный | Дополнительный плюс |
| Оптимизация под rknn (int8) | Бонусный | Приветствуется |

---

## 4. Ограничения (hard constraints)

Нарушать нельзя ни под каким предлогом:

1. **Только локально развёрнутые модели и библиотеки.** Любые внешние онлайн-сервисы запрещены.
2. **Облачные API запрещены.** Это вышибает: OpenAI, Anthropic, Gemini, Yandex GPT API, GigaChat API, Google Vision, Tinkoff OCR API, любые managed SaaS-OCR.
3. **Ручная разметка данных запрещена.** Только pretrained / self-supervised / synthetic / pseudo-labeling.
4. **Модель должна быть относительно лёгкой.** Тяжёлые модели разрешены при технологическом обосновании их применимости.
5. **Pretrained модели только с открытыми лицензиями.**
6. **Решение локально воспроизводимо.** Скриптовые языки и чёткие инструкции запуска предпочтительны.

---

## 5. Открытые вопросы организаторам

Эти вопросы критично уточнить ДО старта разработки, иначе оптимизация идёт вслепую:

1. Что именно входит в формулу «точность распознавания ценника» — какие из 29 полей считаются, есть ли веса между полями?
2. С каким IoU bbox-координаты считаются корректными?
3. Какая точность для `frame_timestamp` (мс)?
4. Как штрафуются false positives — выданный ценник, которого нет в кадре?
5. Учитываются ли пропущенные ценники (missed detections) в знаменателе общей метрики?
6. Если на ценнике нет поля и я выдал пусто (а должен «нет») — это ошибка для метрики?
7. Где найти шаблоны ценников «Ленты» с расшифровкой (ссылка из стр. 8 PDF постановки)?
8. Контрольное видео — это отдельный hidden set или public-набор, на котором мы и тестируем?

---

## 6. Схема входных данных

**Видеозаписи с робота**, public-набор. Доступ по ссылке из приложения (стр. 10 PDF постановки).

**Зоны охвата:** алкогольная продукция, молочные товары, мёд, джемы, сиропы.

**Сценарии движения робота:**
- **С остановками** — робот фиксируется перед каждой полкой → стабильные кадры
- **Без остановок** — непрерывное движение вдоль стеллажа → motion blur

**Особенности съёмки:**
- Смешанное освещение (естественный + светильники магазина), блики, тени
- Ценники на разной высоте и под разными углами к камере
- Часть ценников частично перекрыта товаром или элементами полки
- В некоторых зонах съёмка ведётся сквозь стеклянное ограждение (актуально для алкоголя)

---

## 7. Схема выходных данных (CSV)

**29 полей.** Каждая строка = один уникальный обнаруженный ценник.

### 7.1. Семантика пропусков (критично)

- Поле НЕТ на ценнике → значение `"нет"`
- Поле есть, но НЕ РАСПОЗНАНО → значение `пусто` (empty string)

Это две разные семантики и они означают разное. Например, если QR-код декодировался успешно, но в нём отсутствует поле `wholesaleLevel1Price` — это `"нет"`. Если QR-код вообще не декодировался — все 11 QR-полей `пусто`.

### 7.2. Группа 1: Данные с ценника (18 полей)

| Поле | Тип | Описание | Источник распознавания |
|---|---|---|---|
| `filename` | str | Имя видеофайла | Служебное |
| `product_name` | str | Наименование товара | OCR/VLM |
| `price_default` | num | Цена без карты | OCR/VLM + regex |
| `price_card` | num | Цена по карте | OCR/VLM + regex |
| `price_discount` | num | Цена по акции | OCR/VLM + regex |
| `barcode` | str | Штрихкод (EAN-13) | zxing-cpp + OCR fallback |
| `discount_amount` | str | Размер скидки | OCR/VLM |
| `id_sku` | str | Артикул | OCR/VLM |
| `print_datetime` | str | Дата и время печати | OCR/VLM + datetime parse |
| `code` | str | Код зоны выкладки | OCR/VLM |
| `additional_info` | str | Дополнительная информация | OCR/VLM |
| `color` | str | Цвет ценника | HSV classifier |
| `special_symbols` | str | Тип выкладки | OCR/VLM |
| `frame_timestamp` | int (ms) | Время от начала видео | Служебное |
| `x_min` | int | Bbox: левый верх, горизонталь | Служебное |
| `y_min` | int | Bbox: левый верх, вертикаль | Служебное |
| `x_max` | int | Bbox: правый низ, горизонталь | Служебное |
| `y_max` | int | Bbox: правый низ, вертикаль | Служебное |

### 7.3. Группа 2: Данные из QR-кода (11 полей)

| Поле | Ключ в QR JSON | Тип |
|---|---|---|
| `qr_code_barcode` | `barcode` / `b` | str |
| `price1_qr` | `price1` / `p1` | num |
| `price2_qr` | `price2` / `p2` | num |
| `price3_qr` | `price3` / `p3` | num |
| `price4_qr` | `price4` / `p4` | num |
| `wholesale_level_1_count` | `wholesaleLevel1Count` / `wL1C` | int |
| `wholesale_level_1_price` | `wholesaleLevel1Price` / `wL1P` | num |
| `wholesale_level_2_count` | `wholesaleLevel2Count` / `wL2C` | int |
| `wholesale_level_2_price` | `wholesaleLevel2Price` / `wL2P` | num |
| `action_price_qr` | `actionPrice` / `aP` | num |
| `action_code_qr` | `actionCode` / `aC` | str |

QR декодируется через `zxing-cpp` + `opencv.wechat_qrcode` (fallback), затем парсится как JSON (с поддержкой UTF-8 и CP-1251 + BOM), поля маппятся в выходную схему.

---

## 8. Архитектура pipeline

```
video.mp4
   │
   ├─[1] Frame sampling
   │    Optical flow → stop detection
   │    Stopped → 1 кадр на остановку
   │    Moving → 2-3 fps sampling
   │
   ├─[2] Detection (per sampled frame)
   │    YOLO11x (openfoodfacts/price-tag-detection) primary
   │    YOLOv8s pretrained ensemble (опционально)
   │    Weighted Box Fusion на IoU=0.5 → boxes[]
   │
   ├─[3] Tracking
   │    ByteTrack по IoU + Kalman filter
   │    track_id ↔ накапливаемые frames per track
   │
   ├─[4] Best frame selection per track
   │    score = laplacian_variance × bbox_area × frontal_score
   │
   ├─[5] Параллельная экстракция per best-crop:
   │    ├─ QR decode (zxing-cpp → WeChat QR) → 11 QR-полей
   │    ├─ Barcode decode (zxing-cpp EAN-13) → barcode
   │    ├─ HSV dominant color → color
   │    └─ PaddleOCR-VL-1.5 с JSON schema prompt → 11 текстовых полей
   │
   ├─[6] Quality gate
   │    Если <70% полей заполнено OR low confidence
   │    → retry с Qwen3-VL-8B (тяжёлый fallback)
   │
   ├─[7] Cross-frame voting
   │    Per field: majority vote с Levenshtein-1 tolerance
   │    Low-agreement → пусто (не угадываем — conservative output)
   │
   ├─[8] Post-processing
   │    Regex parse цен и дат
   │    Pydantic validation
   │    Правила "нет" / пусто согласно §7.1
   │
   └─[9] CSV output
        Одна строка на уникальный track_id, 29 полей
```

---

## 9. Технологический стек

### 9.1. Серверная версия (для метрики и публичной демки)

| Слой | Технология | Лицензия | Заметка |
|---|---|---|---|
| Detection primary | `openfoodfacts/price-tag-detection` (YOLOv11x, ONNX) | AGPL-3.0 | 206 epochs на supermarket shelf data |
| Detection ensemble | YOLOv8s pretrained | AGPL-3.0 | Для recall на нестандартных ценниках |
| Tracking | ByteTrack через `supervision` | MIT | Saturated SOTA для slow-motion |
| QR primary | `zxing-cpp` | Apache 2.0 | Fastest open-source |
| QR fallback | OpenCV `wechat_qrcode` | Apache 2.0 | Робастен к blur, ~24× быстрее classical |
| Barcode | `zxing-cpp` EAN-13 | Apache 2.0 | |
| Color | OpenCV HSV dominant + classifier | BSD | 4 строки кода |
| OCR/VLM primary | PaddleOCR-VL-1.5 (0.9B) | Apache 2.0 | OmniDocBench 94.5%, кириллица |
| OCR/VLM fallback | Qwen3-VL-8B-Instruct + vLLM | Apache 2.0 | OCRBench 905 |
| OCR sanity-check | PaddleOCR PP-OCRv5 East-Slavic | Apache 2.0 | 81.6% line-exact на русском |
| UI | Gradio | Apache 2.0 | Нативно дружит с HF Spaces |
| Backend | FastAPI / Litestar | MIT | |
| Structured output | Pydantic + outlines | MIT | JSON schema enforcement |
| Observability | Langfuse (self-hosted) | MIT | VLM trace inspection |
| Деплой | HF Spaces + ZeroGPU (H200) | — | Public Space без auth |

### 9.2. Edge-версия (RKNN-демонстрация, бонус)

| Слой | Технология | Заметка |
|---|---|---|
| Detection | YOLO11s → RKNN INT8 | airockchip/rknn_model_zoo recipe |
| OCR | PP-OCRv5 det+rec → RKNN INT8 | airockchip/rknn_model_zoo recipe |
| VLM (опционально) | InternVL3.5-1B → RKLLM W8A8 | Qengineering port, 24 tok/s на RK3588 |
| Runtime | `rknn-runtime` + `rknn-toolkit-lite` | На физическом железе |
| Симулятор | `rknn-toolkit2` NPU simulator | Для замеров без железа |

### 9.3. Что НЕ берём и почему

| Технология | Причина отказа |
|---|---|
| Surya OCR | RAIL-M лицензия, требует платёж выше $2M ARR |
| Qwen2.5-VL-3B на RK3588 | OCR деградирует под W8A8 (Rockchip issue #387) |
| YOLO26 INT8 на RK3588 | TopK segfault при export (Ultralytics issue #23753) |
| TrOCR / Donut / GOT-OCR / DeepSeek-OCR | Autoregressive, не маппятся на RKNPU2 |
| Grounding DINO 1.6 / DINO-X Pro | API-only, нет локального деплоя |
| `boxmot` tracker library | AGPL-3.0, используем upstream MIT-варианты |
| Tesseract LSTM | ~18% CER на печатном русском, неконкурентно |
| EasyOCR | Медленнее PP-OCRv5, нет RKNN-порта |

---

## 10. Deployment strategy

### 10.1. Две версии pipeline в одном репозитории

- **`pipeline_server.py`** — serverside, на ZeroGPU, тяжёлые модели, метрика измеряется здесь
- **`pipeline_edge.py`** — RKNN-стек, замеряется через симулятор и/или физическое железо

### 10.2. HF Spaces deployment

- Public Space без авторизации
- ZeroGPU (H200) для VLM inference
- Gradio UI: upload видео → progress bar → preview с боксами → download CSV
- Интерактивная таблица: clickable row → показывает crop ценника + frame timestamp

### 10.3. Fallback по деплою

Если HF Spaces ZeroGPU не тянет PaddleOCR-VL-1.5:
- Modal serverless GPU (T4, $0.000164/sec) — публичный URL без auth
- Или RunPod Serverless (у команды есть опыт с инфрой под MontazhList)

### 10.4. Локальный запуск

```bash
docker-compose up                                           # поднимает весь стек
python pipeline_server.py path/to/video.mp4 --out result.csv
python edge/convert_to_rknn.py                              # генерация .rknn артефактов
python edge/benchmark_simulator.py                          # latency на NPU симуляторе
```

---

## 11. План работы (4 недели)

### Неделя 1 (12-18 мая): Baseline end-to-end

**Цель:** работающая цепочка video → CSV, измерен floor метрики.

| День | Задача |
|---|---|
| 1-2 | Скачать видео и шаблоны ценников, посмотреть глазами, отправить вопросы организаторам |
| 3-4 | Stage 1+2 (frame sampling + детекция через openfoodfacts) + Stage 5 QR/barcode |
| 5-7 | Stage 5 PaddleOCR-VL-1.5 + CSV writer + первый run метрики |

**Definition of Done:** `pipeline.py video.mp4 → out.csv` работает, floor метрики записан в Langfuse.

### Неделя 2 (19-25 мая): Tracking + voting (главный прирост метрики)

| День | Задача |
|---|---|
| 8-9 | ByteTrack + best frame selector |
| 10-11 | Cross-frame voting, confidence aggregation |
| 12-13 | YOLOv8s ensemble (опционально, замерить выигрыш) |
| 14 | Полный run, целевая метрика ≥60% |

**Definition of Done:** tracking pipeline стабилен, метрика выше baseline на >15 п.п.

### Неделя 3 (26 мая – 1 июня): UI, деплой, RKNN-трек

| День | Задача |
|---|---|
| 15-16 | Gradio UI с боксами и интерактивной таблицей |
| 17-18 | Deploy на HF Spaces + ZeroGPU |
| 19-20 | Prompt engineering для PaddleOCR-VL-1.5, quality gate с Qwen3-VL-8B |
| 21 | Финальный benchmark, целевая метрика ≥70-75% |

**Параллельно в фоне:**
- RKNN conversion: YOLO11s + PP-OCRv5 → INT8 через airockchip/rknn_model_zoo
- Замеры через RKNN NPU simulator, цифры идут в README

**Definition of Done:** публичная ссылка на UI работает, RKNN-артефакты в репо, latency-замеры на симуляторе документированы.

### Неделя 4 (2-11 июня): Презентация + полировка

| День | Задача |
|---|---|
| 22-24 | Подготовка презентации (9 слайдов) |
| 25-27 | Видео-демо (60-90 сек), финализация README, edge pipeline assembly |
| 28-30 | Резерв на баги, последние tweak'и, целевая метрика ≥75%+ |
| 31 (11 июня) | Submit |

---

## 12. Структура репозитория

```
lenta-shelf-vision/
├── README.md                       # архитектура + локальный запуск
├── SPEC.md                         # этот документ
├── pyproject.toml
├── docker-compose.yml
├── app.py                          # Gradio UI entrypoint
│
├── pipeline/
│   ├── __init__.py
│   ├── preprocessing.py            # optical flow, sampling, deblur
│   ├── detection.py                # YOLO + WBF ensemble
│   ├── tracking.py                 # ByteTrack, best frame selector
│   ├── extraction/
│   │   ├── qr.py                   # zxing-cpp + WeChat QR
│   │   ├── barcode.py              # EAN-13
│   │   ├── color.py                # HSV classifier
│   │   └── vlm.py                  # PaddleOCR-VL + Qwen3-VL fallback
│   ├── voting.py                   # cross-frame majority voting
│   ├── postprocess.py              # regex, Pydantic validation
│   └── schema.py                   # Pydantic-модели для 29 полей
│
├── prompts/
│   ├── paddleocr_vl_prompt.txt
│   └── qwen3_vl_fallback.txt
│
├── edge/
│   ├── convert_to_rknn.py          # YOLO11 + PP-OCRv5 → RKNN INT8
│   ├── pipeline_edge.py            # RKNN runtime stack
│   ├── benchmark_simulator.py      # NPU simulator latency
│   └── artifacts/
│       ├── yolo11s.rknn
│       ├── ppocrv5_det.rknn
│       └── ppocrv5_rec.rknn
│
├── eval/
│   ├── metric.py                   # доля ценников с accuracy ≥80%
│   └── benchmark.py                # runner с Langfuse-логированием
│
├── tests/
│   └── sample_videos/              # 3-5 коротких clip'ов для smoke tests
│
└── notebooks/
    └── exploration.ipynb           # глазная проверка crops, debug
```

---

## 13. Структура презентации (минимум 9 слайдов)

| # | Слайд | Ключевое содержание |
|---|---|---|
| 1 | Титул | Название решения + название команды |
| 2 | Команда | Имена участников, роли, контакты |
| 3 | Анализ задачи и данных | Особенности видео: стекло, перекрытия, два сценария движения. Pitfalls: дедупликация, «нет» vs пусто, метрика per-tag |
| 4 | Архитектура pipeline | Data flow diagram + логика по 9 этапам |
| 5 | Технологический стек | Таблица моделей + обоснования выбора |
| 6 | UI + примеры CSV | Скриншоты интерфейса + строки выходного CSV |
| 7 | Метрики | Финальная цифра + breakdown: QR закрывает X% полей, OCR Y%, VLM Z% |
| 8 | Edge-путь (RKNN) | Схема deploy, latency на NPU симуляторе, путь до прода на RK3588 |
| 9 | Ограничения + scaling | Урок X5 (bandwidth-dominant cost), масштаб на разные форматы стеллажей, ссылка на репо + ссылка на деплой |

---

## 14. Риски и митигации

| Риск | Вероятность | Митигация |
|---|---|---|
| Метрика интерпретируется иначе, чем мы оптимизируем | Высокая | Задать вопросы организаторам сразу (см. §5) |
| openfoodfacts детектор плохо ловит ценники Ленты | Средняя | YOLOv8s ensemble + опциональный fine-tune на pseudo-labels от Grounded SAM 2 |
| HF Spaces ZeroGPU не тянет PaddleOCR-VL-1.5 | Низкая | Fallback на Modal serverless GPU |
| RKNN conversion ломается на нашем графе | Средняя | Использовать только модели из airockchip/rknn_model_zoo с проверенными рецептами |
| Качество OCR на русском хуже ожидаемого | Средняя | Двойной OCR: PaddleOCR-VL + PP-OCRv5 sanity-check, agreement-фильтр |
| Стеклянные ограждения ломают детекцию (алкоголь) | Средняя | Domain-specific augmentation (рефлексы) + low-conf reject |
| Не успеваем презу за день до deadline | Средняя | Структура слайдов готова с недели 1, заполняем по мере появления данных |
| Команда не успевает по неделям | Средняя | Чёткие Definition of Done на каждой неделе + еженедельные синхронизации |

---

## 15. Команда и распределение ролей

*[Заполнить]*

| Участник | Роль | Зона ответственности | Контакт |
|---|---|---|---|
| | Tech Lead | Архитектура, integration | |
| | CV Engineer | Detection + tracking | |
| | ML Engineer | OCR / VLM | |
| | Backend | Pipeline, API, UI | |
| | DevOps | Deploy, RKNN-трек | |

---

## 16. Контакты организаторов

- Changellenge: `info@changellenge.com`
- vk.com/changellengeglobal
- ООО «Лента Тех»: `https://lenta.tech/`

---

## Приложение A: ключевые ссылки

- PDF постановки задачи: см. корень репо
- Видео-датасет: *[ссылка из стр. 10 PDF, получить у организаторов]*
- Шаблоны ценников Ленты с расшифровкой: *[ссылка из стр. 8 PDF, получить у организаторов]*
- Кейс на платформе Changellenge: *[ссылка]*

## Приложение B: референсы

- Polyakov et al. (2022), Future Internet 14(3):88 — YOLOv4-Tiny на ценниках Ленты, 96.92% cross-val accuracy
- `openfoodfacts/price-tag-detection` на HuggingFace — pretrained YOLOv11x
- `airockchip/rknn_model_zoo` — официальные RKNN-рецепты от Rockchip
- MWS Vision Bench (MTS AI) — единственный русскоязычный VLM бенчмарк
- X5 Tech, Setters Media 2020 — урок про bandwidth-dominant cost для retail CV
