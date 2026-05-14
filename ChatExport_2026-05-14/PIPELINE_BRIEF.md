# «Полка под контролем» — production pipeline brief

**Версия:** финальная
**Дата:** 12 мая 2026
**Заказчик:** ООО «Лента Тех»

---

## TL;DR

Edge-first production-решение для распознавания ценников с робота, развёрнутое в 6000+ магазинов Группы Лента. **Ноль AGPL-зависимостей**, весь stack под Apache 2.0 / MIT / BSD. Edge-инференс на Rockchip NPU отдаёт наверх только структурированный JSON (~50 КБ/ценник vs ~5 МБ/сек видео — снижение backhaul в 100,000×). Continuous learning loop без облачных API: pipeline сам себя дотюнивает на собранных данных через pseudo-labeling.

---

## Production-архитектура

```
┌──────────────────────────────────────────────────────────────────────┐
│  РОБОТ В ТОРГОВОМ ЗАЛЕ                                               │
│  Rockchip RK3576 (6 TOPS NPU, W4A16, Flash Attention, 16GB LPDDR5)  │
│                                                                       │
│  Camera ─► RGA preproc ─► Optical flow stop-detector                 │
│                            │                                          │
│                            ▼                                          │
│              ┌─────────────────────────┐                              │
│              │ D-FINE-S INT8 RKNN      │  Apache 2.0                  │
│              │ NPU core 0, 960²        │  48.5 mAP COCO               │
│              │ ~30-40 FPS              │                              │
│              └────────────┬────────────┘                              │
│                           │                                           │
│              ┌─────────────────────────┐                              │
│              │ ByteTrack (CPU)         │  MIT                         │
│              │ + best-frame selector   │                              │
│              └────────────┬────────────┘                              │
│                           │                                           │
│              ┌────────────┼────────────────────────┐                  │
│              ▼            ▼                        ▼                  │
│      ┌──────────────┐  ┌─────────────┐  ┌──────────────────────┐    │
│      │ PP-OCRv5     │  │ zxing-cpp   │  │ InternVL3.5-1B       │    │
│      │ East-Slavic  │  │ 3.0         │  │ W8A8 RKLLM           │    │
│      │ INT8 RKNN    │  │ (CPU)       │  │ NPU core 2           │    │
│      │ NPU core 1   │  │             │  │ async, для low-conf  │    │
│      │              │  │ QR + EAN +  │  │                      │    │
│      │ 81.6% RU     │  │ DataMatrix +│  │ 24 tok/s, 1.9GB RAM  │    │
│      │ ~25ms/crop   │  │ PDF417      │  │ MIT                  │    │
│      │ Apache 2.0   │  │ Apache 2.0  │  │                      │    │
│      └──────────────┘  └─────────────┘  └──────────────────────┘    │
│                           │                                           │
│                           ▼                                           │
│              Cross-frame voting + Pydantic validation                │
│                           │                                           │
│                           ▼                                           │
│              Structured JSON / 29 полей per ценник                   │
└──────────────────────────┬───────────────────────────────────────────┘
                           │ ~50 KB/ценник через MQTT/HTTPS
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  LENTA CLOUD CONTOUR (внутренний, без облачных API)                 │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ INGESTION                                                     │  │
│  │ FastAPI + Kafka (lenta-tech stack из стр. 6 PDF)             │  │
│  │ Topic: shelf.pricetags.raw                                    │  │
│  └────────┬─────────────────────────────────────────────────────┘  │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ QUALITY GATE                                                  │  │
│  │ Pydantic schema + business rules:                             │  │
│  │ • EAN-13 checksum                                             │  │
│  │ • price_card ≤ price_default                                  │  │
│  │ • datetime в окне печати                                      │  │
│  └────────┬───────────────────────────┬─────────────────────────┘  │
│           │ PASS                       │ FAIL → heavy retry        │
│           ▼                            ▼                            │
│  ┌──────────────┐    ┌─────────────────────────────────────────┐  │
│  │ Greenplum    │    │ HEAVY VLM RETRY (vLLM на A100)          │  │
│  │ + ClickHouse │    │ PaddleOCR-VL-1.5 + Qwen3-VL-8B fallback│  │
│  │ (их stack)   │    │ Apache 2.0, XGrammar guided JSON        │  │
│  └──────┬───────┘    │ XGrammar ≥0.1.32 (CVE-2026-25048!)      │  │
│         │            └────────────┬────────────────────────────┘  │
│         │                         │                                 │
│         │       ◄─────────────────┘ resolved JSON                   │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ DOWNSTREAM ИНТЕГРАЦИИ                                         │  │
│  │ • Автозаказ (mismatch detected_pricetag ↔ PIM)               │  │
│  │ • Планограмма compliance                                      │  │
│  │ • BIRD (поведение клиентов × выкладка)                       │  │
│  │ • Алёрты в корпоративный мессенджер                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ CONTINUOUS LEARNING (полный внутренний контур)               │  │
│  │ • Сырые crops с edge → S3-совместимое хранилище              │  │
│  │ • Grounded SAM 2 + Qwen3-VL-8B → автогенерация pseudo-labels │  │
│  │ • D-FINE-S и PP-OCRv5 auto-retrain раз в неделю на A100      │  │
│  │ • Langfuse v3 self-hosted: трейсы VLM, eval datasets         │  │
│  │ • Canary deploy: 1% магазинов → 10% → full rollout           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Production-стек

### Edge (на роботе)

| Компонент | Выбор | Лицензия | RKNN |
|---|---|---|---|
| Frame sampling | OpenCV optical flow | BSD | CPU |
| Детекция ценников | **D-FINE-S** | Apache 2.0 | ONNX→RKNN INT8 |
| Tracking | ByteTrack через `roboflow/trackers` | Apache 2.0 | CPU |
| OCR primary | **PP-OCRv5 East-Slavic** (det + rec) | Apache 2.0 | ✅ official airockchip recipe |
| QR + EAN + DataMatrix + PDF417 | **zxing-cpp 3.0** | Apache 2.0 | CPU |
| Цвет ценника | OpenCV HSV classifier | BSD | CPU |
| VLM resolver (low-conf retry) | **InternVL3.5-1B** через RKLLM | MIT | ✅ Qengineering port |
| Schema validation | Pydantic 2.13 | MIT | CPU |

### Server (Lenta Cloud, внутренний контур)

| Компонент | Выбор | Лицензия |
|---|---|---|
| Heavy retry OCR | **PaddleOCR-VL-1.5** (0.9B) | Apache 2.0 |
| Heavy retry VLM fallback | **Qwen3-VL-8B-Instruct** через vLLM 0.20.2 | Apache 2.0 |
| Structured JSON | XGrammar ≥ 0.1.32 (CVE-2026-25048!) | Apache 2.0 |
| Pseudo-labeling teacher (detection) | Grounding DINO 1.5 Edge + MM-Grounding-DINO ensemble | Apache 2.0 |
| Pseudo-labeling masks | Grounded SAM 2 | Apache 2.0 |
| Pseudo-labeling orchestration | Autodistill | Apache 2.0 |
| Pseudo-labeling labels полей | Qwen3-VL-8B + PaddleOCR-VL agreement filter | Apache 2.0 |
| Observability | Langfuse v3 self-hosted (MIT core, без /ee) | MIT |
| Ingestion | FastAPI + Kafka (нативный стек Lenta Tech) | MIT |
| Storage | Greenplum + ClickHouse (нативный стек Lenta Tech) | — |

### Деплой (для хакатона/демки)

| Компонент | Выбор | Лицензия |
|---|---|---|
| UI | Gradio 6.14.0 | Apache 2.0 |
| Public hosting | HF Spaces ZeroGPU xlarge (141 GB H200) | — |
| Fallback hosting | Modal serverless или RunPod | — |

---

## Ключевые архитектурные решения

**1. Edge-first против centralized.**
Robot считает локально, наверх летит структурированный JSON (~50 КБ/ценник), не видео (~5 МБ/сек). Это снимает то самое узкое место, из-за которого X5 в 2020 свернул централизованный pipeline для shelf monitoring — bandwidth и per-store compute dominated. Для 6000+ магазинов Ленты разница между viable и nonviable.

**2. Ноль AGPL-зависимостей.**
Весь production stack Apache 2.0 / MIT / BSD. D-FINE-S вместо YOLO11 как primary detector — ICLR 2025, 48.5 mAP на COCO (выше YOLO11s в 47.0) при сопоставимых FLOPS. Лента может shipать в магазины без legal review каждого релиза и без зависимости от лицензионных переговоров с Ultralytics Inc.

**3. RK3576, не RK3588.**
Robot v1 на RK3576 — у него W4A16 квантизация и Flash Attention в NPU, чего нет на RK3588. Та же 6 TOPS, но жирнее VLM влезает. Robot v2 (2027) — RK1828 SO-DIMM, 20 TOPS, для 7B VLM на борту без сервера.

**4. PP-OCRv5 на edge + PaddleOCR-VL-1.5 на сервере как retry.**
На edge нужен детерминизм и скорость — PP-OCRv5 East-Slavic закрывает 80%+ полей с 81.6% line-exact-match на русском. PaddleOCR-VL-1.5 на сервере подключается только когда edge не справился — 94.5% OmniDocBench v1.5, 92.05% на Real5-OmniDocBench (это ровно наш кейс: блики, стекло, motion blur).

**5. zxing-cpp 3.0 закрывает 11/29 полей бесплатно.**
QR ценника Ленты содержит 11 полей JSON. EAN-13 на штрихкоде, **DataMatrix Честного Знака** на молочке, **PDF417 EGAIS** на алкоголе — всё одна Apache-библиотека. Это 40% полей CSV без OCR вообще. У Ленты в видео есть и молочка, и алкоголь — все четыре формата кодов попадут.

**6. Continuous learning без внешних API.**
Grounded SAM 2 + Qwen3-VL-8B генерят pseudo-labels из новых crops внутри контура Ленты. D-FINE-S и PP-OCRv5 дообучаются раз в неделю на A100. Никакой ручной разметки никогда. Никакой зависимости от внешних сервисов разметки.

**7. Quality gate с business rules + heavy retry.**
EAN-13 checksum, price_card ≤ price_default, datetime sanity — отсекают 95% мусора до Greenplum. Failed records идут на heavy retry с PaddleOCR-VL-1.5 + Qwen3-VL-8B (vLLM + XGrammar guided JSON), не отбрасываются. Двухуровневая модель экономит compute: тяжёлые модели запускаются только когда лёгкие не справились.

**8. Canary rollout + Langfuse traces.**
Новая модель: 1% магазинов (≈60 точек) → 10% → full. Langfuse трейсит каждый VLM-вызов. Дашборд показывает регрессии по магазинам/категориям до раскатки на всю сеть.

---

## Что это даёт Ленте конкретно

| Метрика | Значение |
|---|---|
| Backhaul на ценник | ~50 KB JSON vs ~5 MB/сек видео = **снижение в 100,000×** |
| Hardware per-store | RK3576 board ~15k₽ vs серверная стойка ~500k₽+ |
| Latency детекции до алёрта сотруднику | секунды, не минуты |
| Privacy | Видео покупателей не покидает edge, наверх — только обезличенный JSON |
| Audit trail | Langfuse трейс per-ценник: почему именно так распознан |
| Стоимость разметки при редизайне ценников | 0₽ человеко-часов — auto-labeling |
| Coverage кодов одной библиотекой | QR + EAN-13 + DataMatrix (Честный Знак) + PDF417 (EGAIS) |
| AGPL exposure | 0 пакетов в production |
| Russian-OCR качество | 81.6% line-exact на edge + 92.05% Real5 на сервере |

---

## Hardware roadmap

| Поколение | Чип | NPU | LLM Quant | Когда |
|---|---|---|---|---|
| Robot v1 | **RK3576** | 6 TOPS | W4A16 + Flash Attention | Shipping с декабря 2025 |
| Robot v2 | **RK1828 SO-DIMM** | 20 TOPS | W4A16/W8A8, 3D-stacked DRAM | Released декабрь 2025, ~5-10× быстрее RK3588 на VLM |
| Robot v3 | RK3668 | 16 TOPS | TBD | Анонсирован 2025, ожидается 2027 |

С Robot v2 на борту бежит уже не InternVL3.5-1B, а 7B-класс VLM. Heavy retry на сервере становится только для адверсарных случаев, не для регулярного fallback.

---

## Соответствие критериям SPEC

| Критерий | Покрытие |
|---|---|
| Локальные модели, без облачных API | ✅ Весь стек локальный, включая Langfuse self-hosted |
| Открытые лицензии | ✅ Apache 2.0 / MIT / BSD везде, ноль AGPL |
| Без ручной разметки | ✅ Grounded SAM 2 + Qwen3-VL labeler + agreement filter |
| Лёгкие модели приоритетны | ✅ D-FINE-S = 10M, PP-OCRv5 = 5M, InternVL3.5-1B на edge |
| Тяжёлые с обоснованием | ✅ PaddleOCR-VL-1.5 (0.9B) и Qwen3-VL-8B только в server-side retry на FAIL |
| Локально воспроизводимо | ✅ docker-compose + закреплённые версии |
| rknn (int8) bonus | ✅ Edge-стек целиком RKNN: D-FINE-S + PP-OCRv5 + InternVL3.5-1B |
| Ограниченные ресурсы | ✅ Весь edge-стек влезает в 6 TOPS NPU + 16 GB RAM |
| Публичный деплой без auth | ✅ HF Spaces ZeroGPU xlarge для хак-демки |
| Масштабируемость на разные стеллажи/ценники | ✅ Open-vocab pseudo-labeling + ensemble детекторов в continuous learning loop |

---

## Критичные version pin'ы (нельзя забыть)

| Пакет | Минимальная версия | Причина |
|---|---|---|
| `xgrammar` | **≥ 0.1.32** | CVE-2026-25048 (DoS через nested grammars) |
| `zxing-cpp` | **≥ 3.0.0** | DataMatrix (Честный Знак) + PDF417 (EGAIS) в одной либе |
| `vllm` | **0.20.2** | XGrammar default backend, mm-encoder-tp-mode для VLM perf |
| `pydantic` | **≥ 2.13** | pydantic-core merged в main репо |
| `gradio` | **6.14.0** | ZeroGPU H200 совместимость |
| `rknn-toolkit2` | **2.3.2** | Mixed precision + improved Norm ops |
| `rknn-llm` | **1.2.3** | InternVL3.5 + Qwen3-VL support |

---

## Явные rejection list (для презы — почему не использовали)

| Технология | Причина отказа |
|---|---|
| Ultralytics YOLO11/YOLO26 | **AGPL-3.0** — copyleft на network use неприемлем для Ленты |
| YOLO26 (любой режим) | Ultralytics issue #23753 — INT8 RKNN segfault, FP16 only |
| Qwen2.5-VL-3B на RKNN | rknn-llm issue #387 — OCR деградирует под W8A8 |
| Surya OCR | RAIL-M с $2M revenue cap — disqualified |
| GOT-OCR-2.0 | Research-only license |
| Chandra v2 | OpenRAIL-M |
| pyzbar / ZBar | Не покрывает DataMatrix и PDF417 — критично для нашего видео |
| Cotype VL (MTS AI) | Closed weights несмотря на лучший русский результат (0.649 MWS Bench) |
| Llama-3.2-Vision | 10% accuracy на русских цифрах (M.Video Habr benchmark) |
| Stable Diffusion 3.5 | Community License с $1M ARR cap |
| FLUX.2-klein 9B / Dev | Non-commercial |
| boxmot | AGPL-3.0 |
| MASA / SAMURAI | Datacenter GPU only или non-commercial |
| Облачные API (OpenAI/Anthropic/Yandex/GigaChat/Tinkoff) | Запрещены SPEC §4 |

---

## References

- **PaddleOCR-VL-1.5**: `PaddlePaddle/PaddleOCR-VL-1.5` на HF, arXiv 2601.21957 (Jan 2026)
- **D-FINE**: ICLR 2025, нативно в 🤗 Transformers
- **PP-OCRv5 East-Slavic**: `PaddlePaddle/eslav_PP-OCRv5_mobile_rec`
- **InternVL3.5-1B RKNN**: `Qengineering/InternVL3.5-1B-NPU`
- **zxing-cpp 3.0**: github.com/zxing-cpp/zxing-cpp
- **Grounded SAM 2**: github.com/IDEA-Research/Grounded-SAM-2
- **Autodistill**: github.com/autodistill/autodistill
- **rknn-toolkit2 v2.3.2**: github.com/airockchip/rknn-toolkit2
- **rknn-llm v1.2.3**: github.com/airockchip/rknn-llm
- **MWS Vision Bench** (только русскоязычный VLM бенчмарк): github.com/mts-ai/MWS-Vision-Bench
- **OpenFoodFacts price-tag-extractor** (related work, Qwen3-VL LoRA): `openfoodfacts/price-tag-extractor` на HF
- **Polyakov et al. 2022** (precedent на ценниках Ленты): Future Internet 14(3):88

---

*Brief составлен на основе ресёрча актуальных SOTA-моделей на май 2026, с учётом всех ограничений SPEC v2.0: только локально развёртываемые модели, open licenses без AGPL в production, без ручной разметки, RKNN-friendly где возможно.*
