# 10. Industry-baseline для shelf-monitoring CV

> Синтез ресерча: что делают **Bossa Nova / Trax / Simbe / Pensa / Cognex** в проде, и что из этого мы можем адаптировать на хакатоне.

---

## Главный месседж

🔥 **Фишай-объектив на shelf-роботах — это компромисс, никто из топов индустрии его не использует.**

Bossa Nova / Simbe / Tally / Pensa ставят **вертикальный массив pinhole-камер с узким FOV** + **LED-подсветку** для короткой экспозиции. Это **физически устраняет** motion blur и дисторсию, которые мы пытаемся бороть пост-обработкой.

> «Они не пытаются «вытащить QR из плохого кадра» — они с самого начала снимают так, чтобы кадр был хорошим.»

Bossa Nova декодирует штрихкоды **на полном разрешении прямо на камере, до сжатия**, а ML работает на ресайзнутых кадрах в облаке. То есть: на их роботе **decode тривиален**, потому что входное качество отличное.

Источники: [Bossa Nova 2020 architecture](https://www.therobotreport.com/bossa-nova-2020-inventory-robot-includes-sharper-vision-for-brick-and-mortar-retailers/), [Trax Image Recognition](https://traxretail.com/solutions/trax-image-recognition/).

**Что это значит для нас:** мы решаем заведомо более сложную задачу, чем индустрия. Реалистичный потолок метрики ниже, чем в их whitepapers.

---

## Бенчмарк open-source QR-декодеров (Dynamsoft, 536 изображений)

| Декодер | Success rate | Лицензия |
|---|---|---|
| Dynamsoft (closed) | **83.3 %** | проприетарная |
| **BoofCV / PyBoof** | **60.7 %** | **Apache 2.0** ✅ — лучший open-source |
| OpenCV WeChat QR | 48.9 % | Apache 2.0 ✅ |
| ZBar (`pyzbar`) | 38.9 % | MIT (libzbar LGPL) ✅ |
| ZXing-cpp | 31.9 % | Apache 2.0 ✅ |

➡️ **Простой win:** добавить **PyBoof** в ансамбль декодеров. Бесплатное улучшение на ~25 п.п. поверх pyzbar+zxing.

Источник: [Dynamsoft QR benchmark](https://www.dynamsoft.com/codepool/qr-code-reading-benchmark-and-comparison.html), [BoofCV QR performance](https://boofcv.org/index.php?title=Performance:QrCode).

---

## Lucky Imaging — как делать multi-frame stacking ПРАВИЛЬНО

Наш текущий подход (median по всем 34 кадрам) — наивный. Astrophotography сообщество решает ту же задачу десятилетиями, и подход известен под названием **Lucky Imaging**.

### Алгоритм

1. Собрать стек из 30-50 кадров одного объекта.
2. Посчитать **sharpness каждого** (`cv2.Laplacian().var()` или Tenengrad / FFT high-freq energy).
3. **Сортировать**, оставить top-10 % самых чётких (= 3-5 кадров из 30-50).
4. **Per-tag alignment** через ECC homography (не глобально!).
5. Weighted median / Wiener-stack этих топ-кадров.

Это даёт типично **4× прирост резкости** в астрономии (где условия съёмки похожие на наши: атмосфера = blur, телескоп едет = motion).

Источники: [Lucky imaging (wiki)](https://en.wikipedia.org/wiki/Lucky_imaging), [AMOS 2011 paper](https://amostech.com/TechnicalPapers/2011/Poster/ZHANG.pdf), [Sky&Telescope tool](https://skyandtelescope.org/astronomy-resources/astrophotography-tips/a-free-tool-for-lucky-imaging/).

**Implementation:** ~200 строк numpy + OpenCV, без сторонних зависимостей.

Готовые библиотеки (только для референса — все GPL, использовать в коммерции нельзя):
- [PlanetarySystemStacker](https://github.com/Rolf-Hempel/PlanetarySystemStacker) — GPL ❌
- [Siril](https://siril.org/) — GPL ❌

---

## Fisheye undistortion + perspective rectification

**Шаг 1:** глобальный fisheye dewarp через OpenCV:
- `cv2.fisheye.undistortImage` (Apache 2.0)
- `cv2.omnidir.undistortImage` (Apache 2.0)

⚠️ Нужна **калибровочная матрица камеры**. В идеале — снять шахматной доской. У нас её нет, но можно:
- Оценить интрики по точкам схода в видео (например, прямые края стеллажей)
- Использовать дефолтные параметры fisheye и подстраивать вручную

Источники: [OpenCV omnidir](https://docs.opencv.org/4.x/dd/d12/tutorial_omnidir_calib_main.html), [SimFIR ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Feng_SimFIR_A_Simple_Framework_for_Fisheye_Image_Rectification_with_Self-supervised_ICCV_2023_paper.pdf).

**Шаг 2:** per-tag quadrilateral fit + homography (как сканирование документа):
- **DocAligner** ([DocsaidLab](https://github.com/DocsaidLab/DocAligner)) — heatmap-based 4 corner regression, ONNX, Apache 2.0
- **DocTr++** ([fh2019ustc](https://github.com/fh2019ustc/DocTr-Plus)) — transformer-based unwarp, MIT
- Baseline: `cv2.approxPolyDP` после YOLO bbox

---

## Pretrained-модели для retail (permissive)

### Детекция полок/ценников

| Источник | Что | Лицензия |
|---|---|---|
| **SKU110K dataset** | 110k размеченных полок | MIT ✅ |
| Pretrained на SKU110K | YOLOX / RT-DETR / DETR | Apache 2.0 ✅ |
| `openfoodfacts/price-tag-detection` (HF) | YOLOv11x, 2.2k ценников Европы | **AGPL** ⚠️ |
| RP2K Retail | большой датасет | CC-BY-NC ❌ (нельзя коммерчески) |

### OCR

| Модель | Что | Лицензия |
|---|---|---|
| **PaddleOCR PP-OCRv5** | RU+EN, ONNX | Apache 2.0 ✅ |
| **RapidOCR** | PaddleOCR-форк, чистый ONNX | Apache 2.0 ✅ — лучшее для RKNN/NPU |
| **TrOCR** | трансформер для печатного текста | MIT ✅ — тяжёлый |
| **Donut** | end-to-end key-value: cropp → JSON {name, price, ...} | MIT ✅ — нужно дообучить |
| **Florence-2 base (230M)** | OCR + grounding + reasoning одним промптом | MIT ✅ — есть ONNX |
| Idefics3 (8B) | мощный VLM | Apache 2.0 ✅, но 8B параметров — не для CPU/NPU |

---

## Релевантные пейперы

1. **EgoQR (Meta, окт 2024)** — `https://arxiv.org/abs/2410.05497`. Ровно наш кейс: **fisheye + ego (роботизированный носитель) + small QR + motion blur**. +34 % над SOTA. Кода Meta не выложила, но идеи переносимы. **Must-read.**

2. **Fast Blur Removal for Wearable QR Scanners (UbiComp 2015)** — `https://dl.acm.org/doi/pdf/10.1145/2802083.2808390`. Классика, легко имплементируется.

3. **Robust QR deblurring via local min/max prior (2024)** — `https://link.springer.com/article/10.1007/s00371-024-03272-y`. Классический Wiener с gradient prior.

4. **Handling Motion Blur in Multi-Frame SR (CVPR 2015)** — `https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ma_Handling_Motion_Blur_2015_CVPR_paper.pdf`.

5. **HDR+ Burst Pipeline (Google, SIGGRAPH 2016)** — `https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/`. Базовый паттерн multi-frame fusion.

6. **AI-Powered Retail Shelf Monitoring (2025)** — `https://www.researchgate.net/publication/396863890_AI-Powered_Retail_Shelf_Monitoring_Using_Vision-OCR_Integration_for_Real-Time_Inventory_Management`.

---

## Финальный рекомендованный stack (всё Apache/MIT)

```
1. Fisheye undistort      cv2.fisheye / cv2.omnidir
2. Детекция ценников      YOLOX или RT-DETR, fine-tune на SKU110K + наши GT
3. Tracking               ByteTrack (MIT)
4. Lucky imaging          top-10% sharpness + ECC homography align + weighted median (~200 LOC)
5. Per-tag rectification  DocAligner (Apache 2.0) или approxPolyDP baseline
6. QR-decoder ensemble    PyBoof + OpenCV WeChat + zxing-cpp (берём первый успешный)
7. OCR primary            RapidOCR / PaddleOCR PP-OCRv5
8. OCR fallback           Donut (для трудных) / Florence-2 (для truly hard)
9. Опциональный deblur    Wiener с estimated kernel (Springer 2024 algo)
```

---

## Реалистичные ожидания метрики

| Сценарий | Ожидаемый decode rate |
|---|---|
| Single-frame open-source на наших HARD-ценниках | 20-40 % |
| Multi-frame Lucky imaging + Wiener | 55-70 % |
| Dynamsoft-tier proprietary | 80-85 % (для нас недоступно) |
| **Composite `QR-decode OR OCR-price-readback`** | **80-90 %** ← **наш target** |

> Композитная метрика — это когда строка считается успешной, если **либо** удалось декодировать штрихкод/QR, **либо** удалось OCR-ить ценник так, что bbox+timestamp+OCR-fields в допуске GT. Это и есть «барcode как первичный, spatial-temporal как вторичный» механика орга.

**Физически нечитаемых ценников ~15-25 %** (sub-50 px QR, угол >60°, перекрытия). Их даже Dynamsoft не возьмёт.

---

## Что меняется в нашем roadmap

| Старый план | Новый план |
|---|---|
| Naive median по 34 кадрам | **Lucky imaging:** top-10% sharpness + ECC homography |
| Только pyzbar + zxing | **Ensemble:** PyBoof + WeChat + zxing-cpp |
| Просто кропать bbox | Сначала **fisheye undistort**, потом **DocAligner rectify** |
| YOLOv8n (AGPL) | YOLOX / RT-DETR (Apache 2.0), fine-tune на SKU110K + наши GT |
| Простой OCR без понимания шаблона | **Donut** (key-value extraction) после дообучения на размеченных кропах |

Это уже не «пальцем в небо», а **известная индустриальная архитектура**, адаптированная под наши ограничения.
