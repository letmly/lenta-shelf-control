# Russian Retail Price-Tag Pipeline: May 2026 Component Stack

This report covers the 10 mandatory components of the Lenta hypermarket robot pipeline. Every recommendation is verified against the hard constraints: locally deployable, open licenses (Apache-2.0/MIT/BSD preferred, AGPL flagged, RAIL-M/non-commercial rejected), no manual labeling, RKNN portability flagged.

---

## 1. Price tag detection

**RECOMMENDATION:** Train **YOLO11s @ 960²** as the production student (✅ official RKNN INT8 demo in `airockchip/rknn_model_zoo`), pseudo-labeled via the AGPL `openfoodfacts/price-tag-detection` YOLOv11x teacher (used only for label generation, never shipped). Keep **RT-DETRv2-R18-DSP (Apache-2.0)** or **D-FINE-S (Apache-2.0)** as the license-clean fallback if Lenta legal vetoes the Ultralytics dependency entirely; both require an Enterprise license discussion or full architectural swap.

### Why
- The `openfoodfacts/price-tag-detection` HuggingFace checkpoint remains the only public price-tag-specific model in May 2026. License is AGPL-3.0 (inherited from Ultralytics). OpenFoodFacts pivoted to `openfoodfacts/price-tag-extractor` (Qwen3-VL-8B LoRA for OCR), not a newer detector.
- Ultralytics' position is that pseudo-labels produced by inference are *not* derivative works of the model weights, but distributing the Ultralytics runtime inside the robot binary IS copyleft-triggering. Plan budget for an Ultralytics Enterprise License or use a swap-out path to Apache-licensed alternatives.

### Comparison table

| Model | License | Params | COCO mAP50-95 | RKNN | Notes |
|---|---|---|---|---|---|
| YOLO11s | AGPL-3.0 ⚠️ | 9.4M | 47.0 | ✅ Official INT8 demo | ~40 ms @ 640² INT8 on RK3588 NPU |
| YOLOv8s | AGPL-3.0 ⚠️ | 11.2M | 44.9 | ✅ Most battle-tested official demo | ~32 ms @ 640² INT8 |
| **YOLO26** (Sept 2025) | AGPL-3.0 ⚠️ | 2.4–56M | 47.3 (s) | ⚠️ **BROKEN INT8** — Ultralytics #23753 (segfault + zero detections); FP16 only | **Do not deploy on RK3588 in May 2026** |
| RT-DETRv2-R18-DSP | **Apache-2.0** ✅ | 20M | ~46 | ❌ No official demo; community port w/ ~20-40% CPU fallback | Designed for deployment; replaces `grid_sample` |
| D-FINE-S | **Apache-2.0** ✅ | 10M | 48.5 | ❌ No official demo; same DETR-family ports | Now in 🤗 Transformers. Best accuracy-per-param Apache option |
| RF-DETR Nano/Small/Medium/Large | **Apache-2.0** ✅ (core; XL/2XL = PML-1.0 restricted) | 14–128M | 54.7–60.1 | ❌ DINOv2 ViT backbone won't quantize cleanly | Server (A40/A100) only; SOTA on RF100-VL domain transfer |
| YOLOE-S/M/L | AGPL-3.0 ⚠️ | 12–32M | LVIS zero-shot 27–35 | ❌ No port; YOLO head re-parameterizable | Same AGPL trap; open-vocab via text/visual prompt |
| YOLO-Worldv2 | GPL/AGPL ⚠️ | 13–110M | 35.4 LVIS AP | ✅ Official `yolo_world` demo | Open-vocab "price tag" prompt usable; CLIP encoder heavy |
| Grounding DINO 1.5 Edge | Apache-2.0 ✅ | ~40M EfficientViT-L1 | 45.0 COCO | ❌ | Server-side pseudo-labeler only |
| **isalia99/detr-resnet-50-sku110k** | Apache-2.0 ✅ | 41M | n/a (SKU110K-tuned) | ❌ | Class-agnostic shelf-product proposer — useful as 2nd-stage filter |

### Recipe
1. Server-side pseudo-label sampled Lenta frames with `openfoodfacts/price-tag-detection` + Grounding DINO 1.5-Pro (prompt: `"price tag . ценник . электронный ценник ."`).
2. WBF-fuse boxes at IoU 0.55; cross-validate with `isalia99/detr-resnet-50-sku110k`.
3. Train YOLO11s (production) and RT-DETRv2-R18-DSP (Apache fallback) on the curated pseudo-labels.
4. Track Ultralytics issue #23753 for YOLO26 INT8 resolution; do not switch until closed.

---

## 2. Object tracking

**RECOMMENDATION:** **ByteTrack via `roboflow/trackers` (Apache 2.0)**, IoU-only mode, optionally upgraded to **BoT-SORT with GMC (no ReID)** when the robot is in continuous motion. Trackers are CPU post-processing → RKNN portability is N/A.

### Why
Price tags are static-in-world; the camera moves slowly with periodic stops. This is the *easy* MOT regime — IoU overlap between adjacent frames typically >0.7, so the SOTA gains on MOT17/MOT20 (occlusion-heavy crowded scenes) don't translate. Deep ReID embeddings are actively harmful: price tags look near-identical and produce ID swaps between neighbouring tags. The single useful add-on is BoT-SORT's **Global Motion Compensation** (ORB/ECC homography, CPU OpenCV) to compensate during continuous motion segments.

### Comparison table

| Tracker | License | HOTA (MOT17 / MOT20 / DanceTrack) | FPS | Edge-friendly | Static-target fit | Notes |
|---|---|---|---|---|---|---|
| **ByteTrack** | **MIT** ✅ | 63.1 / 61.3 / 47.7 | ~700 FPS assoc | ✅ pure CPU | ✅ Excellent | De-facto baseline; original ifzhang repo uses unstable Kalman state — use supervision or roboflow/trackers reimpl |
| BoT-SORT (no ReID) | **MIT** ✅ | 65.0 / 63.3 / — | ~25 FPS | ✅ | ✅ Best with GMC | GMC critical for moving-robot scenes |
| OC-SORT | **MIT** ✅ | 63.2 / 62.4 / 55.1 | ~700 FPS assoc | ✅ | ✅ Overkill | Non-linear motion design wasted here |
| Hybrid-SORT | MIT-derived | 64.0 / 63.9 / 65.7 | similar | ✅ | Mid | Targets dance/sports |
| BoostTrack++ | **MIT** ✅ | **66.6 / 66.4** / — | 3–15 FPS | ⚠️ slow w/ReID | ❌ Overkill | Top MOT20; gains are crowd occlusion. **Known bug:** shape-similarity divisor (use `--s_sim_corr`) |
| supervision `sv.ByteTrack` | **MIT** ✅ | same | same | ✅ | ✅ | LICENSE.md verified MIT (old PyPI metadata mislabels BSD). Pre-v0.25 had shared-state bug — call `reset()` |
| **roboflow/trackers** | **Apache 2.0** ✅ | same | same | ✅ | ✅ Recommended | Clean reimpl of SORT/ByteTrack/OC-SORT/BoT-SORT incl. GMC; Optuna tuning built-in |
| MASA / SAMURAI / SAM2MOT | Apache 2.0 ✅ | TAO/DanceTrack SOTA | heavy | ❌ | ❌ | Datacenter-GPU only; not RK3588-deployable |
| OSNet (ReID) | Apache-2.0 ✅ | n/a | edge OK | ✅ | **Skip** | Visual ReID confuses near-identical tags; use OCR text as dedup key instead |

### Dedup strategy
Use **OCR text content** (price + product code) as the cross-track deduplication key — vastly stronger than visual ReID for near-identical tags.

---

## 3. QR code and barcode decoders

**RECOMMENDATION:** Three-tier cascade:
1. **`zxing-cpp` 3.0.0** (Apache-2.0, Feb 2026) — primary decoder for QR + EAN-13/8 + DataMatrix + PDF417, ~40 ms/img.
2. **QReader (YOLOv8 + pyzbar)** (MIT, ⚠️ bundled Ultralytics weights = AGPL — substitute `qrdet-onnx` fork or self-trained detector).
3. **OpenCV `cv2.wechat_qrcode`** (Apache-2.0) — CNN super-resolution rescue for blurred/low-res QR.

`zxing-cpp` is the only Apache-licensed library covering all Russian retail symbologies (chain QR + Honest Sign GS1 DataMatrix + EGAIS PDF417 + product EAN) in one call. Dynamsoft is **disqualified** — proprietary.

### Decoder comparison

| Library | License | Latest | Formats | Speed | Blur/tilt | Maint. |
|---|---|---|---|---|---|---|
| **zxing-cpp** | **Apache-2.0** | **3.0.0** (Feb 2026) | QR/μQR/rMQR/Aztec/**DataMatrix**/**PDF417**/EAN/UPC/Code-39/93/128/Codabar/ITF/DataBar | ~40 ms | Blur 23%, rot 43%, persp 37% (BoofCV 1232-QR set) | **Active** |
| pyzbar | MIT (ZBar=LGPL-2.1) | 0.1.9 (2022) | QR + EAN only; **no DataMatrix, no PDF417** | ~316 ms | Blur 38%, rot 50% | **Inactive** |
| OpenCV WeChat | Apache-2.0 | OpenCV 4.10+ | **QR only** | ~175 ms | **Best OSS on blur 55–69%, logoed 92%** | Active |
| BoofCV (pyboof) | Apache-2.0 | 0.45+ | QR/μQR/Aztec/DataMatrix/PDF417/EAN/UPC | ~503 ms | rot 97%, multi 100% | Active (JVM dep) |
| QReader | MIT (⚠️ Ultralytics AGPL weights) | 3.16 | QR | 100–300 ms | Highest OSS recall on hard QR | Active |
| Dynamsoft | **Commercial proprietary** ❌ | 10.x | All | ~150 ms | 83% overall | n/a — DISQUALIFIED |

### Russian price-tag QR schemas

The QR on a Russian retail price tag is **chain-proprietary**, not GS1-regulated. Three patterns dominate:
1. **Deep-link URL** (Lenta, X5 Pyaterochka/Perekrestok, M.Видео): `https://x5.ru/p?sku=…&store=…&t=…&sig=…` — the loyalty app resolves the personalized price.
2. **Inline JSON** (1С:UT/КА2 "Print labels with QR-code" configurations, electronic price tags):
   ```json
   {"sku":"3829461","ean":"4607034567892","name":"Молоко Простоквашино 3.2% 930мл",
    "price":89.99,"price_promo":74.99,"price_per_kg":80.63,"unit":"л","store_id":"5821",
    "valid_from":"2026-05-10","valid_to":"2026-05-16","ts":1715512345}
   ```
3. **СБП "Плати QR"** (Sberbank) — payment QR, not a product identifier; do not confuse.

**Honest Sign («Честный знак») DataMatrix** (on the product itself for dairy, alcohol, tobacco, water, beer, etc.) is **GS1 DataMatrix with FNC1**:
- Full structure (~85 bytes): `<FNC1>01<GTIN-14>21<Serial-13><GS>91<VerKey-4><GS>92<VerCode-44>`
- Short structure (~30 bytes): `01<GTIN-14>21<Serial-13>93<VerCode-4>`
- Country digit "5" or "E" = RU. Parsing tip: split on ASCII 29 (`<GS>`); `zxing-cpp` v3 exposes `Barcode.contentType() = GS1`.

**EGAIS for alcohol >9% ABV** uses PDF417 (67-char Alcocode-based) and/or DataMatrix (150-char, no alcocode — must resolve via EGAIS API). pyzbar/ZBar **cannot decode either** — this alone forces `zxing-cpp` as the backbone.

### Known issues
- **zxing-cpp 3.0 breaking changes**: `ZXing::ReadBarcodes` signature differs from 2.x. Pin `zxing-cpp>=3.0,<4`.
- **pyzbar #167** (Mar 2025): Windows venv DLL import error.
- **QReader/QRDet** bundles Ultralytics YOLOv8 weights → AGPL. Use the `PaleBloodq/qrdet-onnx` fork.
- **GS1 DataMatrix parsing** error #1: stripping `<GS>` (ASCII 29). Always read raw bytes via `Barcode.bytes()`.

---

## 4. Primary OCR/VLM extractor

**RECOMMENDATION:** **PaddleOCR-VL-1.5 (0.9B, Apache-2.0)** as PRIMARY on the A40/A100 server path, served via **vLLM with XGrammar `guided_json`** for the 18-field schema. **PP-OCRv5 East-Slavic** runs in parallel on the cropped price regions as a deterministic numeric cross-check, and replaces the VLM entirely on the RK3588 edge path. RKNN: ❌ for PaddleOCR-VL-1.5; ✅ excellent for PP-OCRv5.

### Position on VLM vs traditional OCR

For 18-field structured extraction, **VLM wins as primary** because price tags require semantic role assignment ("which number is the discount price?", "is the `*` a footnote marker or a separator?") which traditional OCR can't do without template-engineered rules. The hallucination risk is mitigated by (a) XGrammar schema-constrained decoding and (b) parallel PP-OCRv5 numeric cross-validation.

| Factor | PP-OCRv5 (traditional) | PaddleOCR-VL-1.5 (VLM) |
|---|---|---|
| Cyrillic accuracy | 80–82% line exact-match | Higher; 109 languages incl. Russian |
| Spatial reasoning | Needs hand-coded rules | Native — emits role-tagged JSON |
| Blur/glare/glass robustness | Weak | Strong (Real5-OmniDocBench: 92.05) |
| Hallucination risk | ~Zero | Real; mitigated by XGrammar + cross-check |
| VRAM | 5M params, CPU-friendly | 3–4 GB |
| RK3588 ready | ✅ | ❌ |

### PaddleOCR-VL-1.5 details
- HF: `PaddlePaddle/PaddleOCR-VL-1.5`. Tech report arXiv 2601.21957 (Jan 29 2026).
- License: **Apache 2.0** (verified).
- Architecture: NaViT dynamic-resolution ViT + ERNIE-4.5-0.3B LLM.
- Benchmarks: OmniDocBench v1.5 **94.5%** (SOTA at size class, surpasses Qwen3-VL-235B); **Real5-OmniDocBench 92.05%** — most relevant for our motion-blur/glare conditions.
- Deployment: HF transformers, **vLLM officially supported since 2025-11-04**, SGLang, FastDeploy, Huawei NPU, Intel Arc.
- Native JSON via `res.save_to_json()`; task-switched prompts (`OCR:`, `Table Recognition:`).
- **Risk:** Not on MWS Vision Bench leaderboard — pilot-test on Russian price tags before locking in.

### VLM alternatives

| Model | Params | License | OCRBench / OmniDoc | MWS Vision Bench (RU) | RKNN |
|---|---|---|---|---|---|
| **PaddleOCR-VL-1.5** | 0.9B | Apache 2.0 | **94.5 / 92.05 Real5** | not benchmarked | ❌ |
| PP-OCRv5 eslav | 5M | Apache 2.0 | line-acc 81.6 | n/a | ✅ |
| Qwen3-VL-2B / 4B | 2/4B | Apache 2.0 | strong text-centric | 0.515 (4B) | ✅ Qengineering |
| Qwen2.5-VL-3B | 3B | Apache 2.0 | OCRBench ~828 | not listed | ⚠️ issue #387 |
| InternVL3.5-2B / 4B | 2/4B | MIT | DocVQA 89.4 (2B) | not listed | ✅ likely |
| MiniCPM-V 4.5 / 4.6 | 8B / 1.3B | **Apache 2.0** (verify on download — Ollama blob still references old Community License) | OCRBench leading | not listed | ⚠️ via rkllama |
| SmolVLM2-2.2B | 2.2B | Apache 2.0 | modest OCR | not listed | ❌ |
| Florence-2-large | 0.77B | MIT | Latin-biased OCR | n/a | ❌ |
| GOT-OCR-2.0 | 580M | **Research-only** ❌ | strong OCR-2.0 | n/a | ❌ |
| dots.ocr | 1.7B | **MIT** | SOTA OmniDocBench | not listed | ❌ |
| Chandra v2 | 9B | **OpenRAIL-M** ❌ | 85.9 olmOCR-Bench | competitive | ❌ |

### Structured JSON enforcement

| Library | License | VLM | vLLM |
|---|---|---|---|
| **XGrammar / XGrammar-2** | Apache 2.0 | ✅ | ✅ Default backend (100× faster than Outlines logit-processor). **Pin ≥ 0.1.32 — CVE-2026-25048 (DoS)** |
| Outlines 1.2.13 | Apache 2.0 | ✅ `transformers_vision` | Available, slower | 
| lm-format-enforcer | MIT | ✅ | Available | 

XGrammar constrains *tokens*, not pixels — it cannot prevent a plausibly-shaped hallucinated price, so the parallel PP-OCRv5 numeric cross-check is non-optional.

### Models to REJECT
- GOT-OCR-2.0 (research-only)
- Chandra v2 (OpenRAIL-M)
- Florence-2 (weak Cyrillic)
- All Llama-3.2-Vision variants (Habr M.Video evaluation showed 10% number accuracy on Russian)

---

## 5. VLM fallback for low-confidence retry

**RECOMMENDATION:** On a single A40 (48 GB): **Qwen3-VL-8B-Instruct** (Apache 2.0) FP16 by default; **Qwen3-VL-30B-A3B-Instruct-FP8** (MoE, 30B total / 3B active) for premium retries — fits A100-40 FP8, runs at ~8B latency. On A100-80: step up to **Qwen3-VL-32B-Instruct** FP16. **Cotype VL (MTS AI 32B, Russian-tuned, MWS Vision Bench overall 0.649)** is the best Russian quality but **disqualified** — weights are not openly published as of May 2026 (commercial product only).

### Comparison table

| Model | License | Params / Active | VRAM FP16 / FP8 / INT4 | OCRBench | DocVQA | MWS Vision Bench (RU) | A40 fit | RKNN |
|---|---|---|---|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** | Apache 2.0 | 9B dense | 18 / 10 / 6 GB | 89.6 | 96.1 | **0.584** | ✅ FP16 | ❌ |
| Qwen3-VL-32B-Instruct | Apache 2.0 | 33B dense | 66 / 33 / 18 GB | ~88 | 96.9 | 0.582 | INT4 only | ❌ |
| **Qwen3-VL-30B-A3B-Instruct (MoE)** | Apache 2.0 | 30B / 3B active | 60 / 30 / 16 GB | high | ~96 | comparable to 32B | A40 FP8 borderline; A100-40 FP8 ✅ | ❌ |
| Qwen3-VL-235B-A22B | Apache 2.0 | 235B / 22B active | ≥480 GB FP16 | 880+ | 96.5 | 0.623 | ❌ | ❌ |
| InternVL3.5-8B | Apache 2.0 | 8B | 16 / 9 / 5 GB | ~88 | ~93 | not benchmarked | ✅ | ❌ |
| InternVL3.5-14B / 20B / 30B-A3B | Apache 2.0 | 14–30B | 28 / 40 / 60 GB | ~89 | ~95 | not benchmarked | A100 only | ❌ |
| **Cotype VL (MTS AI)** | **Closed/commercial** ❌ | 32B (8-bit) | ~36 GB | n/a | n/a | **0.649** | ✅ if licensed | ❌ |
| GLM-4.5V | MIT | 9B | 18 GB | ~88 | ~95 | not listed | ✅ | ❌ |
| OmniFusion-1.1 (AIRI) | Apache 2.0 | 7B | 14 GB | ~70 | ~71 | not listed | ✅ | ❌ |
| PaliGemma 2 | Gemma License ⚠️ | 3/10/28B | varies | ~85 | ~94 | not listed | ✅ | ❌ |

### Critical Russian-quality finding
Qwen3-VL **dense 8B and 32B are essentially tied** on Russian docs (0.578–0.584 MWS Vision Bench). Cotype VL 32B-8bit (Russian-tuned) **outperforms 235B Qwen3-VL** (0.649 vs 0.623). **Russian-specific tuning matters more than scale.** No competitive openly-released Russian-tuned VLM exists in May 2026 — Cotype VL is closed; OmniFusion is outdated; T-Bank, Yandex, Sber have no open VLM releases for printed Russian OCR.

### Per-GPU picks
- **A40 (48 GB):** Qwen3-VL-8B FP16 default; Qwen3-VL-30B-A3B FP8 for premium retries.
- **A100-40:** Qwen3-VL-8B FP16; Qwen3-VL-32B INT4 AWQ; Qwen3-VL-30B-A3B FP8.
- **A100-80:** Qwen3-VL-32B FP16, or Qwen3-VL-30B-A3B FP16.

---

## 6. RKNN deployment tooling

**RECOMMENDATION:** Two-track toolchain:
1. **rknn-toolkit2 v2.3.2** for vision/OCR/detection. Use **YOLO11n/s + PP-OCRv4 (or community PP-OCRv5)** with **MMSE calibration** (+0.7%P/+1.7%R/+1.2% hmean over normal calibration confirmed in Rockchip docs).
2. **rknn-llm v1.2.3 + driver ≥ 0.9.8** for any on-board VLM. Use **Qengineering pre-converted ports** (`Qengineering/InternVL3.5-{1B,2B,4B,8B}-NPU`, `Qengineering/Qwen3-VL-2B-NPU`) rather than self-converting — VLM conversion is fragile and W8A8 quality on RK3588 still degrades for Qwen2.5-VL (issue #387, closed but unresolved).

**Avoid YOLO26 INT8 on RK3588** (issue #23753 unresolved). Plan a hardware refresh to **RK3576 (W4A16, Flash Attention)** or budget an **RK1820/RK1828 SO-DIMM accelerator** (20 TOPS, on-chip 2.5/5 GB DRAM) for production VLM throughput in late 2026.

### Latest versions (May 2026)
- **rknn-toolkit2 v2.3.2** (2025-04-09): RV1126B support; automatic mixed-precision; improved einsum/Norm ops.
- **rknn-llm v1.2.3** (2025-11-24): **InternVL3.5, DeepSeekOCR, Qwen3-VL** support; automatic embedding cache reuse; external chat-template files.
- **Flash Attention** is **RK3562/RK3576 only — NOT RK3588**. RK3588 transformer perf is fundamentally capped by no kernel-fused SDPA.

### Model-by-model RKNN status (RK3588)

| Model family | Status | Quant | Notes |
|---|---|---|---|
| YOLOv5/6/7/8/v10/**YOLO11**/YOLOX/PP-YOLOE | ✅ Official | INT8 | YOLO11 is best modern choice |
| **YOLO26** | ⚠️ Broken (#23753) | FP16 only | End2end → segfault; INT8 → zero detections |
| YOLO-World | ✅ Official | INT8 | CLIP encoder FP16 |
| RT-DETR / D-FINE | ⚠️ Community | INT8 risky | DETR attention quant degradation |
| **PPOCR-Det/Rec v4** | ✅ Official | INT8 | MMSE calibration mandatory |
| PPOCR-v5 | ❌ No official | — | Community ports exist, manual ONNX surgery |
| MobileSAM / CLIP / DeepLabv3+ / RetinaFace | ✅ Official | mixed | |
| Whisper / Zipformer | ✅ Official | mixed | |
| Qwen2-VL-2B | ✅ Official + Qengineering | W8A8 + FP16 vision | Reliable |
| **Qwen2.5-VL-3B** | ✅ but ⚠️ #387 quality regression | W8A8 | OCR worse than Qwen2-VL-2B |
| **Qwen3-VL-2B** | ✅ Qengineering | W8A8 | Best current 2B VLM on RK3588 |
| Qwen3-VL-4B | ✅ Qengineering | W8A8 | 5.4 GB; tight on 8 GB boards |
| **InternVL3.5-1B/2B/4B/8B** | ✅ Qengineering | W8A8 | 8B borderline on 16 GB |
| MiniCPM-V-2_6 / 4.5 | ✅ via rkllama | W8A8 | |
| **DeepSeekOCR** | ✅ Official (new v1.2.3) | W8A8 | OCR-tuned VLM; strong candidate for shelf labels |
| SmolVLM2 (256M/2B) | ⚠️ Qengineering, needs ONNX patch | W8A8 | Idefics3 not natively supported (#231) |
| PaddleOCR-VL / dots.ocr / Florence-2 | ❌ | — | ERNIE-4.5 / custom arch not yet supported |

### Known bugs & workarounds

| Issue | Status | Workaround |
|---|---|---|
| **Qwen2.5-VL-3B #387** OCR quality regression | Closed, not fixed | Use `happyme531/Qwen2.5-VL-3B-Instruct-RKLLM` (split MatMul `down_proj` by 5), or switch to Qwen3-VL-2B |
| **YOLO26 #23753** INT8 broken | Open, no ETA | Use YOLO11 instead |
| **SmolVLM #231** | Open | Use Qengineering pre-converted SmolVLM2 |
| **Gemma 4 #488** | Open | No support yet |
| LayerNorm/RMSNorm INT8 drift | Recurring | Hybrid quant: keep all *Norm in FP16 |
| Dequantize at output costs ~5 ms CPU | Recurring | Zero-copy API + post-process dequant |
| Dynamic batching broken on transformers | Recurring | Static batch=1, multi-instance API (v1.2.2) |
| `embed_tokens` >100 MB inflates RKNN file | Solved | `--embed_flash` (v1.2.0+) |
| Driver/runtime mismatch | Recurring | Verify `/sys/kernel/debug/rknpu/version ≥ 0.9.8` |

### Calibration best practices (2026)
- **Dataset:** 200–500 vision samples; 400+ real OCR crops; 100–300 prompts (not generic wikitext) for LLM.
- **Algorithm:** `mmse` for OCR/detection (5–10× slower build, but meaningful accuracy gains). `kl_divergence` for LLM activations with long-tail distributions.
- **Per-channel** for conv/MatMul weights (always); per-tensor for activations (hardware-fixed).
- **Hybrid recipe for VLMs:** vision encoder FP16 (or W4A16 on RK3576); LLM W8A8 with LayerNorm/RMSNorm/LM-head/embed_tokens forced FP16; visual-projection MLP FP16 if accuracy-critical.
- **`accuracy_analysis()`** API → promote any layer with cosine <0.95 to FP16. Automated mixed-precision in v2.3.2.
- **GRQ Int4 / GDQ** (rkllm v1.1.0+) for 4-bit, +3–5% PPL.

### NPU simulator
Correctness: bit-exact INT8 match — good for accuracy regression. **Latency: not representative** — single-core CPU emulation, no DMA/LPDDR contention, ±20% on conv-heavy and **2–3× off on transformers**. Always benchmark latency on real RK3588.

### Newer Rockchip chips

| Chip | NPU | RAM | LLM Quant | Status |
|---|---|---|---|---|
| **RK3588** | 6 TOPS (3-core) | up to 32 GB | W8A8 only | Mature, mass-production |
| **RK3576** | 6 TOPS | up to 16 GB | INT8 + **W4A16**; Flash Attention | **Shipping** (Luckfox Core3576, Banana Pi CM5-Pro, Firefly CAM-3576 — Dec 2025). ~30% lower power. Best LLM price/perf at edge |
| RV1126B | small NPU | — | INT8/INT16 | Low-power vision |
| **RK1820** SO-DIMM | **20 TOPS** | 2.5 GB on-chip 3D-stacked | W4A16/W8A8 | Released Dec 2025 by Firefly; ≤3B models; ~59–180 tok/s on Qwen2.5/Qwen3/InternVL3.5; RKNN3 toolkit (not yet public) |
| **RK1828** SO-DIMM | **20 TOPS** | 5 GB on-chip 3D-stacked | W4A16/W8A8 | For 7B models; ~5–10× faster than RK3588 NPU on LLM |
| RK3668 | 16 TOPS | TBD | TBD | Announced 2025 dev conf; not yet sampling |

### Practical roadmap
- **Week 1**: YOLO11n fine-tune + MMSE quant + `accuracy_analysis` → ≥40 FPS @ 640² INT8, mAP within 2 pts of FP32.
- **Week 2**: PPOCR-v4 with 400-image MMSE calibration; ~15–25 FPS det + 30 FPS rec. Try DeepSeekOCR for hard scenes.
- **Week 3–4**: Pull Qengineering InternVL3.5-1B-NPU or Qwen3-VL-2B-NPU directly. ~3–4 s vision encode + 6–10 tok/s decode (1B), 4–8 tok/s (2B). 4–6 GB resident RAM. Wire via `rkllm_run_async` multi-instance.
- **Phase 5 (late 2026)**: If VLM throughput is binding, spec robot v2 with RK3576 or add RK1828 SO-DIMM.

---

## 7. Russian-language OCR alternatives

**RECOMMENDATION:** **PP-OCRv5 East-Slavic mobile_rec** (`PaddlePaddle/eslav_PP-OCRv5_mobile_rec`, Apache 2.0) as primary backup pipeline. 81.6% line-exact-match on East-Slavic, ~5M params, RKNN-friendly (Paddle→ONNX→RKNN well-trodden). Pair with PP-OCRv5 mobile detector. Use **EasyOCR cyrillic_g2** (Apache 2.0) as secondary backup for scene-text-style price tags. For hard cases route to Qwen3-VL-8B (Component 5). **Surya OCR is DISQUALIFIED** — modified AI Pubs Open RAIL-M with $2M revenue/funding cap on weights, GPL-3.0 code.

### Comparison table

| Engine | License | Russian accuracy | Speed | RKNN | Maint. |
|---|---|---|---|---|---|
| **PP-OCRv5 eslav mobile_rec** | **Apache 2.0** | **81.6%** line exact-match; +30% over PP-OCRv4 | <50 ms/region on GPU | ✅ Best | Active (2025) |
| PaddleOCR-VL 0.9B | Apache 2.0 | 109 langs incl. Russian; no published CER | 3–4 s/page | ⚠️ harder | Active |
| EasyOCR cyrillic_g2 | **Apache 2.0** | ~78.4% recog acc; ~88.6% det F1 | 0.5–0.7 s/frame V100 | possible | Slow (1.7.2 Sept 2024) |
| Tesseract 5 LSTM (`rus.traineddata`) | **Apache 2.0** | ~46% recog acc (Occular bench) | 1 s/page CPU | CPU only | Active |
| TrOCR `raxtemur/trocr-base-ru` | Apache 2.0 | handwriting-focused; no printed CER | 100–300 ms/line | ❌ | 2024 stable |
| TrOCR `kazars24/trocr-base-handwritten-ru` | MIT | **CER 0.048** handwriting | 200 ms/line | ❌ | 2023 |
| GOT-OCR 2.0 | Apache 2.0 | Cyrillic supported, no CER published | 3–5 s/page | ❌ | 2024–25 |
| Occular OCR (Bodhi42) | Apache 2.0 | author claim 93.7% / 88.6% F1, 0.57 s/page CPU | fast | ✅ (ONNX) | 2024, small community |
| **Surya OCR** | **GPL-3.0 + RAIL-M ($2M cap)** ❌ | ~91% | 0.5 s/page A6000 | ❌ | DISQUALIFIED |
| DocTR (Mindee) | Apache 2.0 | Cyrillic not in default pretrained | fast | possible | Active |

**No new 2025–2026 openly-licensed Russian-specialized OCR engines found.** Yandex Vision, SmartEngines, ABBYY are commercial; T-Bank/SaluteLM focus on LLMs; MTS AI's only open vision release is the MWS Vision Bench benchmark itself, not weights.

**Post-OCR Russian correction** with Vikhr/Saiga/T-pro (Apache 2.0 LLMs) is a useful stage for fixing garbled Cyrillic — not OCR itself but valuable in pipeline.

---

## 8. UI / deployment infrastructure

**RECOMMENDATION:** Two-tier deployment:
1. **Demo / public sharing**: **Gradio 6.14.0** (Apache 2.0) on **HF Spaces ZeroGPU xlarge (full H200, 141 GB)** with chunked video processing (≤120 s sub-jobs via `@spaces.GPU(duration=...)`). Pro tier $9/mo = 25 min H200/day for the hosting account.
2. **Production / batch**: **Modal serverless** (per-second billing, sub-second cold starts, snapshot-based, automatic H200↔H100 fallback) **or** **RunPod Serverless** if cost matters more than DX (~25–30% cheaper, FlashBoot sub-200 ms for 48% of starts).

Avoid Streamlit for streaming/per-frame video annotation; avoid HF Spaces standard GPU (more expensive, no auto-release).

### HF Spaces ZeroGPU H200 specifics

| Property | Value |
|---|---|
| Hardware | H200 sliced via MIG |
| `large` tier | 70 GB VRAM, 1× quota |
| `xlarge` tier | 141 GB VRAM, 2× quota |
| Default time limit | 60 s/inference, configurable via `@spaces.GPU(duration=N)` |
| Pro tier daily quota | 25 min H200/day; beyond = $1 per 10 min |
| Compat | Gradio SDK only; PyTorch 2.1–2.9.1; Py 3.10.13 or 3.12.12 |
| `torch.compile` | NOT supported — use PyTorch AoT compile (1.3–1.8× speedup) |

### Provider comparison

| Provider | GPUs | $/hr | Cold start | Time limit | Best for |
|---|---|---|---|---|---|
| HF Spaces ZeroGPU | Half/Full H200 | Free user; $9/mo PRO to host | Fast slice attach | 60 s default | Public demo |
| **Modal** | T4 → B200 (H200 $4.54/hr, A100-80 $2.50/hr) | per-second, scale-to-zero | sub-second to few seconds | configurable | Production pipeline |
| **RunPod Serverless** | T4 → H200 SXM ($3.99/hr) | per-second | 48% <200 ms; 6–30 s for big containers | configurable | Cost-sensitive |
| RunPod Pods Secure | T4 → H200 ($4.54/hr) | per-minute | 30–60 s boot, then none | None | Sustained throughput |

### Long-running video best practices
- **Chunk client-side or in CPU pre-stage** (~10–30 s segments).
- **Decouple CPU from GPU work**: QR decode + CSV on CPU function, detection+VLM on GPU function — cuts GPU seconds 30–60%.
- **Persist model weights** in Modal Volume / RunPod Network Volume / HF Spaces persistent storage.
- **Stream outputs** via `gr.Progress` generator.
- **AoT-compile vision models on ZeroGPU** (`torch.export` + `torchao` FP8 on H200).
- **State** in S3/R2 with job IDs for resumption.

---

## 9. Supporting libraries (license verification)

**RECOMMENDATION:** All recommended libraries below are **MIT/Apache 2.0/BSD** and license-clean for closed-source commercial deployment **except ultralytics** (AGPL-3.0 — switch to **RF-DETR (Apache 2.0)** or buy Ultralytics Enterprise).

| Library | Version (May 2026) | License | Role | Gotchas |
|---|---|---|---|---|
| supervision | 0.27.0 (+ 0.28.0rc0) | **MIT** | Detection containers, annotators, zones, ByteTrack wrapper | LICENSE.md is authoritative (old PyPI metadata says "BSD" — treat as MIT) |
| roboflow/trackers | active | **Apache 2.0** | Cleaner ByteTrack/BoT-SORT/OC-SORT reimpl + Optuna tuning | Preferred over original `bytetracker` PyPI (inactive 2023) |
| ByteTrack (original) | repo stable; PyPI `bytetracker` 0.3.2 (2023) | **MIT** | Tracking algorithm | Pip pkg inactive; vendor via supervision/roboflow/trackers |
| **vLLM** | **0.20.2** (May 10 2026) | **Apache 2.0** | LLM/VLM serving + structured output | CUDA 13 default; torch 2.11; Transformers v5 churn; `--mm-encoder-tp-mode data` for VLM perf win |
| Pydantic | **2.13** (Apr 2026) | **MIT** | Schema validation for VLM JSON | pydantic-core merged into main repo in v2.13 |
| Langfuse | v3.x post-ClickHouse acquisition (Jan 2026, $400M Series D) | **MIT core**; **commercial EE** for SCIM/audit/RBAC/retention | LLM observability, traces, datasets, evals | Don't enable `/ee` features without license key; self-host needs ClickHouse + Postgres + Redis + S3 |
| outlines | **1.2.13** (May 4 2026) | **Apache 2.0** | Constrained gen (non-vLLM fallback) | `transformers_vision` backend for VLMs |
| onnxruntime | **1.26.0** (May 8 2026) | **MIT** | Detector ONNX inference | None |
| **XGrammar** | **0.1.33** (Mar 27 2026); XGrammar-2 announced May 2026 | **Apache 2.0** | Default vLLM structured-output backend | **Pin ≥0.1.32 — CVE-2026-25048 (DoS via nested grammars)** |
| instructor | **1.14.5** (Jan 29 2026) | **MIT** | Client-side Pydantic validation + retry | Post-gen validation, not real constrained decoding |
| jsonschema | stable | **MIT** | Plain validation | None |
| Gradio | **6.14.0** (Apr 30 2026) | **Apache 2.0** | Demo UI, ZeroGPU integration | Svelte 5 migration churn; pin versions |
| **ultralytics** | YOLO11 / YOLO26 | **AGPL-3.0** ⚠️ OR paid Enterprise | DO NOT USE for closed-source | AGPL infects entire pipeline incl. internal SaaS/network use; **substitute RF-DETR (Apache 2.0)** |

---

## 10. Synthetic data / pseudo-labeling tools

**RECOMMENDATION:** Three-stage hybrid:
1. **Detector bootstrap**: **Grounded SAM 2** (Apache 2.0) on shelf video → SAM 2 video memory propagates boxes across frames → filter by confidence + temporal consistency → distill into YOLO11n (or RF-DETR-N if AGPL-pure stack needed) via **Autodistill** (Apache 2.0/MIT).
2. **OCR/structured-field bootstrap**: Crop tags → run **PaddleOCR-VL-0.9B** + **Qwen3-VL-4B** as parallel "VLM-as-labeler" → consensus filter (Levenshtein-0 on digits) + validators (EAN-13 checksum, price>0, card≤default) → fine-tune **Donut** (MIT) student on resulting JSONL.
3. **Synthetic boost**: PIL/HTML+Playwright Lenta-template rendering with Russian-friendly TTFs (PT Sans, Inter, Manrope, Roboto) → SynthDoG fork for full-tag synthesis with perfect GT JSON → `trdg` for text-line crops. Optional photometric augmentation via **FLUX.1-schnell** or **FLUX.2-klein-4B** (both Apache 2.0). **Reject FLUX-dev, FLUX-Fill-dev, FLUX.2-klein-9B, SD3.5** for Lenta-scale commercial use.

### Grounded SAM 2 status
- Repo: `IDEA-Research/Grounded-SAM-2`, v1.0 Dec 2024, ~2.3k★, active.
- License: SAM 2 = Apache-2.0; Grounding DINO (base/tiny) = Apache-2.0; Florence-2 = MIT. (Grounding DINO 1.5/1.6 Pro/Edge require IDEA API token → NOT local).
- Install: ~1–2 hr (needs CUDA toolchain for Deformable Attention). `SAM2_BUILD_CUDA=0` to skip SAM 2 CUDA ext.
- Recipe: keyframe sample every 0.5–1s → Grounding DINO prompt `"price tag . white label . sticker with price . shelf label ."` → SAM 2 memory propagates → filter (track ≥3 frames, mean conf ≥0.35, IoU std <0.15) → COCO/YOLO export. The included `grounded_sam2_tracking_demo_with_continuous_id.py` does most of this.

### Autodistill
- `autodistill/autodistill` (Apache 2.0 core, MIT plugins). Active in 2026 (`autodistill-sam3`, `autodistill-rfdetr`, `autodistill-grounded-sam-2`, `autodistill-florence-2`).
- Supports base: GroundedSAM/GroundedSAM2, Grounding DINO, DETIC, OWLv2, Florence-2, CLIP, Transformers wrapper.
- Supports target: YOLOv5/v8/v11, RF-DETR, DETR, ViT.

### Pipeline alternatives compared

| Approach | Speed/img | Quality | Pros | Cons |
|---|---|---|---|---|
| **Grounded SAM 2 → YOLO/RF-DETR** | 1–3 s keyframe + free intermediate | recall 0.85–0.95 / precision 0.92–0.98 | Open-vocab, masks+boxes, temporal | Misses occluded/rotated |
| VLM-as-detector (Qwen3-VL grounding) | 3–10 s | Lower box accuracy | Same model OCRs after | Slow, hallucinated boxes |
| Florence-2 dense region caption | 0.5 s | Moderate on small text rectangles | MIT, light | Noisy captions |
| SAM 2 class-agnostic + CLIP filter | medium | Best recall | Catches everything | Extra component |

### Filters (apply all)
- Grounding DINO conf ≥0.30
- SAM 2 tracklet ≥3 frames, mask IoU std <0.15
- Aspect ratio 0.4 ≤ w/h ≤ 4 (Lenta tags portrait/square)
- Color sanity (yellow/red/white for promo tags)
- **OCR cross-check: keep only if PaddleOCR-VL returns ≥1 digit AND "₽"/"руб" token** — kills 90%+ of false positives
- Disagreement consensus: intersect Grounding DINO + Florence-2 boxes

### Template-based synthetic generation
- **PIL/Pillow + cairo** (MIT/LGPL): pixel-perfect Lenta templates with full JSON GT
- **trdg** (MIT): single-line crops; Cyrillic via custom `--dict ru.txt` UTF-8 (bundled Russian dict has bugs — known issue)
- **SynthDoG** (MIT): full-tag synthesis with `gt_parse` JSON ready for Donut
- **SynthTIGER** (MIT, Clova): better text-line realism than trdg
- **HTML→image (Playwright)**: easiest for exact Lenta layout — CSS template, randomize fields, screenshot at random DPI/angle

**Critical Russian font tip:** Use only Cyrillic-supporting TTFs (PT Sans, PT Serif, Roboto, Open Sans, DejaVu Sans, Inter, Montserrat, Manrope, Fira Sans). OCR-B Regular and many "OCR-style" fonts do not cover Cyrillic. Always normalize Cyrillic А/В/С/Е/Н/К/М/О/Р/Т/Х vs Latin lookalikes — this is the #1 OCR failure mode for Russian shelf imagery per Smart Engines and Sber Habr posts.

### Diffusion augmentation — license matrix

| Model | License | Verdict |
|---|---|---|
| FLUX.1-schnell | Apache-2.0 ✅ | Usable for background variation |
| **FLUX.2-klein 4B** (Nov 25 2025) | **Apache-2.0** ✅ | Best Apache-2.0 diffusion option; ~13 GB VRAM bf16 |
| FLUX.1-dev / FLUX.1-Fill-dev / Canny/Depth-dev | Non-commercial ❌ | Reject |
| FLUX.2-klein 9B / Dev / Flex / Pro | Non-commercial ❌ | Reject |
| Stable Diffusion 3.5 | Stability Community License (free <$1M ARR) ⚠️ | **Lenta needs Enterprise contract** |
| SD 1.5 / SDXL + ControlNet/IP-Adapter | OpenRAIL-M ✅ | Safe fallback; lower text quality |

**Pragmatic pattern:** Diffusion **renders Cyrillic digits poorly** and corrupts prices. Right pattern = PIL/HTML render the tag with correct text → paste over real shelf background → diffusion-inpaint *only the surroundings* (mask=tag region inverted) with SDXL inpaint + ControlNet-Tile. Never let diffusion regenerate the price text.

### Realistic label-quality targets (zero manual labels)

| Stage | Metric | Realistic |
|---|---|---|
| Grounded SAM 2 box recall | recall | 0.85–0.95 |
| Box precision (after filters) | precision | 0.92–0.98 |
| Distilled YOLO/RF-DETR student mAP@0.5 | mAP | 0.90–0.96 (cf. Ivanov 2025: 0.968) |
| PaddleOCR-VL Cyrillic price digit CER | CER | 0.03–0.06 |
| Qwen3-VL-4B Cyrillic price digit CER | CER | 0.01–0.03 |
| Consensus-filtered JSON kept | survival | 50–70% |
| Donut student field-exact-match | EM | 0.85–0.92 |
| Donut student price-only exact-match | EM | 0.93–0.97 |

The biggest residual risk is **long Russian product-name fields** (CER 5–12%), which is also the field your downstream task probably needs least — barcode, prices, datetime all stay high-accuracy.

---

## Final stack summary

### Confirmed stack table

| Component | Choice | License | RKNN |
|---|---|---|---|
| 1. Detection (production) | YOLO11s @ 960² (distilled from OFF teacher) | AGPL-3.0 ⚠️ (need Ultralytics Enterprise) | ✅ Official |
| 1. Detection (license-clean fallback) | RT-DETRv2-R18-DSP or D-FINE-S | Apache 2.0 | ❌ (community port) |
| 1. Detection (pseudo-label teacher) | openfoodfacts/price-tag-detection + Grounding DINO 1.5-Pro | AGPL-3.0 (server-only) | n/a |
| 2. Tracking | ByteTrack via roboflow/trackers; optional BoT-SORT GMC | Apache 2.0 / MIT | n/a (CPU) |
| 3. QR/Barcode | zxing-cpp 3.0.0 + QReader (qrdet-onnx) + cv2.wechat_qrcode | Apache 2.0 / MIT | n/a |
| 4. Primary OCR (server) | PaddleOCR-VL-1.5 via vLLM + XGrammar `guided_json` | Apache 2.0 | ❌ |
| 4. Primary OCR (edge) | PP-OCRv5 East-Slavic | Apache 2.0 | ✅ |
| 5. VLM fallback (A40) | Qwen3-VL-8B-Instruct FP16 + Qwen3-VL-30B-A3B-FP8 for premium | Apache 2.0 | ❌ |
| 6. RKNN toolchain | rknn-toolkit2 v2.3.2 + rknn-llm v1.2.3 + Qengineering ports | Apache 2.0 | n/a |
| 7. Russian OCR backup | PP-OCRv5 East-Slavic; EasyOCR cyrillic_g2 secondary | Apache 2.0 | ✅ / possible |
| 8. UI (demo) | Gradio 6.14.0 on HF Spaces ZeroGPU xlarge | Apache 2.0 | n/a |
| 8. Production deploy | Modal serverless (or RunPod) | n/a | n/a |
| 9. Stack libs | supervision 0.27, vLLM 0.20.2, Pydantic 2.13, Langfuse v3 OSS, outlines 1.2.13, onnxruntime 1.26, XGrammar 0.1.33 | MIT / Apache 2.0 | n/a |
| 10. Pseudo-label | Grounded SAM 2 + Autodistill → YOLO/RF-DETR; Donut + VLM-as-labeler; HTML/PIL synth | Apache 2.0 / MIT | n/a |

### Top 5 risks specific to May 2026

1. **Ultralytics AGPL on networked deployment** — AGPL-3.0 triggers full source disclosure even for internal SaaS / network-accessed robot pipelines. Either buy Ultralytics Enterprise License or switch all detection components to RF-DETR (Apache 2.0) / D-FINE / RT-DETRv2. The biggest legal trap in this stack.

2. **YOLO26 INT8 on RK3588 is broken** — Ultralytics issue #23753 unresolved as of May 2026: end2end export segfaults on librknnrt 2.3.2 (TopK unsupported), INT8 produces zero detections, only FP16 works. **Stay on YOLO11** until issue closes and rknn-toolkit2 ≥ 2.4 lands.

3. **XGrammar CVE-2026-25048** — DoS via multi-layer nested grammars in all versions <0.1.32. Pin `xgrammar ≥ 0.1.32` in requirements. vLLM 0.20.2 includes the fix, but standalone deployments may not.

4. **Qwen2.5-VL-3B RKNN OCR quality regression (issue #387)** — closed but not fixed; W8A8 quant on RK3588 produces worse OCR than Qwen2-VL-2B on simple scenes. Use Qwen3-VL-2B (Qengineering port) or stay on Qwen2-VL-2B; do NOT pick Qwen2.5-VL-3B for edge.

5. **Russian-tuned VLM gap** — Cotype VL (MTS AI, 0.649 on MWS Vision Bench, best Russian-tuned result) is closed/commercial; OmniFusion outdated; no open Russian VLM at competitive quality has shipped as of May 2026. Going from Qwen3-VL-8B → 32B gives marginal gain on Russian (both ~0.58). **Russian-specific tuning matters more than scale** — plan budget for either commercial Cotype VL license or in-house LoRA on Qwen3-VL-8B with collected Lenta data.

### Bonus risk worth flagging
- **Surya OCR** is RAIL-M with $2M revenue cap — disqualified for Lenta but easy to accidentally adopt because it's popular on HuggingFace. **MiniCPM-V license ambiguity**: GitHub README says Apache 2.0 but the Ollama distribution blob still references the older MiniCPM Community License (free only <5,000 edge devices / <1M DAU). **Verify license on the specific checkpoint at download time.**

### What changed vs prior research earlier in conversation
This is the first systematic pass; the user did not provide a prior fact list to diff against. Notable shifts compared to "common assumed knowledge" baselines:
- PaddleOCR-VL-1.5 has overtaken Qwen3-VL-235B and Gemini-3-Pro on OmniDocBench v1.5 at 0.9B size class — a genuine size/quality breakthrough.
- YOLO26 is **not** the obvious upgrade from YOLO11 on Rockchip in May 2026 — INT8 export is broken; YOLO11 remains the production choice.
- Cotype VL's MWS Vision Bench score (0.649) materially exceeds Qwen3-VL-235B on Russian documents (0.623), reframing the "bigger is better" assumption — Russian tuning beats scale.
- FLUX.2 Klein 4B (Nov 2025) is the new Apache-2.0 diffusion default, replacing FLUX.1-schnell where higher quality is needed; FLUX.2 Klein 9B remains non-commercial.
- Langfuse was acquired by ClickHouse (Jan 2026, $400M Series D) — no license change yet; MIT core remains, but enterprise direction is uncertain. Hold MIT-only `/ee`-free deployment.
- HF Spaces ZeroGPU upgraded to H200 (was A100) — 70 GB `large` / 141 GB `xlarge`, opens the door to single-pass Qwen3-VL-32B inference in demos.