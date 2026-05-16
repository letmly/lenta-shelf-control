# lenta-autoresearch (Kaggle backend)

Autonomous experiment loop. Каждый experiment = push новой версии Kaggle kernel, ожидание, скачивание результатов, лог.

**Все вычисления на Kaggle GPU**, локальная машина user'а только редактирует код и пушит.

## Files

- `kaggle_experiment.ipynb` — **агент модифицирует**. Notebook который Kaggle запустит.
- `_kernel/kernel-metadata.json` — fixed Kaggle kernel config (id, datasets, GPU).
- `results.tsv` — лог.
- `runs/<commit>/` — output папка после каждого experiment.
- `program.md` — этот файл.

## Datasets (already uploaded to Kaggle)

- `letmlytt/lenta-labeled` — 5 видео + GT csv (440 MB)
- `letmlytt/lenta-v10-pred` — наши v10 pred csvs (для detector-bbox экспов)

## Метрика

Каждый kernel выводит в конце:
```
main_metric: 0.NNNNNN     ← главная (qualified_pct на subset)
tie_break:   0.NNNNNN     ← mean_score
matched:     X/71 (0.NNNN)
qualified:   X/71
n_pred:      N
```

Higher main_metric is better. Subset = 26_12-20 (71 GT-ценник).

## Setup (one-time)

1. Branch: `git checkout -b autoresearch/may16` (если не существует).
2. Verify Kaggle CLI: `D:/.dev/Products/hacks/lenta/.venv/Scripts/kaggle.exe kernels list -m | head -3` should list user's kernels.
3. Initial `kaggle_experiment.ipynb` уже создан с baseline (RapidOCR + upscale 3 + YOLO conf 0.1).
4. Init `results.tsv`:
   ```
   commit	main_metric	tie_break	matched_pct	status	description
   ```

## Loop (KEEP RUNNING)

```python
LOOP FOREVER:
  1. Read results.tsv → знаешь текущий best.
  2. Decide ONE change. Edit kaggle_experiment.ipynb.
  3. git add lenta_autoresearch/kaggle_experiment.ipynb
     git commit -m "exp: <description>"
  4. Push kernel:
     cd lenta_autoresearch/_kernel
     cp ../kaggle_experiment.ipynb .
     D:/.dev/Products/hacks/lenta/.venv/Scripts/kaggle.exe kernels push
  5. WAIT (polling):
     until status COMPLETE | ERROR | CANCEL:
       sleep 60
       kaggle kernels status letmlytt/lenta-autoresearch
  6. Download output:
     mkdir runs/<commit_short>
     kaggle kernels output letmlytt/lenta-autoresearch -p runs/<commit_short>
  7. Parse main_metric + tie_break from output.json or log:
     grep "^main_metric:\|^tie_break:" runs/<commit>/lenta-autoresearch.log
  8. Append row to results.tsv:
     <commit>\t<main_metric>\t<tie_break>\t<matched_pct>\t<status>\t<description>
     - status: keep если main_metric > prev best (tie-break by tie_break)
              discard если хуже/равно
              crash если grep пусто
  9. If keep → commit advances.
     If discard → git reset --hard HEAD~1.
 10. Repeat.
```

## What you CAN modify in `kaggle_experiment.ipynb`

- Knobs at top (UPSCALE, OCR_ENGINE, YOLO_CONF, etc.)
- Preprocessing pipeline
- OCR engine choice (rapidocr / easyocr / tesseract / Qwen-VL — последний нужен GPU)
- Parser tricks
- Post-processing на pred CSV

## What you CANNOT modify

- `prepare_lenta.py` — eval ground truth
- `_kernel/kernel-metadata.json` (только если меняешь datasets/GPU explicitly)
- `program.md`

## Time budget per experiment

~10-20 минут (зависит от OCR):
- RapidOCR ~10 мин
- EasyOCR ~12 мин (CPU on Kaggle)
- Tesseract ~8 мин
- Qwen-VL-3B ~15 мин (GPU)

Kaggle ограничение: 30 GPU-часов/неделя. На batch session 2 GPU параллельно. **Не пушь параллельно >2 экспериментов.**

## Ideas (run combos, not just one knob)

### OCR engine swap
- `OCR_ENGINE = "easyocr"` — +5pp vs RapidOCR (доказано на Kaggle прежним benchmark'ом)
- `OCR_ENGINE = "qwen25_vl_3b"` — +20pp но 1s/tag

### Preprocessing
- `UPSCALE = 4.0` (vs 3.0) — больше пикселей для VLM
- `USE_CLAHE = True` — контраст
- `USE_DENOISE = True` — шум

### Combos
- Qwen-VL + UPSCALE 4 + CLAHE
- EasyOCR + DEFAULT_NET_FOR_TEXT
- RapidOCR + post-process ALWAYS_NET для text

### Detection
- `YOLO_CONF = 0.05` (либеральнее)
- `BBOX_PADDING = 0.50` (больше контекста для OCR)

### Parser
- `DEFAULT_NET_FOR_TEXT_FIELDS = True`

## Never stop

Loop runs until user stops manually. Don't ask permission. If stuck → think harder, try radical combos.
