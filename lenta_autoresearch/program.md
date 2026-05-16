# lenta-autoresearch

This is an autonomous experiment loop for Lenta shelf-control pipeline. You are an AI researcher whose job is to maximize the metric on the subset by tweaking the pipeline in `pipeline_experiment.py`. You run experiments in a loop until manually stopped.

## Setup (once per session)

1. **Run tag**: Use today's date, e.g. `may16`. Branch `autoresearch/<tag>` must not exist yet.
2. **Create branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read in-scope files**:
   - `lenta_autoresearch/prepare_lenta.py` — fixed eval harness. **DO NOT MODIFY**.
   - `lenta_autoresearch/pipeline_experiment.py` — the file you edit.
   - `scripts/pipeline_v10.py` — current pipeline implementation (reference only, do not modify).
   - `scripts/evaluate.py` — evaluator (reference only).
4. **Verify data**: `data/labeled/26_12-20/26_12-20.mp4` and `.csv` exist locally.
5. **Initialize results.tsv**: Create `lenta_autoresearch/results.tsv` with header:
   ```
   commit	qualified_pct	mean_score	matched_pct	status	description
   ```
6. **Baseline run**: Run `pipeline_experiment.py` as-is to establish baseline metric. Commit it as baseline before any changes.

## What you CAN modify

Only `lenta_autoresearch/pipeline_experiment.py`. Specifically:
- Knobs at the top (UPSCALE, YOLO_CONF, OCR_ENGINE, USE_SHARPEN, etc.)
- `preprocess()` function (try denoise/sharpen/CLAHE combos)
- `OCRWrapper` class (add new OCR engines, try better prompts)
- Post-processing logic in `run()` (parser tricks, field defaults)

## What you CANNOT modify

- `prepare_lenta.py` — eval is ground truth
- `scripts/pipeline_v10.py` or other files in scripts/
- Anything outside lenta_autoresearch/

## Metric

**Primary**: `main_metric` (= qualified_pct, % ценников ≥80% полей правильно). Higher is better.
**Tie-break**: `tie_break` (= mean_score, средний % полей).

Extract from log:
```
grep "^main_metric:\|^tie_break:" run.log
```

## The experiment loop

Each experiment runs on subset of **71 ценников из 26_12-20**. Time budget: **5-8 минут** per run (RapidOCR ~5min, EasyOCR ~8min, Tesseract ~6min на CPU).

LOOP FOREVER:

1. Look at current git commit. Read `results.tsv` to see what was tried.
2. Decide on ONE change to try. Edit `pipeline_experiment.py`.
3. `git add pipeline_experiment.py && git commit -m "exp: <description>"`
4. Run: `D:/.dev/Products/hacks/lenta/.venv/Scripts/python.exe lenta_autoresearch/pipeline_experiment.py > lenta_autoresearch/run.log 2>&1`
5. Get metrics: `grep "^main_metric:\|^tie_break:\|^matched:" lenta_autoresearch/run.log`
6. If grep empty/crash → `tail -50 lenta_autoresearch/run.log` to read error.
7. Get short git hash: `git rev-parse --short HEAD`
8. Append to results.tsv (tab-separated):
   ```
   <hash>	<qualified_pct>	<mean_score>	<matched_pct>	<status>	<description>
   ```
   - status: `keep` (improved), `discard` (worse/same), `crash`
9. If improved (qualified_pct > best, or equal qual + better mean_score):
   - **keep**: leave the commit, this becomes new baseline
10. If not improved:
    - **discard**: `git reset --hard HEAD~1` to undo
11. Repeat from step 1.

**Timeout**: each experiment ≤ 12 минут. If exceeds — `pkill -f pipeline_experiment.py`, log crash.

**Never stop**: human may be asleep. Keep iterating until manually stopped.

## Ideas to explore (don't just iterate one knob; combine)

### OCR engine swap (big impact)
- `OCR_ENGINE = "easyocr"` (~+5pp on Cyrillic from chat tests)
- `OCR_ENGINE = "tesseract"` (variance high, sometimes wins)

### Preprocessing
- `UPSCALE = 4.0` или 2.0 (sweep)
- `USE_CLAHE = True` для контраста
- `USE_DENOISE = True` для шумных кропов
- `USE_SHARPEN = True` (Kaggle показал что вредит на VLM — но для classical может OK)
- Combos

### Detection
- `YOLO_CONF = 0.05` (более liberal)
- `YOLO_CONF = 0.2` (более strict)

### Sampling
- `FPS_SAMPLE = 1` (медленнее, больше шансов)
- `FPS_SAMPLE = 5` (быстрее, меньше детекций)

### Parser tricks
- `DEFAULT_NET_FOR_TEXT_FIELDS = True` (fills empty product_name='нет')

### Комбо (high-value)
- EasyOCR + UPSCALE 4 + CLAHE
- Tesseract + DENOISE
- RapidOCR + DEFAULT_NET + CLAHE

## Output format requirements

The script MUST end with:
```
---
matched:       X/71 (0.NNNN)
mean_score:    0.NNNN
qualified:     X/71 (0.NNNN)
n_pred:        N
main_metric:   0.NNNNNN
tie_break:     0.NNNNNN
```

If you don't see this format → run crashed, log as `crash` and revert.

## Git discipline

- `pipeline_experiment.py` IS tracked.
- `run.log` and `_runs/` and `results.tsv` are NOT tracked (leave them in working dir).
- Use short descriptive commit messages: `exp: easyocr + upscale 4 + clahe`.

## Going further

If you're stuck after 10+ experiments without improvement:
- Look at `_runs/26_12-20_exp.csv` to see WHICH fields are failing.
- Re-read `scripts/evaluate.py` to understand what compare_fields counts.
- Try radical changes (TODO: write your own OCR call, use VLM via API, etc.)
- BUT don't fundamentally redesign — small incremental changes are best.

Now go.
