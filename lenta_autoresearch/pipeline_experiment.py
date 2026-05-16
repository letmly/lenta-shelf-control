"""Pipeline experiment file — AGENT EDITS THIS ONE.

Запуск: cd lenta/ && uv run lenta_autoresearch/pipeline_experiment.py > run.log 2>&1
Метрика: maximize qualified_pct (главная), tie-break by mean_score.

==================================================================
EXPERIMENT KNOBS — здесь крутить.
==================================================================
"""
from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "lenta_autoresearch"))

from prepare_lenta import SUBSET_VIDEO, SUBSET_NAME, evaluate_pipeline, print_summary

# === KNOBS (агент крутит) ===

# OCR engine: 'rapidocr' | 'easyocr' | 'tesseract'
OCR_ENGINE = "rapidocr"

# Preprocessing на crop перед OCR
UPSCALE = 3.0            # 1.0 = no upscale; 2-4 типично
USE_SHARPEN = False      # UnsharpMask (Kaggle показал что вредит на VLM)
USE_CLAHE = False        # CLAHE contrast equalization
USE_DENOISE = False      # fastNlMeansDenoising

# Detection
YOLO_CONF = 0.10         # 0.05/0.10/0.15/0.20

# Sampling
FPS_SAMPLE = 3           # 1/2/3/5

# Parser tricks
DEFAULT_NET_FOR_TEXT_FIELDS = False  # текст-поля → 'нет' если пусто

# === END KNOBS ===


# === Preprocessing helper ===
def preprocess(img):
    out = img
    if USE_DENOISE and out.ndim == 3:
        out = cv2.fastNlMeansDenoisingColored(out, None, 7, 7, 5, 11)
    if USE_CLAHE and out.ndim == 3:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if USE_SHARPEN:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(out, -1, kernel)
    return out


# === OCR wrapper providing RapidOCR-compatible interface ===
class OCRWrapper:
    """Унифицированная обёртка: __call__(img, **kw) → (result, time)
    где result = [[bbox, text, confidence], ...] (как у RapidOCR)
    """
    def __init__(self, engine: str):
        self.engine = engine
        if engine == "rapidocr":
            from rapidocr_onnxruntime import RapidOCR
            self._ocr = RapidOCR()
        elif engine == "easyocr":
            import easyocr
            self._ocr = easyocr.Reader(['ru', 'en'], gpu=False, verbose=False)
        elif engine == "tesseract":
            import pytesseract
            self._ocr = pytesseract
        else:
            raise ValueError(f"Unknown OCR_ENGINE: {engine}")

    def __call__(self, img, **kwargs):
        img = preprocess(img)
        if self.engine == "rapidocr":
            try: return self._ocr(img, **kwargs)
            except TypeError: return self._ocr(img)
        if self.engine == "easyocr":
            res = self._ocr.readtext(img, detail=1, paragraph=False)
            # convert to RapidOCR format: [[bbox, text, conf], ...]
            out = [[list(r[0]), r[1], float(r[2])] for r in res]
            return out, 0.0
        if self.engine == "tesseract":
            try:
                txt = self._ocr.image_to_string(img, lang="rus+eng")
            except Exception:
                return None, 0.0
            lines = [l.strip() for l in txt.split("\n") if l.strip()]
            out = [[[0, 0, 0, 0], l, 0.5] for l in lines]
            return out, 0.0


# === Main ===
def run():
    print(f"=== EXPERIMENT ===", flush=True)
    print(f"OCR_ENGINE={OCR_ENGINE} UPSCALE={UPSCALE} SHARPEN={USE_SHARPEN} "
          f"CLAHE={USE_CLAHE} DENOISE={USE_DENOISE}", flush=True)
    print(f"YOLO_CONF={YOLO_CONF} FPS_SAMPLE={FPS_SAMPLE} DEFAULT_NET={DEFAULT_NET_FOR_TEXT_FIELDS}",
          flush=True)

    # Monkey-patch pipeline_v10's OCR class before calling
    import pipeline_v10
    pipeline_v10.RapidOCR = lambda: OCRWrapper(OCR_ENGINE)

    out_dir = ROOT / "lenta_autoresearch" / "_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{SUBSET_NAME}_exp.csv"

    from pipeline_v10 import process_video
    df_pred = process_video(
        SUBSET_VIDEO,
        fps_sample=FPS_SAMPLE,
        output_csv=out_csv,
        filename_in_csv=f"{SUBSET_NAME}.mp4",
        upscale=UPSCALE,
        yolo_conf=YOLO_CONF,
    )

    if df_pred is None or len(df_pred) == 0:
        print("ERROR: empty pred", flush=True)
        return

    if DEFAULT_NET_FOR_TEXT_FIELDS:
        df = pd.read_csv(out_csv, encoding="utf-8")
        for f in ["product_name", "additional_info", "code", "special_symbols", "print_datetime"]:
            if f in df.columns:
                df[f] = df[f].fillna("нет").apply(
                    lambda x: "нет" if str(x).strip() in ("", "nan", "None") else x
                )
        df.to_csv(out_csv, index=False, encoding="utf-8")

    metrics = evaluate_pipeline(out_csv)
    print_summary(metrics)


if __name__ == "__main__":
    run()
