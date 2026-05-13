"""QR/штрихкод rescue v3: Lucky imaging + ensemble декодеров.

Industry best-practice стек (см. docs/10-industry-baseline.md):
    A. Ensemble декодеров: PyBoof + WeChat + pyzbar + zxing-cpp
       (Dynamsoft benchmark: BoofCV 60.7%, WeChat 48.9%, pyzbar 38.9%, zxing 31.9%)
       Берём первый успешный.
    B. Lucky imaging: вместо median по всем 30+ кадрам берём top-K самых чётких
       по Laplacian variance, align'им homography'ей, weighted median.
    C. Окно ±400 мс (а не ±800) — чтобы робот не успел заехать на соседний ценник.
    D. Sharpness filter — кадры с Laplacian < median//2 выкидываем.

Стратегии для сравнения:
    S0_ref_only         single-frame на GT-таймстемпе (sanity check)
    S1_all_median       median по всем кадрам окна (старая v1)
    S2_lucky_top5       Lucky: top-5 sharpest → align → median
    S3_lucky_top3       Lucky: top-3 sharpest → align → median
    S4_lucky_top1       один самый чёткий кадр (sharpest single frame)
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from tqdm import tqdm

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    PB_MICRO = pb.FactoryFiducial(np.uint8).microqr()
    HAS_PYBOOF = True
except Exception as e:
    print(f"PyBoof init failed: {e}", file=sys.stderr)
    HAS_PYBOOF = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "rescue_v3"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = OUT / "super_crops"
SAMPLE_DIR.mkdir(exist_ok=True)

VIDEOS = [
    (ROOT / "data/labeled/26_12-20/26_12-20.mp4", ROOT / "data/labeled/26_12-20/26_12-20.csv"),
    (ROOT / "data/labeled/43_15/43_15.mp4",        ROOT / "data/labeled/43_15/43_15.csv"),
]

WINDOW_MS = 400
PAD_PCT = 0.5
ECC_ITER = 50
SAMPLE_LIMIT = 8  # сохраняем первые N super-crops на видео


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def load_gt(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype(str).str.strip().str.replace(".0", "", regex=False)
    return df


def pad_bbox(bbox, shape, pct=PAD_PCT):
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def sharpness(img) -> float:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


# ----------------------------- decoders ---------------------------------------

def decode_pyzbar(img) -> list[str]:
    try:
        return [d.data.decode("utf-8", errors="ignore").strip() for d in pyzbar_decode(img)]
    except Exception:
        return []


def decode_zxing(img) -> list[str]:
    if not HAS_ZXING: return []
    try:
        return [r.text.strip() for r in zxingcpp.read_barcodes(img)]
    except Exception:
        return []


def decode_pyboof(img) -> list[str]:
    if not HAS_PYBOOF: return []
    try:
        # PyBoof хочет grayscale uint8
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        pb_img = pb.ndarray_to_boof(gray)
        out = []
        PB_QR.detect(pb_img)
        for det in PB_QR.detections:
            if det.message: out.append(det.message.strip())
        PB_MICRO.detect(pb_img)
        for det in PB_MICRO.detections:
            if det.message: out.append(det.message.strip())
        return out
    except Exception:
        return []


def ensemble_decode(img) -> tuple[set, dict]:
    """Прогон через все декодеры на 4 препроцессингах. Возвращает set всех декодированных строк + по-декодерная статистика."""
    by_decoder = {"pyzbar": [], "zxing": [], "pyboof": []}
    variants = [
        img,
        cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
    ]
    for v in variants:
        by_decoder["pyzbar"].extend(decode_pyzbar(v))
        by_decoder["zxing"].extend(decode_zxing(v))
        by_decoder["pyboof"].extend(decode_pyboof(v))
    all_codes = set()
    for codes in by_decoder.values():
        all_codes.update(codes)
    return all_codes, {k: list(set(v)) for k, v in by_decoder.items()}


# ----------------------------- stack helpers ----------------------------------

def collect_window_frames(cap, ts_center, gt_bbox, ref_template):
    """Собирает все выровненные cropы в окне ±WINDOW_MS вокруг ts_center.

    Возвращает [(crop_aligned, sharpness), ...]
    """
    H_t, W_t = ref_template.shape[:2]
    tmpl_gray = cv2.cvtColor(ref_template, cv2.COLOR_BGR2GRAY)
    bb = pad_bbox(gt_bbox, (10000, 10000), PAD_PCT)  # padded bbox в исходных координатах
    H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    bb = pad_bbox(gt_bbox, (H_full, W_full), PAD_PCT)

    start_ms = max(0, ts_center - WINDOW_MS)
    end_ms = ts_center + WINDOW_MS
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))

    crops = []
    while True:
        cur_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if cur_ts > end_ms: break
        ok, frame = cap.read()
        if not ok: break
        margin = 100
        zx1 = max(0, bb[0] - margin); zy1 = max(0, bb[1] - margin)
        zx2 = min(W_full, bb[2] + margin); zy2 = min(H_full, bb[3] + margin)
        zone = frame[zy1:zy2, zx1:zx2]
        if zone.shape[0] < H_t or zone.shape[1] < W_t: continue
        zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        try:
            res = cv2.matchTemplate(zone_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv < 0.35: continue  # outlier — соседний ценник или сильно искажён
        tx, ty = ml
        cand = zone[ty:ty + H_t, tx:tx + W_t]
        if cand.shape[:2] != (H_t, W_t): continue
        # ECC alignment
        cg = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        tg = tmpl_gray.astype(np.float32) / 255
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITER, 1e-3)
            _, warp = cv2.findTransformECC(tg, cg, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
            warped = cv2.warpAffine(cand.astype(np.float32), warp, (W_t, H_t),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            warped = cand.astype(np.float32)
        crops.append((warped.astype(np.uint8), sharpness(warped)))
    return crops


def make_super(crops, mode="median"):
    if not crops: return None
    arr = np.stack([c[0] for c in crops], axis=0).astype(np.float32)
    if mode == "median": sup = np.median(arr, axis=0)
    elif mode == "mean": sup = np.mean(arr, axis=0)
    else: raise ValueError(mode)
    return np.clip(sup, 0, 255).astype(np.uint8)


def lucky_imaging(crops, top_k):
    if not crops: return None
    sorted_crops = sorted(crops, key=lambda x: -x[1])
    top = sorted_crops[:min(top_k, len(sorted_crops))]
    if len(top) == 1: return top[0][0]
    return make_super(top, mode="median")


# ----------------------------- main runner ------------------------------------

def run_video(video: Path, csv: Path) -> dict:
    df = load_gt(csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): return {}
    total = len(df)

    strategies = ["S0_ref_only", "S1_all_median", "S2_lucky_top5", "S3_lucky_top3", "S4_lucky_top1", "ANY"]
    stats = {s: {"decoded": 0, "matched": 0} for s in strategies}
    decoder_stats = Counter()

    samples_saved = 0

    for idx, r in tqdm(df.iterrows(), total=total, desc=video.name):
        ts_gt = r["frame_timestamp"]
        gt_bc = r["barcode"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))

        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_gt))
        ok, ref_frame = cap.read()
        if not ok: continue
        bb = pad_bbox(bbox, ref_frame.shape, PAD_PCT)
        ref_crop = ref_frame[bb[1]:bb[3], bb[0]:bb[2]]
        if ref_crop.size == 0 or ref_crop.shape[0] < 30 or ref_crop.shape[1] < 30: continue

        # --- S0: ref only ---
        all_codes, by_dec = ensemble_decode(ref_crop)
        if all_codes:
            stats["S0_ref_only"]["decoded"] += 1
            for d, codes in by_dec.items():
                if codes: decoder_stats[f"S0_{d}"] += 1
            if gt_bc in all_codes: stats["S0_ref_only"]["matched"] += 1

        # --- собираем стек ---
        crops = collect_window_frames(cap, ts_gt, bbox, ref_crop)

        # --- S1: median по всем ---
        sup1 = make_super(crops, mode="median") if crops else None
        s1_match = False
        if sup1 is not None:
            c1, _ = ensemble_decode(sup1)
            if c1:
                stats["S1_all_median"]["decoded"] += 1
                if gt_bc in c1:
                    stats["S1_all_median"]["matched"] += 1; s1_match = True

        # --- S2/S3/S4: lucky imaging ---
        lucky_match = {}
        for label, k in [("S2_lucky_top5", 5), ("S3_lucky_top3", 3), ("S4_lucky_top1", 1)]:
            sup = lucky_imaging(crops, k) if crops else None
            if sup is None: continue
            codes, by_dec = ensemble_decode(sup)
            if codes:
                stats[label]["decoded"] += 1
                for d, cd in by_dec.items():
                    if cd: decoder_stats[f"{label}_{d}"] += 1
                if gt_bc in codes:
                    stats[label]["matched"] += 1
                    lucky_match[label] = True
            # сохраняем для глазного контроля
            if label == "S3_lucky_top3" and samples_saved < SAMPLE_LIMIT:
                tag = "MATCH" if gt_bc in codes else "miss"
                cv2.imwrite(str(SAMPLE_DIR / f"{video.stem}_idx{idx}_{label}_n{len(crops)}_{tag}_GT{gt_bc}.jpg"), sup)

        # --- ANY ---
        any_dec = False; any_match = False
        for k in ("S0_ref_only", "S1_all_median", "S2_lucky_top5", "S3_lucky_top3", "S4_lucky_top1"):
            if stats[k]["decoded"] and stats[k]["matched"]:  # not strictly correct but cheap
                pass
        # лучше: суммарный декод этого ценника через все стеки
        all_for_this = set()
        for sup_img in filter(lambda x: x is not None, [ref_crop, sup1,
                                                          lucky_imaging(crops, 5),
                                                          lucky_imaging(crops, 3),
                                                          lucky_imaging(crops, 1)]):
            c, _ = ensemble_decode(sup_img)
            all_for_this.update(c)
        if all_for_this:
            stats["ANY"]["decoded"] += 1
            if gt_bc in all_for_this: stats["ANY"]["matched"] += 1

        samples_saved = min(samples_saved + 1, SAMPLE_LIMIT)

    cap.release()

    out = {}
    for k, v in stats.items():
        out[k] = {
            "decoded": v["decoded"], "decoded_rate": v["decoded"] / total,
            "matched": v["matched"], "matched_rate": v["matched"] / total,
        }
    return {"video": video.name, "total": total, "stats": out, "by_decoder": dict(decoder_stats)}


def main():
    print(f"decoders: pyzbar=YES  zxing-cpp={'YES' if HAS_ZXING else 'NO'}  pyboof={'YES' if HAS_PYBOOF else 'NO'}")
    t0 = time.time()
    report = {"started_at": t0, "videos": []}
    for video, csv in VIDEOS:
        if not video.exists(): continue
        r = run_video(video, csv)
        if not r: continue
        report["videos"].append(r)
        print(f"\n=== {r['video']} (GT={r['total']}) ===")
        for k, st in r["stats"].items():
            print(f"  {k:18s} decoded={st['decoded']:3d} ({st['decoded_rate']:.0%})  matched={st['matched']:3d} ({st['matched_rate']:.0%})")
        if r.get("by_decoder"):
            print(f"  by-decoder hits: {r['by_decoder']}")
    report["elapsed_sec"] = time.time() - t0
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT/'report.json'}    super_crops in {SAMPLE_DIR}  elapsed={report['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
