"""Stacking v2 (fast): один последовательный проход по видео + параллелизация.

Оптимизации vs v1:
    1. SEQUENTIAL READ. Читаем видео слева-направо через .read(), без seek.
       Видео-seek очень дорог; sequential read 10-20x быстрее.
    2. ONE PASS. На каждом кадре проверяем, попадает ли его timestamp в окно
       какого-то ценника. Если да — сразу вырезаем padded crop и кладём в список
       этого ценника. Память: только нужные cropы, не полные кадры.
    3. PARALLEL по ценникам. Сборка стека + ECC + median + decode параллелится
       multiprocessing.Pool по числу ядер CPU.
    4. Меньшее окно: ±800 мс шаг 100 мс → ~17 кадров на ценник
       (вместо 61 в v1; для median'а 17 кадров уже даёт большой буст).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "stacking_fast"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = OUT / "super_crops"
SAMPLE_DIR.mkdir(exist_ok=True)

VIDEOS = [
    (ROOT / "data/labeled/26_12-20/26_12-20.mp4", ROOT / "data/labeled/26_12-20/26_12-20.csv"),
    (ROOT / "data/labeled/43_15/43_15.mp4",        ROOT / "data/labeled/43_15/43_15.csv"),
]

WINDOW_MS = 800
PAD_PCT = 0.5
ECC_ITER = 50
ECC_EPS = 1e-3
SAMPLE_LIMIT = 10  # сохраняем первые N super-cropов на видео


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


def pad(b, shape, pct=PAD_PCT):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def collect_crops_one_pass(video: Path, gt: pd.DataFrame):
    """ОДИН проход по видео. Возвращает {gt_idx: [list of crops]} + ref bboxes.

    Для каждого кадра, если |frame_ts - gt_ts| <= WINDOW_MS для какого-то ценника,
    вырезаем padded crop вокруг bbox этого ценника.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {}, {}
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # подготовим pad-bbox для каждого ценника на основе референсного кадра (берём из GT — bbox уже есть)
    ref_bboxes = {}
    ref_crops_template = {}
    for i, r in gt.iterrows():
        bb = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
        # full padded bbox + большая margin для поиска через matchTemplate
        ref_bboxes[i] = bb

    crops_by_tag = {i: [] for i in gt.index}
    gt_ts_list = gt["frame_timestamp"].tolist()
    H_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    pbar = tqdm(total=total_frames, desc=f"read {video.name}")
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        ts_ms = (frame_idx + 1) * (1000.0 / fps) if fps else frame_idx
        # для каждого ценника проверяем, попадает ли этот ts в его окно
        for i, gt_ts in enumerate(gt_ts_list):
            if abs(ts_ms - gt_ts) > WINDOW_MS:
                continue
            bb = ref_bboxes[i]
            # padded extraction zone с margin для поиска
            margin = 80
            x1, y1, x2, y2 = pad(bb, frame.shape, PAD_PCT)
            ex1 = max(0, x1 - margin); ey1 = max(0, y1 - margin)
            ex2 = min(W_full, x2 + margin); ey2 = min(H_full, y2 + margin)
            zone = frame[ey1:ey2, ex1:ex2].copy()
            crops_by_tag[i].append({"zone": zone, "ref_padded_bbox": (x1, y1, x2, y2),
                                     "zone_origin": (ex1, ey1), "frame_ts": ts_ms})
        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    return crops_by_tag, ref_bboxes


def align_ecc(template_gray, target_gray):
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITER, ECC_EPS)
        _, warp = cv2.findTransformECC(template_gray, target_gray, warp, cv2.MOTION_TRANSLATION, criteria, None, 5)
        return warp, True
    except cv2.error:
        return None, False


def process_one_tag(args):
    """Обработка одного ценника: собрать стек, выровнять, median, decode."""
    gt_idx, crops_data, gt_barcode = args
    if not crops_data:
        return gt_idx, {"stack_size": 0, "decoded": [], "matched": False, "super_crop": None}

    # template — это padded crop эталонного кадра (тот, чей frame_ts ближе к GT)
    # выбираем кадр с самой высокой sharpness как template
    def sharpness(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    # извлекаем padded crops из zones
    extracted = []
    for c in crops_data:
        x1, y1, x2, y2 = c["ref_padded_bbox"]
        zx, zy = c["zone_origin"]
        # координаты bbox внутри zone
        lx1, ly1 = x1 - zx, y1 - zy
        lx2, ly2 = x2 - zx, y2 - zy
        h_zone, w_zone = c["zone"].shape[:2]
        lx1 = max(0, lx1); ly1 = max(0, ly1)
        lx2 = min(w_zone, lx2); ly2 = min(h_zone, ly2)
        crop = c["zone"][ly1:ly2, lx1:lx2]
        if crop.size > 0:
            extracted.append({"crop": crop, "zone": c["zone"], "ref_pad": (x1, y1, x2, y2), "origin": (zx, zy)})
    if not extracted:
        return gt_idx, {"stack_size": 0, "decoded": [], "matched": False, "super_crop": None}

    # выбираем sharpest как template
    best = max(extracted, key=lambda x: sharpness(x["crop"]))
    template = best["crop"]
    H_t, W_t = template.shape[:2]
    if H_t < 50 or W_t < 50:
        return gt_idx, {"stack_size": 0, "decoded": [], "matched": False, "super_crop": None}

    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # для каждого crop'а — ищем в zone через matchTemplate, затем ECC, кладём в стек
    stack = []
    for e in extracted:
        zone = e["zone"]
        if zone.shape[0] < H_t or zone.shape[1] < W_t:
            continue
        zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        try:
            res = cv2.matchTemplate(zone_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv < 0.25:
            continue
        tx, ty = ml
        cand = zone[ty:ty + H_t, tx:tx + W_t]
        if cand.shape[:2] != (H_t, W_t):
            continue
        # ECC
        c_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        t_gray = tmpl_gray.astype(np.float32) / 255
        warp, ok = align_ecc(t_gray, c_gray)
        if ok:
            warped = cv2.warpAffine(cand.astype(np.float32), warp, (W_t, H_t),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            warped = cand.astype(np.float32)
        stack.append(warped)

    if len(stack) < 3:
        return gt_idx, {"stack_size": len(stack), "decoded": [], "matched": False, "super_crop": None}

    arr = np.stack(stack, axis=0)
    super_crop = np.clip(np.median(arr, axis=0), 0, 255).astype(np.uint8)

    # decode (несколько препроцессингов)
    decoded = set()
    variants = [
        super_crop,
        cv2.resize(super_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
        cv2.cvtColor(super_crop, cv2.COLOR_BGR2GRAY),
        cv2.resize(cv2.cvtColor(super_crop, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
    ]
    for v in variants:
        try:
            for d in pyzbar_decode(v):
                decoded.add(d.data.decode("utf-8", errors="ignore").strip())
        except Exception: pass
        if HAS_ZXING:
            try:
                for r in zxingcpp.read_barcodes(v):
                    decoded.add(r.text.strip())
            except Exception: pass
    matched = gt_barcode in decoded
    return gt_idx, {"stack_size": len(stack), "decoded": list(decoded), "matched": matched, "super_crop": super_crop}


def run_video(video: Path, csv: Path) -> dict:
    gt = load_gt(csv)
    total = len(gt)
    print(f"\n=== {video.name} (GT={total}) ===")
    t0 = time.time()
    crops_by_tag, _ = collect_crops_one_pass(video, gt)
    t_read = time.time() - t0
    print(f"  sequential read: {t_read:.1f}s")

    # параллельная обработка
    args = [(i, crops_by_tag[i], gt.loc[i, "barcode"]) for i in gt.index]
    n_workers = max(1, mp.cpu_count() - 1)
    t1 = time.time()
    results = {}
    with mp.Pool(n_workers) as pool:
        for gt_idx, res in tqdm(pool.imap_unordered(process_one_tag, args), total=len(args), desc=f"stack {video.name}"):
            results[gt_idx] = res
    t_proc = time.time() - t1
    print(f"  parallel stack+decode: {t_proc:.1f}s on {n_workers} workers")

    # статистика
    decoded_cnt = sum(1 for r in results.values() if r["decoded"])
    matched_cnt = sum(1 for r in results.values() if r["matched"])
    # save first N super_crops
    saved = 0
    for i, r in results.items():
        if saved >= SAMPLE_LIMIT: break
        if r.get("super_crop") is not None:
            gt_bc = gt.loc[i, "barcode"]
            tag = "MATCH" if r["matched"] else "miss"
            path = SAMPLE_DIR / f"{video.stem}_idx{i}_n{r['stack_size']}_{tag}_GT{gt_bc}.jpg"
            cv2.imwrite(str(path), r["super_crop"])
            saved += 1

    return {
        "video": video.name,
        "total_gt": total,
        "read_time_s": t_read,
        "process_time_s": t_proc,
        "decoded": decoded_cnt,
        "decoded_rate": decoded_cnt / total,
        "matched": matched_cnt,
        "matched_rate": matched_cnt / total,
        "avg_stack_size": float(np.mean([r["stack_size"] for r in results.values() if r["stack_size"] > 0]))
                          if any(r["stack_size"] > 0 for r in results.values()) else 0,
    }


def main():
    print(f"zxing-cpp: {'YES' if HAS_ZXING else 'NO'}")
    print(f"CPU cores available: {mp.cpu_count()}")
    report = {"started_at": time.time(), "videos": []}
    for video, csv in VIDEOS:
        if not video.exists(): continue
        r = run_video(video, csv)
        report["videos"].append(r)
        print(f"\n--- summary {r['video']} ---")
        print(f"  decoded: {r['decoded']}/{r['total_gt']} ({r['decoded_rate']:.0%})")
        print(f"  matched: {r['matched']}/{r['total_gt']} ({r['matched_rate']:.0%})")
        print(f"  avg stack size: {r['avg_stack_size']:.0f}")
    report["elapsed_sec"] = time.time() - report["started_at"]
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT/'report.json'}  super_crops in {SAMPLE_DIR}  elapsed={report['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
