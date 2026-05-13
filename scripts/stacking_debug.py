"""Визуальный debug стейкинга на 2 ценниках.

Что делает:
    Для каждого из ВЫБРАННЫХ ценников:
        1. Читает все кадры в окне ±800 мс
        2. Для каждого ищет ценник через matchTemplate + ECC
        3. Сохраняет КАЖДЫЙ найденный crop (всё что попало в стек) — глазами видно
        4. Сохраняет grid-изображение со всеми crop'ами вместе
        5. Сохраняет median super-crop
        6. Декодирует super-crop через pyzbar/zxing — печатает результат

Это позволяет глазами проверить:
    a) сколько копий ценника удалось найти
    b) как они различаются (углы, blur)
    c) во что они слились median'ом
    d) декодируется ли super-crop
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode

try:
    import zxingcpp
    HAS_ZXING = True
except ImportError:
    HAS_ZXING = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "stacking_debug"
OUT.mkdir(parents=True, exist_ok=True)

# Берём 2 ценника из 43_15 (короткое видео = быстрее)
SELECTED = [
    ("data/labeled/43_15/43_15.mp4", "data/labeled/43_15/43_15.csv", 0),
    ("data/labeled/43_15/43_15.mp4", "data/labeled/43_15/43_15.csv", 5),
]

WINDOW_MS = 800
PAD_PCT = 0.5
ECC_ITER = 50


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def pad(b, shape, pct=PAD_PCT):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def grid(images, cols=6, gap=4, bg=(40, 40, 40)):
    """Склейка списка изображений в один grid с подписями."""
    if not images: return None
    # нормализуем размер
    h_max = max(im.shape[0] for im in images)
    w_max = max(im.shape[1] for im in images)
    rows = (len(images) + cols - 1) // cols
    H = rows * (h_max + gap) + gap
    W = cols * (w_max + gap) + gap
    canvas = np.full((H, W, 3), bg, dtype=np.uint8)
    for k, img in enumerate(images):
        r = k // cols; c = k % cols
        y = gap + r * (h_max + gap)
        x = gap + c * (w_max + gap)
        canvas[y:y+img.shape[0], x:x+img.shape[1]] = img
    return canvas


def process(mp4_rel, csv_rel, gt_idx):
    video = ROOT / mp4_rel
    csv = ROOT / csv_rel
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype(str).str.strip().str.replace(".0", "", regex=False)

    if gt_idx >= len(df):
        print(f"  gt_idx={gt_idx} out of range ({len(df)})")
        return

    r = df.iloc[gt_idx]
    ts_gt = r["frame_timestamp"]
    bbox_gt = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
    gt_bc = r["barcode"]

    print(f"\n=== {video.name} gt_idx={gt_idx}  ts={ts_gt}ms  GT_barcode={gt_bc} ===")
    print(f"  bbox_gt={bbox_gt}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("  cannot open video"); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    # ----- эталонный кадр + template (padded crop) -----
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_gt))
    ok, ref_frame = cap.read()
    if not ok:
        print("  no ref frame"); return
    bb = pad(bbox_gt, ref_frame.shape, PAD_PCT)
    template = ref_frame[bb[1]:bb[3], bb[0]:bb[2]]
    H_t, W_t = template.shape[:2]
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # ----- сохранение полного эталонного кадра с обведённым bbox -----
    annotated_ref = ref_frame.copy()
    cv2.rectangle(annotated_ref, (bbox_gt[0], bbox_gt[1]), (bbox_gt[2], bbox_gt[3]), (0, 255, 0), 6)
    cv2.rectangle(annotated_ref, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 255), 4)
    cv2.putText(annotated_ref, f"GT ts={int(ts_gt)}ms  bc={gt_bc}",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
    cv2.imwrite(str(OUT / f"{video.stem}_idx{gt_idx}_00_ref_frame.jpg"),
                cv2.resize(annotated_ref, (1920, 1080)))

    # ----- проходим окно ±WINDOW_MS, ищем template -----
    # читаем все кадры в окне последовательно (без seek)
    start_ms = max(0, ts_gt - WINDOW_MS)
    end_ms = ts_gt + WINDOW_MS
    cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))
    found_crops = []         # сырые crop'ы из найденных позиций
    aligned_crops = []       # они же после ECC
    debug_full_frames = []   # уменьшенные кадры с обведёнными bbox для просмотра
    match_quality = []       # значения template-matching корреляции
    while True:
        cur_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
        if cur_ts > end_ms: break
        ok, frame = cap.read()
        if not ok: break
        # margin вокруг bb для поиска
        margin = 100
        h_f, w_f = frame.shape[:2]
        zx1 = max(0, bb[0] - margin); zy1 = max(0, bb[1] - margin)
        zx2 = min(w_f, bb[2] + margin); zy2 = min(h_f, bb[3] + margin)
        zone = frame[zy1:zy2, zx1:zx2]
        if zone.shape[0] < H_t or zone.shape[1] < W_t: continue
        zone_gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        try:
            res = cv2.matchTemplate(zone_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, mv, _, ml = cv2.minMaxLoc(res)
        match_quality.append(mv)
        if mv < 0.25:  # ценник не виден — пропускаем
            continue
        tx, ty = ml
        cand = zone[ty:ty + H_t, tx:tx + W_t]
        if cand.shape[:2] != (H_t, W_t): continue
        found_crops.append(cand)
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
        aligned_crops.append(warped.astype(np.uint8))
        # сохраним маленький full-frame с обведённым bbox-найденным-позиции
        found_bbox_abs = (zx1 + tx, zy1 + ty, zx1 + tx + W_t, zy1 + ty + H_t)
        ann = frame.copy()
        cv2.rectangle(ann, (found_bbox_abs[0], found_bbox_abs[1]), (found_bbox_abs[2], found_bbox_abs[3]),
                      (255, 0, 0), 6)
        cv2.putText(ann, f"ts={int(cur_ts)}ms  match={mv:.2f}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 0, 0), 3)
        debug_full_frames.append(cv2.resize(ann, (640, 360)))
    cap.release()

    print(f"  frames in window: {len(match_quality)}  passed match_th>=0.25: {len(found_crops)}")
    if match_quality:
        print(f"  match quality: min={min(match_quality):.2f} median={np.median(match_quality):.2f} max={max(match_quality):.2f}")

    if not found_crops:
        print("  no crops collected — nothing to do")
        return

    # ----- grid: сырые crop'ы -----
    g_raw = grid(found_crops, cols=8, gap=4, bg=(60, 60, 60))
    if g_raw is not None:
        cv2.imwrite(str(OUT / f"{video.stem}_idx{gt_idx}_01_raw_crops_grid.jpg"), g_raw)

    # ----- grid: выровненные crop'ы -----
    g_aligned = grid(aligned_crops, cols=8, gap=4, bg=(60, 60, 60))
    if g_aligned is not None:
        cv2.imwrite(str(OUT / f"{video.stem}_idx{gt_idx}_02_aligned_crops_grid.jpg"), g_aligned)

    # ----- grid: уменьшенные full frames -----
    g_full = grid(debug_full_frames, cols=4, gap=4, bg=(60, 60, 60))
    if g_full is not None:
        cv2.imwrite(str(OUT / f"{video.stem}_idx{gt_idx}_03_full_frames_grid.jpg"), g_full)

    # ----- median super-crop -----
    arr = np.stack(aligned_crops, axis=0).astype(np.float32)
    super_crop = np.clip(np.median(arr, axis=0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"{video.stem}_idx{gt_idx}_04_median_super.jpg"), super_crop)
    # увеличенная версия для глазного контроля
    big = cv2.resize(super_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / f"{video.stem}_idx{gt_idx}_05_median_super_3x.jpg"), big)

    # ----- decode попытки на median super-crop -----
    decoded = set()
    for variant_name, v in [
        ("raw", super_crop),
        ("2x", cv2.resize(super_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
        ("3x", big),
        ("gray", cv2.cvtColor(super_crop, cv2.COLOR_BGR2GRAY)),
        ("3x_gray", cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)),
    ]:
        pz = pyzbar_decode(v)
        for d in pz:
            decoded.add(("pyzbar/" + variant_name, d.data.decode("utf-8", errors="ignore"), d.type))
        if HAS_ZXING:
            for r in zxingcpp.read_barcodes(v):
                decoded.add(("zxing/" + variant_name, r.text, "?"))

    print(f"  decoded {len(decoded)} codes from median super-crop:")
    for d in decoded:
        match_tag = "  <-- MATCH GT" if d[1].strip() == gt_bc else ""
        print(f"    [{d[0]}]  type={d[2]}  data={d[1][:80]}{match_tag}")

    print(f"  results saved to {OUT}/{video.stem}_idx{gt_idx}_*.jpg")


def main():
    t0 = time.time()
    for mp4, csv, idx in SELECTED:
        process(mp4, csv, idx)
    print(f"\n[OK] elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
