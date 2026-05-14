"""Multi-frame QR fusion на уровне СИРОТА QR (а не всего ценника).

Подход:
    1. Для каждого GT-ценника: окно ±1000 мс, 40 кадров
    2. На каждом кадре через QReader-detector ищем QR (boundingbox)
    3. Если QR детектится на N кадрах (N≥5) → собираем кропы ТОЛЬКО QR
    4. Выравниваем QR-кропы через ECC к самому чёткому
    5. Weighted-average кропов по sharpness
    6. Декодируем результирующий "super-QR"

Это качественно отличается от предыдущего стэкинга (где мы стэкали ВЕСЬ ценник
и QR размазывался из-за того что QR — крошечная часть ценника):
    + QR-кропы выровнены прицельно (не "якорь по красному блоку")
    + Размер QR-кропа фиксирован → grid алгоритма точнее
    + Sharpness-веса по самому QR, а не по всему ценнику
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyzbar.pyzbar import decode as pyzbar_decode
from qreader import QReader
from tqdm import tqdm

import zxingcpp

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "qr_multiframe_fusion"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR = OUT / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

print("Initializing QReader...")
qr = QReader(model_size="s", min_confidence=0.2)
print("Ready.")

VIDEOS = ["26_12-20", "43_15"]
WINDOW_MS = 1000
STEP_MS = 50  # ~40 кадров в окне
PAD_PCT = 0.8


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


def pad(b, shape, pct):
    h, w = shape[:2]
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * pct / 2), int(bh * pct / 2)
    return (max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py))


def sharpness(img):
    if img.dtype != np.uint8: img = np.clip(img, 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def find_qr_on_frame(crop):
    """Возвращает [(bbox, conf), ...] для всех найденных QR на кропе."""
    try:
        detections = qr.detect(image=crop)
        result = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            conf = float(d.get("confidence", 0))
            result.append(((int(x1), int(y1), int(x2), int(y2)), conf))
        return result
    except Exception:
        return []


def decode_all(img):
    res = set()
    for variant in [img,
                    cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
                    cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img]:
        try:
            for d in pyzbar_decode(variant): res.add(d.data.decode("utf-8", errors="ignore"))
        except Exception: pass
        try:
            for r in zxingcpp.read_barcodes(variant): res.add(r.text)
        except Exception: pass
        if HAS_PYBOOF:
            try:
                gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if variant.ndim == 3 else variant
                pb_img = pb.ndarray_to_boof(gray)
                PB_QR.detect(pb_img)
                for det in PB_QR.detections:
                    if det.message: res.add(det.message)
            except Exception: pass
        # QReader decode тоже
        try:
            decoded = qr.detect_and_decode(image=variant)
            for d in decoded:
                if d: res.add(d)
        except Exception: pass
    return res


def fuse_qr_crops(qr_crops_with_sharp):
    """Выравнивает QR-кропы к самому чёткому, weighted average по sharpness."""
    if not qr_crops_with_sharp: return None, 0
    # сортируем по sharpness убывая
    sorted_crops = sorted(qr_crops_with_sharp, key=lambda x: -x[1])
    template = sorted_crops[0][0]  # самый чёткий
    H_t, W_t = template.shape[:2]
    if H_t < 30 or W_t < 30: return None, 0
    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

    aligned = [template.astype(np.float32)]
    weights = [sorted_crops[0][1]]
    for crop, sh in sorted_crops[1:20]:  # max 20 кадров
        # ресайз к размеру template
        c = cv2.resize(crop, (W_t, H_t))
        c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
            _, warp = cv2.findTransformECC(tmpl_gray, c_gray, warp, cv2.MOTION_TRANSLATION, crit, None, 5)
            warped = cv2.warpAffine(c.astype(np.float32), warp, (W_t, H_t),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error:
            warped = c.astype(np.float32)
        aligned.append(warped)
        weights.append(sh)
    if len(aligned) < 2:
        return template, 1
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    arr = np.stack(aligned, axis=0)
    fused = np.tensordot(weights, arr, axes=(0, 0))
    return np.clip(fused, 0, 255).astype(np.uint8), len(aligned)


def run_video(video_name: str) -> dict:
    csv = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4 = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype(str).str.strip().str.replace(".0", "", regex=False)

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened(): return {}
    total = len(df)
    qr_detected_count = 0
    decoded_count = 0
    matched_count = 0
    matched_rows = []
    stack_size_dist = []

    for i, r in tqdm(df.iterrows(), total=total, desc=video_name):
        ts_gt = r["frame_timestamp"]
        gt_bc = r["barcode"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))

        # 1. собираем кадры в окне, на каждом ищем QR
        qr_crops = []
        start_ms = max(0, ts_gt - WINDOW_MS)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms))
        while True:
            cur_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            if cur_ts > ts_gt + WINDOW_MS: break
            ok, frame = cap.read()
            if not ok: break
            b = pad(bbox, frame.shape, PAD_PCT)
            crop = frame[b[1]:b[3], b[0]:b[2]]
            if crop.size == 0: continue
            detections = find_qr_on_frame(crop)
            for (qx1, qy1, qx2, qy2), conf in detections:
                qcrop = crop[qy1:qy2, qx1:qx2]
                if qcrop.shape[0] >= 30 and qcrop.shape[1] >= 30:
                    qr_crops.append((qcrop, sharpness(qcrop)))

        if not qr_crops:
            continue
        qr_detected_count += 1
        stack_size_dist.append(len(qr_crops))

        # 2. fuse
        super_qr, used_n = fuse_qr_crops(qr_crops)
        if super_qr is None: continue

        # 3. decode
        codes = decode_all(super_qr)
        if codes:
            decoded_count += 1
        matched = gt_bc in codes
        if matched:
            matched_count += 1
            matched_rows.append({"idx": int(i), "GT": gt_bc, "decoded": list(codes), "stack_n": used_n})

        # save sample for first 8 and for matches
        if i < 8 or matched:
            tag = "MATCH" if matched else ("dec" if codes else "miss")
            cv2.imwrite(str(SAMPLES_DIR / f"{video_name}_idx{i:02d}_n{used_n}_{tag}_GT{gt_bc}.jpg"), super_qr)
    cap.release()

    print(f"\n=== {video_name} (total={total}) ===")
    print(f"  QR detected by QReader in window: {qr_detected_count}/{total} ({qr_detected_count/total:.0%})")
    if stack_size_dist:
        print(f"  Stack size: min={min(stack_size_dist)}  median={int(np.median(stack_size_dist))}  max={max(stack_size_dist)}")
    print(f"  Anything decoded:                 {decoded_count}/{total} ({decoded_count/total:.0%})")
    print(f"  Matched GT barcode:               {matched_count}/{total} ({matched_count/total:.0%})")
    if matched_rows:
        print(f"  --- MATCHED ---")
        for r in matched_rows: print(f"    idx{r['idx']:02d}  GT={r['GT']}  decoded={r['decoded']}  stack_n={r['stack_n']}")
    return {"video": video_name, "total": total, "qr_detected": qr_detected_count,
            "decoded": decoded_count, "matched": matched_count, "matched_rows": matched_rows,
            "stack_sizes": stack_size_dist}


def main():
    t0 = time.time()
    report = {"videos": []}
    for v in VIDEOS:
        report["videos"].append(run_video(v))
    report["elapsed_sec"] = time.time() - t0
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] elapsed {report['elapsed_sec']:.0f}s  -> {OUT/'report.json'}")
    print(f"     samples -> {SAMPLES_DIR}")


if __name__ == "__main__":
    main()
