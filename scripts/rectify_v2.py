"""Rectify v2: ищем ВЕСЬ ценник (красный + белый), а не только красный блок.

Подходы:
    A. adaptive threshold по grayscale → external contours → берём максимальный
       по площади четырёхугольник
    B. red HSV mask + dilate сильно вниз/вверх → пытаемся прихватить белую часть
    C. Canny edges + cv2.connectedComponents → большой connected region
    D. (fallback) — minAreaRect самого большого контура с любого метода

Цель: получить warped изображение, на котором ВИДЕН и QR, и штрихкод.
"""
from __future__ import annotations

import json
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

try:
    import pyboof as pb
    PB_QR = pb.FactoryFiducial(np.uint8).qrcode()
    HAS_PYBOOF = True
except Exception:
    HAS_PYBOOF = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "rectify_v2"
OUT.mkdir(parents=True, exist_ok=True)

VIDEOS = ["26_12-20", "43_15"]
PAD_PCT = 0.8  # ещё больше padding чтобы захватить весь ценник


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


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                     pts[np.argmax(s)], pts[np.argmax(diff)]], dtype=np.float32)


def find_quad_method_A_adaptive(crop):
    """Adaptive threshold + крупный external контур."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # инверсия: ценник (светлый+красный) на сером (тёмном) фоне → ценник = светлый
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 51, -5)
    k = np.ones((7, 7), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 5000: return None
    peri = cv2.arcLength(big, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(big, eps * peri, True)
        if len(approx) == 4: return approx
    # fallback — minAreaRect
    return cv2.boxPoints(cv2.minAreaRect(big)).astype(np.int32).reshape(-1, 1, 2)


def find_quad_method_B_red_extended(crop):
    """Красная маска, потом растягиваем вниз чтобы захватить белую часть."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    red = m1 | m2
    if red.sum() < 1000: return None
    # белая часть — высокая яркость, низкая насыщенность
    white = cv2.inRange(hsv, (0, 0, 180), (180, 50, 255))
    # объединяем red OR white, затем большая dilation чтобы соединить блоки
    combined = red | white
    k = np.ones((9, 9), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k, iterations=3)
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 5000: return None
    peri = cv2.arcLength(big, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(big, eps * peri, True)
        if len(approx) == 4: return approx
    return cv2.boxPoints(cv2.minAreaRect(big)).astype(np.int32).reshape(-1, 1, 2)


def find_quad_method_C_canny(crop):
    """Canny edges + dilate → крупный контур."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    k = np.ones((9, 9), np.uint8)
    edges = cv2.dilate(edges, k, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    big = max(contours, key=cv2.contourArea)
    if cv2.contourArea(big) < 5000: return None
    peri = cv2.arcLength(big, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(big, eps * peri, True)
        if len(approx) == 4: return approx
    return cv2.boxPoints(cv2.minAreaRect(big)).astype(np.int32).reshape(-1, 1, 2)


METHODS = [
    ("A_adaptive", find_quad_method_A_adaptive),
    ("B_red+white", find_quad_method_B_red_extended),
    ("C_canny", find_quad_method_C_canny),
]


def warp_to_rect(img, quad, scale=2.5):
    src = order_corners(quad)
    w1 = np.linalg.norm(src[0] - src[1]); w2 = np.linalg.norm(src[3] - src[2])
    h1 = np.linalg.norm(src[0] - src[3]); h2 = np.linalg.norm(src[1] - src[2])
    out_w = int(max(w1, w2) * scale)
    out_h = int(max(h1, h2) * scale)
    if out_w < 50 or out_h < 50: return None
    dst = np.array([[0, 0], [out_w-1, 0], [out_w-1, out_h-1], [0, out_h-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (out_w, out_h))


def decode_all(img) -> set:
    res = set()
    for v in [img, cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
              cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)]:
        try:
            for d in pyzbar_decode(v): res.add(d.data.decode("utf-8", errors="ignore").strip())
        except Exception: pass
        if HAS_ZXING:
            try:
                for r in zxingcpp.read_barcodes(v): res.add(r.text.strip())
            except Exception: pass
        if HAS_PYBOOF:
            try:
                gray = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) if v.ndim == 3 else v
                pb_img = pb.ndarray_to_boof(gray)
                PB_QR.detect(pb_img)
                for det in PB_QR.detections:
                    if det.message: res.add(det.message.strip())
            except Exception: pass
    return res


def process(video_name: str) -> dict:
    csv = ROOT / f"data/labeled/{video_name}/{video_name}.csv"
    mp4 = ROOT / f"data/labeled/{video_name}/{video_name}.mp4"
    out_dir = OUT / video_name
    out_dir.mkdir(exist_ok=True)
    df = pd.read_csv(csv, encoding="utf-8")
    for c in ("x_min", "y_min", "x_max", "y_max", "frame_timestamp"):
        df[c] = df[c].apply(parse_ru)
    df = df.dropna(subset=["x_min", "y_min", "x_max", "y_max", "frame_timestamp"]).reset_index(drop=True)
    cap = cv2.VideoCapture(str(mp4))

    method_stats = {m: {"quad": 0, "decoded": 0, "matched": 0} for m, _ in METHODS}
    any_decoded = 0; any_matched = 0
    matched_rows = []

    for i, r in df.iterrows():
        ts = r["frame_timestamp"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))
        gt_bc = str(r.get("barcode", "")).strip().replace(".0", "")
        cap.set(cv2.CAP_PROP_POS_MSEC, float(ts))
        ok, frame = cap.read()
        if not ok: continue
        b = pad(bbox, frame.shape, PAD_PCT)
        crop = frame[b[1]:b[3], b[0]:b[2]]
        if crop.size == 0 or crop.shape[0] < 30: continue

        all_decoded_this = set()
        warps = []
        for mname, mfn in METHODS:
            quad = mfn(crop)
            if quad is None: continue
            method_stats[mname]["quad"] += 1
            warped = warp_to_rect(crop, quad.reshape(-1, 2), scale=2.5)
            if warped is None: continue
            decoded = decode_all(warped)
            if decoded: method_stats[mname]["decoded"] += 1
            if gt_bc in decoded: method_stats[mname]["matched"] += 1
            all_decoded_this.update(decoded)
            warps.append((mname, warped))

        if all_decoded_this: any_decoded += 1
        matched = gt_bc in all_decoded_this
        if matched: any_matched += 1; matched_rows.append({"idx": i, "GT": gt_bc, "decoded": list(all_decoded_this)})

        # сохраняем артефакты для первых 10
        if i < 10 or matched:
            for mname, warped in warps:
                tag = "MATCH" if matched else "miss"
                cv2.imwrite(str(out_dir / f"idx{i:02d}_GT{gt_bc}_{tag}_{mname}_warped.jpg"), warped)

    cap.release()
    total = len(df)
    print(f"\n=== {video_name} (total={total}) ===")
    for m, s in method_stats.items():
        print(f"  {m:12s}  quad={s['quad']:3d} ({s['quad']/total:.0%})  "
              f"decoded={s['decoded']:3d} ({s['decoded']/total:.0%})  "
              f"matched={s['matched']:3d} ({s['matched']/total:.0%})")
    print(f"  ANY method:    decoded={any_decoded} ({any_decoded/total:.0%})  matched={any_matched} ({any_matched/total:.0%})")
    if matched_rows:
        print(f"  --- MATCHED ---")
        for r in matched_rows:
            print(f"    idx{r['idx']:02d}  GT={r['GT']}  decoded={r['decoded']}")
    return {"video": video_name, "total": total, "methods": method_stats,
            "any_decoded": any_decoded, "any_matched": any_matched, "matches": matched_rows}


results = []
for v in VIDEOS:
    results.append(process(v))
(OUT / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[OK] {OUT/'report.json'}")
