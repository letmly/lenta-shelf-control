"""Multi-frame image stacking для восстановления QR/штрихкода.

Идея:
    На отдельных кадрах QR смазан. Но за ~2 секунды до и после GT-timestamp
    мы имеем ~60 кадров с тем же ценником. Если выровнять их по нашему
    эталонному кропу и усреднить (median), motion blur одиночных кадров
    подавится — должен получиться более чёткий "суперкадр", на котором
    декодер QR/штрихкода заработает.

Параметры:
    окно: ±1500 мс от GT timestamp
    шаг: 50 мс (~30 кадров до + 30 после = 60 кандидатов)

Пайплайн на 1 ценник:
    1. Эталонный кадр на ts_gt → cropped с padding +50%
    2. Для каждого ts в окне:
        - читаем кадр
        - ищем "тот же ценник" в нём через ECC alignment + ORB fallback
        - получаем выровненный crop
    3. Стек из ~60 crops → median по оси кадров → super_crop
    4. Прогон pyzbar + zxing-cpp на super_crop
    5. Опц.: сохраняем super_crop для глазного просмотра

Стратегии для сравнения:
    REF_only       — decode только эталонного кадра (это уже было: 0%)
    STACK_median   — медиана 60 выровненных кадров
    STACK_mean     — среднее 60 выровненных кадров
    STACK_max      — max-projection (для тёмных штрихкодов)
"""
from __future__ import annotations

import json
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
OUT = ROOT / "outputs" / "stacking"
OUT.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = OUT / "super_crops"
SAMPLE_DIR.mkdir(exist_ok=True)

VIDEOS = [
    (ROOT / "data/labeled/26_12-20/26_12-20.mp4", ROOT / "data/labeled/26_12-20/26_12-20.csv"),
    (ROOT / "data/labeled/43_15/43_15.mp4",        ROOT / "data/labeled/43_15/43_15.csv"),
]

WINDOW_MS = 1500     # ±1.5 секунды вокруг GT timestamp
STEP_MS = 50         # шаг 50 мс
PAD_PCT = 0.5        # padding +50% от bbox
ECC_ITER = 100
ECC_EPS = 1e-4
SAVE_FIRST_N_SAMPLES = 5  # сохраняем super_crops для визуальной проверки


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


def grab_frame(cap, ts_ms):
    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts_ms))
    ok, frame = cap.read()
    return frame if ok else None


def align_ecc(template_gray, target_gray, warp_mode=cv2.MOTION_TRANSLATION):
    """ECC alignment. Возвращает (warped_target, success). Если не сошлось — fallback на NULL warp."""
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp = np.eye(2, 3, dtype=np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITER, ECC_EPS)
        cc, warp = cv2.findTransformECC(template_gray, target_gray, warp, warp_mode, criteria, None, 5)
        return warp, True
    except cv2.error:
        return None, False


def warp_image(img, warp, shape, mode):
    h, w = shape
    if mode == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(img, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return cv2.warpAffine(img, warp, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)


def collect_aligned_crops(cap, ref_crop, ref_bbox_padded, gt_ts):
    """Собираем выровненные кропы в окне ±WINDOW_MS вокруг gt_ts."""
    ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    H, W = ref_crop.shape[:2]
    rx1, ry1, rx2, ry2 = ref_bbox_padded
    stack = [ref_crop.astype(np.float32)]  # эталонный кадр всегда первый

    for dt in range(-WINDOW_MS, WINDOW_MS + 1, STEP_MS):
        if dt == 0: continue
        ts = gt_ts + dt
        frame = grab_frame(cap, ts)
        if frame is None: continue
        # берём ROI в той же области (плюс запас для сдвига робота)
        margin = 80
        h_f, w_f = frame.shape[:2]
        ex1 = max(0, rx1 - margin); ey1 = max(0, ry1 - margin)
        ex2 = min(w_f, rx2 + margin); ey2 = min(h_f, ry2 + margin)
        roi = frame[ey1:ey2, ex1:ex2]
        if roi.shape[0] < H or roi.shape[1] < W: continue
        # шиф через template matching как первое приближение
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        ref_gray_u8 = (ref_gray * 255).astype(np.uint8)
        try:
            res = cv2.matchTemplate(roi_gray, ref_gray_u8, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.3:
            continue  # ценник не виден в этом кадре
        tx, ty = max_loc
        # вырезаем кадр того же размера что ref_crop, начиная с (tx, ty)
        cand = roi[ty:ty+H, tx:tx+W]
        if cand.shape[:2] != (H, W): continue
        # затем уточняем ECC
        cand_gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        warp, ok = align_ecc(ref_gray, cand_gray, cv2.MOTION_TRANSLATION)
        if ok:
            warped = warp_image(cand.astype(np.float32), warp, (H, W), cv2.MOTION_TRANSLATION)
        else:
            warped = cand.astype(np.float32)
        stack.append(warped)
    return stack


def stack_to_super(stack, mode="median"):
    arr = np.stack(stack, axis=0)  # (N, H, W, 3)
    if mode == "median":
        sup = np.median(arr, axis=0)
    elif mode == "mean":
        sup = np.mean(arr, axis=0)
    elif mode == "max":
        sup = np.max(arr, axis=0)
    else:
        raise ValueError(mode)
    return np.clip(sup, 0, 255).astype(np.uint8)


def try_all_decoders(img):
    results = {"pyzbar": [], "zxing": []}
    for variant_name, variant in [
        ("raw", img),
        ("2x", cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
        ("gray", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
        ("2x_gray", cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
    ]:
        try:
            pz = pyzbar_decode(variant)
            for d in pz:
                results["pyzbar"].append(d.data.decode("utf-8", errors="ignore").strip())
        except Exception: pass
        if HAS_ZXING:
            try:
                zx = zxingcpp.read_barcodes(variant)
                for r in zx:
                    results["zxing"].append(r.text.strip())
            except Exception: pass
    # dedup
    results["pyzbar"] = list(set(results["pyzbar"]))
    results["zxing"] = list(set(results["zxing"]))
    return results


def run(video: Path, csv: Path):
    df = load_gt(csv)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): return None
    total = len(df)
    stats = {
        "REF_only":     {"decoded": 0, "matched": 0},
        "STACK_median": {"decoded": 0, "matched": 0},
        "STACK_mean":   {"decoded": 0, "matched": 0},
        "STACK_max":    {"decoded": 0, "matched": 0},
        "ANY_STACK":    {"decoded": 0, "matched": 0},
    }
    samples_saved = 0

    for idx, r in tqdm(df.iterrows(), total=total, desc=video.name):
        ts = r["frame_timestamp"]
        gt_bc = r["barcode"]
        bbox = (int(r["x_min"]), int(r["y_min"]), int(r["x_max"]), int(r["y_max"]))

        ref_frame = grab_frame(cap, ts)
        if ref_frame is None: continue
        bb = pad(bbox, ref_frame.shape, PAD_PCT)
        ref_crop = ref_frame[bb[1]:bb[3], bb[0]:bb[2]]
        if ref_crop.size == 0: continue

        # --- REF only ---
        rd = try_all_decoders(ref_crop)
        all_rd = rd["pyzbar"] + rd["zxing"]
        if all_rd:
            stats["REF_only"]["decoded"] += 1
            if gt_bc in all_rd: stats["REF_only"]["matched"] += 1

        # --- собрать стек ---
        stack = collect_aligned_crops(cap, ref_crop, bb, ts)
        if len(stack) < 5:
            continue  # стек слишком маленький, не имеет смысла

        # --- три варианта стека ---
        any_dec = False; any_match = False
        for mode in ("median", "mean", "max"):
            sup = stack_to_super(stack, mode=mode)
            d = try_all_decoders(sup)
            all_d = d["pyzbar"] + d["zxing"]
            key = f"STACK_{mode}"
            if all_d:
                stats[key]["decoded"] += 1
                any_dec = True
                if gt_bc in all_d:
                    stats[key]["matched"] += 1
                    any_match = True
            # сохраним первые N median super_crops для глазного контроля
            if mode == "median" and samples_saved < SAVE_FIRST_N_SAMPLES:
                cv2.imwrite(str(SAMPLE_DIR / f"{video.stem}_idx{idx}_stacksize{len(stack)}_GT{gt_bc}.jpg"), sup)
                samples_saved += 1
        if any_dec:  stats["ANY_STACK"]["decoded"] += 1
        if any_match: stats["ANY_STACK"]["matched"] += 1

    cap.release()
    out = {"video": video.name, "total": total, "stats": {k: {**v, "decoded_rate": v["decoded"]/total, "matched_rate": v["matched"]/total} for k, v in stats.items()}}
    return out


def main():
    print(f"zxing-cpp: {'YES' if HAS_ZXING else 'NO'}")
    t0 = time.time()
    report = {"started_at": t0, "videos": []}
    for video, csv in VIDEOS:
        if not video.exists(): continue
        r = run(video, csv)
        if r is None: continue
        report["videos"].append(r)
        print(f"\n=== {r['video']} (GT={r['total']}) ===")
        for k, st in r["stats"].items():
            print(f"  {k:14s} decoded={st['decoded']:3d} ({st['decoded_rate']:.0%})  matched={st['matched']:3d} ({st['matched_rate']:.0%})")
    report["elapsed_sec"] = time.time() - t0
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] {OUT/'report.json'}    super_crops in {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
