"""HTML-дашборд по этапам pipeline с примерами.

Для каждого этапа: вход → выход → метрика + 1-2 примера картинок.

Этапы:
    1. Raw frame (вход видео)
    2. Rotation CCW
    3. Multicolor detection → bboxes
    4. Tracker dedup → tracks
    5. Best-frame по sharpness → crop
    6. Crop rotate + upscale → ready for OCR
    7. RapidOCR → OCR entries
    8. Parser → field values
    9. CSV writer → final row
    10. Evaluate → matched + score

Также: текущие метрики, история версий, что взяли из материалов команды.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_v8 import detect_multicolor

OUT = ROOT / "pipeline_dashboard.html"
TMP = ROOT / "outputs" / "dashboard_examples"
TMP.mkdir(parents=True, exist_ok=True)


def img_b64(path: Path, max_kb=500) -> str:
    if not path.exists(): return ""
    if path.stat().st_size > max_kb * 1024: return ""
    return f"data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def save_and_b64(img, name, max_w=800) -> str:
    p = TMP / name
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return img_b64(p)


def parse_ru(v):
    s = str(v).replace(",", ".").replace(" ", "")
    try: return float(s)
    except: return None


# === Готовим примеры ===
print("Generating examples...")

# 1-6. На примере одного ts=6595 из 26_12-20
cap = cv2.VideoCapture(str(ROOT / "data/labeled/26_12-20/26_12-20.mp4"))
cap.set(cv2.CAP_PROP_POS_MSEC, 6595)
ok, frame = cap.read()
cap.release()

# 1. Raw frame
raw_b64 = save_and_b64(frame, "stage1_raw.jpg", max_w=1200)

# 2. Rotated CCW (то как должно быть)
rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
rot_b64 = save_and_b64(rotated, "stage2_rotated_ccw.jpg", max_w=720)

# 3. Detection
dets = detect_multicolor(frame)
det_vis = frame.copy()
for bb, color in dets:
    x1, y1, x2, y2 = map(int, bb)
    col = {"red": (0,0,255), "yellow": (0,255,255), "green": (0,255,0)}.get(color, (255,255,255))
    cv2.rectangle(det_vis, (x1,y1), (x2,y2), col, 6)
    cv2.putText(det_vis, color, (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
det_b64 = save_and_b64(det_vis, "stage3_detections.jpg", max_w=1200)

# 4-5. Crop one ценник + rotate
if dets:
    # выбираем крупный red bbox
    reds = [b for b, c in dets if c == "red"]
    if reds:
        bb = max(reds, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        # padding 25%
        bw, bh = bb[2]-bb[0], bb[3]-bb[1]
        H, W = frame.shape[:2]
        x1 = max(0, bb[0] - int(bw*0.25)); y1 = max(0, bb[1] - int(bh*0.25))
        x2 = min(W, bb[2] + int(bw*0.25)); y2 = min(H, bb[3] + int(bh*0.25))
        crop = frame[y1:y2, x1:x2]
        crop_b64 = save_and_b64(crop, "stage5_crop_raw.jpg", max_w=600)
        crop_rot = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        crop_rot_up = cv2.resize(crop_rot, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        crop_ready_b64 = save_and_b64(crop_rot, "stage6_crop_rotated.jpg", max_w=400)
    else:
        crop_b64 = ""; crop_ready_b64 = ""
else:
    crop_b64 = ""; crop_ready_b64 = ""

# === Загружаем метрики ===
metrics_versions = []
for v in ["v4", "v7", "v8"]:
    pj = ROOT / f"outputs/pipeline_{v}/summary.json"
    if pj.exists():
        data = json.loads(pj.read_text(encoding="utf-8"))
        metrics_versions.append({
            "version": v,
            "matched": data.get("total_matched", 0),
            "qualified": data.get("total_qualified", 0),
            "gt": data.get("total_gt", 0),
            "main": data.get("main_metric", 0),
        })

# По полям на 26_12-20 — из последнего что есть
field_breakdown_html = ""
for v in ["v8", "v7", "v4"]:
    csv_p = ROOT / f"outputs/pipeline_{v}/26_12-20.csv"
    if csv_p.exists():
        try:
            from evaluate import match_rows, compare_fields, FIELDS_TO_COMPARE
            df_pred = pd.read_csv(csv_p, encoding="utf-8")
            df_gt = pd.read_csv(ROOT / "data/labeled/26_12-20/26_12-20.csv", encoding="utf-8")
            matches = match_rows(df_pred.copy(), df_gt.copy())
            matched_pairs = [(g,p) for g,p,_ in matches if p is not None]
            field_stats = {f: {"correct": 0, "total": 0} for f in FIELDS_TO_COMPARE}
            for g_idx, p_idx in matched_pairs:
                _, per_field = compare_fields(df_pred.iloc[p_idx], df_gt.iloc[g_idx])
                for f, info in per_field.items():
                    field_stats[f]["total"] += 1
                    if info["match"]: field_stats[f]["correct"] += 1
            rows = []
            for f, s in field_stats.items():
                if s["total"] == 0: continue
                pct = s["correct"]/s["total"]
                color = "bg-green-100" if pct >= 0.8 else "bg-yellow-100" if pct >= 0.3 else "bg-red-100"
                rows.append(f'<tr class="{color}"><td><code>{f}</code></td><td>{s["correct"]}/{s["total"]}</td><td>{pct:.0%}</td></tr>')
            field_breakdown_html = f'''<table class="data-table">
                <thead><tr><th>Поле</th><th>Correct/Matched</th><th>%</th></tr></thead>
                <tbody>{"".join(rows)}</tbody></table>'''
            break
        except Exception as e:
            print(f"  failed for {v}: {e}")

# === Этапы pipeline ===
stages_html = f'''
<div class="stage">
  <h3>1. RAW frame (вход)</h3>
  <div class="grid-2">
    <div>
      <p class="text-sm text-slate-600"><b>Вход:</b> mp4 видео 3840×2160, ~20 fps, h264, fisheye</p>
      <p class="text-sm text-slate-600"><b>Выход:</b> кадр-numpy (2160, 3840, 3) в BGR</p>
      <p class="text-sm text-slate-600"><b>Алгоритм:</b> <code>cv2.VideoCapture.read()</code></p>
      <p class="text-sm text-slate-600"><b>Параметры камеры от орга (14.05 18:37):</b><br>
        • 3840×2160<br>
        • Объектив 16/2.8mm (1/2.8" матрица, focal 2.8mm)<br>
        • Видео повёрнуто на 90° против часовой стрелки!<br>
        • Дисторсия: подбирать самим</p>
    </div>
    <div><img src="{raw_b64}" class="img-thumb"/></div>
  </div>
</div>

<div class="stage">
  <h3>2. Rotation CCW (фикс ориентации) — NEW в v8!</h3>
  <div class="grid-2">
    <div>
      <p class="text-sm text-slate-600"><b>Вход:</b> кадр (2160, 3840, 3)</p>
      <p class="text-sm text-slate-600"><b>Выход:</b> кадр (3840, 2160, 3) в правильной ориентации</p>
      <p class="text-sm text-slate-600"><b>Алгоритм:</b> <code>cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)</code></p>
      <p class="text-sm text-slate-600 bg-amber-100 p-2 rounded"><b>Почему:</b> текст и ценники теперь горизонтально → OCR работает.</p>
      <p class="text-sm text-slate-600"><b>Примечание:</b> делаем поворот ТОЛЬКО на crop'е перед OCR, координаты bbox остаются в исходной системе (как в GT).</p>
    </div>
    <div><img src="{rot_b64}" class="img-thumb"/></div>
  </div>
</div>

<div class="stage">
  <h3>3. Multicolor detection</h3>
  <div class="grid-2">
    <div>
      <p class="text-sm text-slate-600"><b>Вход:</b> кадр</p>
      <p class="text-sm text-slate-600"><b>Выход:</b> список (bbox, color) — найдено <b>{len(dets)}</b> на этом кадре</p>
      <p class="text-sm text-slate-600"><b>Алгоритм:</b> HSV маски red/yellow/green → морфология (close+open) → connected components → bbox</p>
      <p class="text-sm text-slate-600"><b>Параметры:</b> min_area=4000 px², aspect 0.25..4, global NMS IoU≥0.3</p>
      <p class="text-sm text-slate-600"><b>Recall на 26_12-20 GT:</b> ~60% (red-only был 60%, multicolor шумит)</p>
    </div>
    <div><img src="{det_b64}" class="img-thumb"/></div>
  </div>
</div>

<div class="stage">
  <h3>4. Tracker (IoU-based)</h3>
  <p class="text-sm text-slate-600"><b>Вход:</b> на каждом 6-м кадре — список bbox-detections</p>
  <p class="text-sm text-slate-600"><b>Выход:</b> треки (один track_id = один ценник через несколько кадров)</p>
  <p class="text-sm text-slate-600"><b>Алгоритм:</b> для каждого dets ищем active track с max IoU ≥ 0.25, иначе создаём новый</p>
  <p class="text-sm text-slate-600"><b>Параметры:</b> IoU threshold 0.25, max_missed_ms 600, min_track_frames 3</p>
  <p class="text-sm text-slate-600 bg-yellow-100 p-2 rounded"><b>Проблема:</b> 3× over-detection (848 pred vs 274 GT). Treker связывает шумные bbox в треки.</p>
</div>

<div class="stage">
  <h3>5. Best-frame per track</h3>
  <div class="grid-2">
    <div>
      <p class="text-sm text-slate-600"><b>Вход:</b> трек с N кадрами</p>
      <p class="text-sm text-slate-600"><b>Выход:</b> один best_crop (max sharpness)</p>
      <p class="text-sm text-slate-600"><b>Алгоритм:</b> <code>cv2.Laplacian(gray, cv2.CV_64F).var()</code></p>
      <p class="text-sm text-slate-600 bg-green-100 p-2 rounded"><b>Memory fix:</b> храним только best_crop в треке, не все кадры. RAM ×50 ниже.</p>
    </div>
    <div><img src="{crop_b64}" class="img-thumb"/></div>
  </div>
</div>

<div class="stage">
  <h3>6. Crop rotation + upscale (готовим к OCR) — NEW в v8!</h3>
  <div class="grid-2">
    <div>
      <p class="text-sm text-slate-600"><b>Вход:</b> best_crop (исходная ориентация)</p>
      <p class="text-sm text-slate-600"><b>Выход:</b> rotated CCW + upscale ×3 → готов для OCR</p>
      <p class="text-sm text-slate-600"><b>Алгоритм:</b> <code>cv2.rotate(crop, CCW)</code> → <code>cv2.resize(fx=3, fy=3, INTER_CUBIC)</code></p>
      <p class="text-sm text-slate-600 bg-green-100 p-2 rounded"><b>Зачем upscale:</b> OCR пасует на crop'ах меньше 200×300 px. Upscale x3 = 600×900 → OCR работает.</p>
    </div>
    <div><img src="{crop_ready_b64}" class="img-thumb"/></div>
  </div>
</div>

<div class="stage">
  <h3>7. RapidOCR</h3>
  <p class="text-sm text-slate-600"><b>Вход:</b> rotated upscaled crop</p>
  <p class="text-sm text-slate-600"><b>Выход:</b> список <code>(bbox, text, confidence)</code> entries</p>
  <p class="text-sm text-slate-600"><b>Алгоритм:</b> DBNet++ детекция текста + CRNN/SVTR распознавание (русский + латиница)</p>
  <p class="text-sm text-slate-600"><b>Лицензия:</b> Apache 2.0, ONNX-runtime, CPU-friendly</p>
  <p class="text-sm text-slate-600"><b>Параметры:</b> <code>use_angle_cls=True</code> (поворачивает буквенные строки если они под углом)</p>
  <p class="text-sm text-slate-600 bg-amber-100 p-2 rounded"><b>Замечание:</b> рассматриваем замену на PP-OCRv5 East-Slavic — 81.6% line-exact на русском (vs ~70% у обычного RapidOCR).</p>
</div>

<div class="stage">
  <h3>8. Parser OCR → CSV fields</h3>
  <p class="text-sm text-slate-600"><b>Вход:</b> OCR entries</p>
  <p class="text-sm text-slate-600"><b>Выход:</b> dict с 23 распознаваемыми полями</p>
  <p class="text-sm text-slate-600"><b>Алгоритм:</b></p>
  <ul class="text-sm text-slate-600 list-disc pl-6">
    <li><code>discount_amount</code>: regex <code>-?(\\d{{1,2}})\\s*%</code></li>
    <li><code>print_datetime</code>: regex <code>\\d{{2}}\\.\\d{{2}}\\.\\d{{4}}</code> + время</li>
    <li><code>price_card</code>: самая высокая <code>^\\d{{2,4}}$</code> строка + поиск 2-цифр копеек рядом</li>
    <li><code>price_default</code>: вторая по высоте (50-95% от price_card)</li>
    <li><code>barcode</code>: <code>^\\d{{8,14}}$</code> + склейка соседних цифровых сегментов</li>
    <li><code>id_sku</code>: <code>^\\d{{10,12}}$</code></li>
    <li><code>product_name</code>: все строки с буквами (фильтр &gt;= 2 букв)</li>
    <li><code>color</code>: от детектора (red/yellow/green)</li>
  </ul>
  <p class="text-sm text-slate-600 bg-red-100 p-2 rounded"><b>Проблема:</b> парсер слабый, 0% на product_name/prices/barcode на матченных строках. Дефолт «нет» для 7 always-нет полей даёт mean=33% на GT-bbox (диагностика).</p>
</div>

<div class="stage">
  <h3>9. CSV writer</h3>
  <p class="text-sm text-slate-600"><b>Вход:</b> словари полей, один на track_id</p>
  <p class="text-sm text-slate-600"><b>Выход:</b> CSV в формате sample.csv (29 колонок, точка = десятичный разделитель)</p>
  <p class="text-sm text-slate-600"><b>Дефолт для 7 always-нет полей:</b> "нет" (price_discount, wholesale_*_count/price, action_*_qr)</p>
</div>

<div class="stage">
  <h3>10. Evaluate (метрика согласно ТЗ + чату орга 14.05)</h3>
  <p class="text-sm text-slate-600"><b>Вход:</b> pred CSV + GT CSV</p>
  <p class="text-sm text-slate-600"><b>Алгоритм:</b></p>
  <ol class="text-sm text-slate-600 list-decimal pl-6">
    <li>Матчинг строк: <b>barcode primary</b> (если совпали), иначе <b>spatial-temporal</b> (|Δts|≤1500ms, IoU≥0.3)</li>
    <li>Для матченной пары — <b>% содержательных полей</b> совпало (24 поля, технические не учитываются)</li>
    <li><b>Зачёт</b>: ≥80% полей правильно</li>
    <li><b>Итоговая метрика</b>: доля зачтённых ценников от |GT|</li>
  </ol>
</div>
'''

# === История версий ===
history_rows = ""
for m in metrics_versions:
    history_rows += f'<tr><td><b>{m["version"]}</b></td><td>{m["gt"]}</td><td>{m["matched"]}</td><td>{m["matched"]/m["gt"]:.0%}</td><td>{m["qualified"]}</td><td>{m["main"]:.0%}</td></tr>'

# === HTML ===
html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Pipeline Dashboard — Lenta Shelf Control</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  body {{ font-family: -apple-system, sans-serif; }}
  .img-thumb {{ max-width: 100%; border-radius: 6px; border: 1px solid #ccc; }}
  .stage {{ background: #fff; padding: 1.2em; margin: 1em 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #3b82f6; }}
  .stage h3 {{ font-size: 1.25rem; font-weight: 700; margin-bottom: 0.5em; color: #1e40af; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1em; }}
  .data-table {{ width: 100%; border-collapse: collapse; }}
  .data-table th, .data-table td {{ padding: 6px 12px; border-bottom: 1px solid #e5e7eb; }}
  .data-table th {{ background: #f9fafb; font-weight: 600; }}
  code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body class="bg-slate-100 p-4 md:p-8">
<div class="max-w-7xl mx-auto space-y-6">

<header class="bg-gradient-to-r from-blue-700 to-indigo-800 text-white p-6 rounded-lg">
  <h1 class="text-3xl font-bold">Pipeline Dashboard</h1>
  <p class="text-blue-100 mt-2">Каждый этап пайплайна с примерами входа и выхода. Все артефакты — на реальных данных видео <code>26_12-20.mp4 ts=6595ms</code>.</p>
</header>

<section class="bg-white p-6 rounded-lg shadow">
  <h2 class="text-2xl font-bold mb-4">📊 История версий пайплайна</h2>
  <table class="data-table">
    <thead><tr><th>Версия</th><th>GT</th><th>Matched</th><th>Match %</th><th>Qualified ≥80%</th><th>Main metric</th></tr></thead>
    <tbody>{history_rows}</tbody>
  </table>
  <p class="text-sm text-slate-600 mt-3">
    <b>v4</b>: memory fix + global NMS<br>
    <b>v7</b>: padding +25% во все стороны + дефолт «нет»<br>
    <b>v8</b>: + поворот crop на 90° CCW перед OCR + use_angle_cls (фикс ориентации — параметры камеры от орга)
  </p>
</section>

<section class="bg-white p-6 rounded-lg shadow">
  <h2 class="text-2xl font-bold mb-4">📋 Поля по точности (на 26_12-20)</h2>
  {field_breakdown_html or "<p>Нет данных</p>"}
</section>

<section>
  <h2 class="text-2xl font-bold mb-4">🔬 Этапы pipeline</h2>
  {stages_html}
</section>

<section class="bg-white p-6 rounded-lg shadow">
  <h2 class="text-2xl font-bold mb-4">🆕 Что взяли из материалов команды (SPEC + PIPELINE_BRIEF)</h2>
  <table class="data-table">
    <thead><tr><th>Идея</th><th>Статус</th><th>Что даёт</th></tr></thead>
    <tbody>
      <tr><td>Pydantic schema (29 полей)</td><td>🟡 в плане</td><td>Чистая структура, EAN-13 checksum, валидация</td></tr>
      <tr><td>zxing-cpp 3.0 (DataMatrix + PDF417)</td><td>✅ установлен</td><td>Покрывает Честный Знак (молочка) + EGAIS (алкоголь)</td></tr>
      <tr><td>PP-OCRv5 East-Slavic для русского</td><td>🟡 в плане</td><td>81.6% line-exact на русском (vs наш RapidOCR ~70%)</td></tr>
      <tr><td>Quality gate (price_card ≤ price_default)</td><td>🟡 в плане</td><td>Отсев противоречий, чистые поля</td></tr>
      <tr><td>D-FINE-S Apache 2.0 fallback</td><td>⬜ не приоритет</td><td>План B на случай AGPL-ограничений</td></tr>
      <tr><td>Edge-first архитектура (50 КБ vs 5 МБ/сек)</td><td>📄 для презы</td><td>Сильный аргумент scaling: 100 000× меньше трафика</td></tr>
    </tbody>
  </table>
</section>

</div>
</body>
</html>
'''

OUT.write_text(html, encoding="utf-8")
print(f"[OK] Pipeline dashboard: {OUT}")
