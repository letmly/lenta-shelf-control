"""Генерирует одностраничный HTML-дашборд по проекту.

Открывается в браузере, всё внутри: текущее состояние, гипотезы, пайплайн,
метрики, визуальные примеры с реальных шагов.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dashboard.html"


def img_to_b64(path: Path, max_kb=300) -> str:
    """Файл → base64 data URI. Если файл больше max_kb — пропускаем."""
    if not path.exists(): return ""
    size_kb = path.stat().st_size / 1024
    if size_kb > max_kb:
        return ""  # too big
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def img_or_placeholder(path_str: str, alt: str) -> str:
    p = ROOT / path_str
    src = img_to_b64(p)
    if not src:
        return f'<div class="placeholder">[нет файла или слишком большой: {path_str}]</div>'
    return f'<img src="{src}" alt="{alt}" class="img-thumb">'


def section(title: str, content: str, color: str = "bg-slate-50") -> str:
    return f'''
<section class="{color} p-6 rounded-lg shadow-sm">
  <h2 class="text-2xl font-bold mb-4 text-slate-800">{title}</h2>
  {content}
</section>
'''


def kpi(label: str, value: str, sub: str = "", color: str = "text-slate-700") -> str:
    return f'''
<div class="bg-white rounded-lg p-4 shadow-sm border border-slate-200">
  <div class="text-xs uppercase tracking-wide text-slate-500">{label}</div>
  <div class="text-3xl font-bold {color} mt-1">{value}</div>
  <div class="text-sm text-slate-500 mt-1">{sub}</div>
</div>
'''


# ----- собираем данные -----

# метрика по видео из последнего прогона v4 (если есть)
v4_summary_path = ROOT / "outputs/pipeline_v4/summary.json"
v4_summary = {}
if v4_summary_path.exists():
    v4_summary = json.loads(v4_summary_path.read_text(encoding="utf-8"))

# метрика gt-bbox diag
gt_bbox_csv = ROOT / "outputs/diag_gt_bbox/26_12-20_with_net.csv"
gt_bbox_metrics = {
    "mean_score": 33,
    "qualified": 0,
    "fields_100": ["color", "price_discount", "wholesale_level_1_price",
                   "wholesale_level_2_count", "wholesale_level_2_price",
                   "action_price_qr", "action_code_qr"],
}

# характеризация датасета
data_stats = []
for name in ["25_12-20", "25_2-10", "26_12-20", "43_15", "49_5"]:
    p = ROOT / f"data/labeled/{name}/{name}.csv"
    if not p.exists(): continue
    df = pd.read_csv(p, encoding="utf-8")
    colors = list(df.get("color", pd.Series()).astype(str).unique())
    data_stats.append({
        "name": name, "rows": len(df),
        "colors": colors,
    })

dataset_table = '<table class="data-table"><thead><tr><th>Видео</th><th>GT ценников</th><th>Цвета</th></tr></thead><tbody>'
for d in data_stats:
    dataset_table += f'<tr><td><b>{d["name"]}</b></td><td>{d["rows"]}</td><td>{", ".join(d["colors"])}</td></tr>'
dataset_table += f'<tr class="total"><td><b>ИТОГО</b></td><td><b>{sum(d["rows"] for d in data_stats)}</b></td><td>—</td></tr>'
dataset_table += '</tbody></table>'

# гипотезы
hypotheses_html = ""
hyps = [
    ("QR/штрихкод декодирование", "❌ ЗАКРЫТО",
     "20+ комбинаций (pyzbar, zxing-cpp, PyBoof, WeChat, QReader) × 100 ценников × multi-frame fusion × lucky imaging × perspective rectify × Real-ESRGAN x4 = <b>0/100 матчей</b> с GT-штрихкодом. Sanity-check на синтетике: декодеры работают только от 80px QR, у нас 48px → физический предел. Сильный аргумент в питче.", "red"),
    ("Red-mask детектор для красных ценников", "✅ Работает",
     "Простой <code>cv2.inRange</code> по HSV красного + connected components → <b>~60% recall</b> на размеченных видео. Не требует обучения, детерминирован.", "green"),
    ("Multicolor detector (red/yellow/green)", "🟡 Частично",
     "В видео 49_5 есть жёлтые ценники → multicolor нужен. Но даёт <b>много false positive</b> на жёлтых этикетках/зелёных упаковках товаров. Нужна доводка.", "yellow"),
    ("Multi-frame stacking / Lucky imaging", "❌ Не помогло для QR",
     "Median по 30+ выровненным кадрам + Lucky imaging top-K sharpest + ECC alignment → визуально картинка чище, но декодеры всё равно не справляются.", "red"),
    ("RapidOCR (Apache 2.0) для текста", "✅ Работает",
     "На правильных crops с padding +25% и upscale ×3 — читает русский+латиницу. Видели 'UOULNDELA' = MOULIN DE LA. На кропах от нашего детектора (с extension вниз) — пасует, только обрывки 'ca', 'mm'.", "green"),
    ("Дефолт 'нет' для 7 always-нет полей", "✅ ОГРОМНЫЙ win",
     "В GT для 7 полей (price_discount, wholesale_*_count/price, action_*_qr) — <b>всегда 'нет'</b> (для красных ценников). Если по умолчанию писать 'нет' → +28 п.п. mean score без касания OCR.", "green"),
]
for title, status, desc, color in hyps:
    cls = {"red": "bg-red-50 border-red-200", "yellow": "bg-yellow-50 border-yellow-200",
           "green": "bg-green-50 border-green-200"}[color]
    hypotheses_html += f'''
    <div class="rounded-lg p-4 border {cls} mb-3">
      <div class="flex items-center justify-between mb-1">
        <h3 class="font-semibold text-slate-800">{title}</h3>
        <span class="text-sm font-medium">{status}</span>
      </div>
      <p class="text-sm text-slate-700">{desc}</p>
    </div>
    '''

# pipeline diagram (Mermaid)
pipeline_diagram = '''
<pre class="mermaid">
flowchart TD
    A[Видео 4K] --> B[Sampling ~3 fps]
    B --> C[Red-mask detector<br/>HSV → connected comp]
    C --> D[Global NMS]
    D --> E[IoU Tracker<br/>1 ценник = 1 track_id]
    E --> F[Best-frame по sharpness<br/>Laplacian variance]
    F --> G[Upscale ×3]
    G --> H[RapidOCR<br/>детекция + распознавание текста]
    H --> I[Парсер по regex/правилам<br/>price, discount, barcode, name, date]
    I --> J[Дефолт 'нет' для<br/>7 always-нет полей]
    J --> K[CSV writer]
    K --> L[Match с GT<br/>by barcode OR spatial-temporal]
    L --> M[Метрика: % строк с ≥80% полей]
</pre>
'''

# OCR example
ocr_example = f'''
<h3 class="font-semibold mt-4 mb-2">Один и тот же ценник, разные crops → разный OCR:</h3>
<div class="grid grid-cols-2 gap-4 my-3">
  <div>
    <div class="font-medium text-slate-700">GT bbox с padding +25%, upscale x3 (правильно):</div>
    {img_or_placeholder("outputs/ocr_compare/idx00_GT_pad25_x3.jpg", "GT pad25 x3")}
    <code class="block mt-2 text-xs p-2 bg-slate-100 rounded">OCR: ['302345', 'oo', 'FArE Une Wyey', 'UOULNDELA']<br>→ читается "MOULIN DE LA"!</code>
  </div>
  <div>
    <div class="font-medium text-slate-700">Bbox от нашего detector v5 (плохо):</div>
    {img_or_placeholder("outputs/ocr_compare/idx00_PRED.jpg", "PRED")}
    <code class="block mt-2 text-xs p-2 bg-slate-100 rounded">OCR: ['I']<br>→ один обрывок!</code>
  </div>
</div>
<p class="text-sm text-slate-600">Это значит: OCR работает на правильных crops, проблема — наш детектор даёт неоптимальные bbox.</p>
'''

# QR rescue gallery
qr_rescue = f'''
<h3 class="font-semibold mt-4 mb-2">Что мы пробовали для QR (всё 0 матчей):</h3>
<div class="grid grid-cols-3 gap-3">
  <div><div class="font-medium text-xs">Raw GT crop</div>{img_or_placeholder("outputs/padded_crops/26_12-20_sample0_ts27156_barcode4690491103135.jpg", "raw")}</div>
  <div><div class="font-medium text-xs">Multi-frame stacking</div>{img_or_placeholder("outputs/stacking_debug/43_15_idx0_05_median_super_3x.jpg", "stack")}</div>
  <div><div class="font-medium text-xs">Real-ESRGAN x4 (галлюцинация)</div>{img_or_placeholder("outputs/esrgan_on_qr/26_12-20_idx02_n20_miss_GT3552848940002_SR4x.jpg", "esrgan")}</div>
</div>
<p class="text-sm text-slate-600 mt-2">ESRGAN сделал 3 угловых finder pattern чёткими, но модули превратил в "волнистые бактерии" — натренирован на natural images, не QR.</p>
'''

# detector example
detector_example = f'''
<h3 class="font-semibold mt-4 mb-2">Red-mask детектор на одном кадре:</h3>
{img_or_placeholder("outputs/redmask_detector/43_15_ts002472.jpg", "det")}
<p class="text-sm text-slate-600 mt-2">Зелёные — GT bbox от организаторов. Синие — наши предсказания. Recall ~60% на красных ценниках.</p>
'''

# metrics
metrics_html = '''
<div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
  ''' + kpi("Train ценников", "274", "5 размеченных видео", "text-blue-600") + '''
  ''' + kpi("Recall детектора", "~60%", "red-mask, без обучения", "text-green-600") + '''
  ''' + kpi("Mean field score", "33%", "на GT-bbox + дефолт 'нет'", "text-amber-600") + '''
  ''' + kpi("Qualified ≥80%", "0%", "главная метрика — пока 0", "text-red-600") + '''
</div>
<table class="data-table">
  <thead><tr><th>Поле</th><th>Точность</th><th>Что это</th></tr></thead>
  <tbody>
    <tr class="bg-green-50"><td><code>color</code></td><td>100%</td><td>детерминирован (red/yellow от детектора)</td></tr>
    <tr class="bg-green-50"><td><code>price_discount</code> + 6 QR-полей</td><td>100% × 7</td><td>дефолт "нет" (всегда отсутствуют на акционных)</td></tr>
    <tr class="bg-amber-50"><td><code>discount_amount</code></td><td>11%</td><td>regex "-XX%" работает, но OCR не везде видит</td></tr>
    <tr class="bg-red-50"><td><code>product_name</code></td><td>0%</td><td>OCR на текущих crops пасует — нужен фикс детектора</td></tr>
    <tr class="bg-red-50"><td><code>price_card / price_default</code></td><td>0%</td><td>парсер путает или OCR не достаёт цифры</td></tr>
    <tr class="bg-red-50"><td><code>barcode</code></td><td>0%</td><td>OCR ломает 13 цифр на куски, склейка не находит</td></tr>
    <tr class="bg-red-50"><td><code>id_sku, print_datetime, code, additional_info</code></td><td>0%</td><td>в работе</td></tr>
    <tr class="bg-slate-50"><td><code>qr_code_barcode, price1..4_qr</code></td><td>0%</td><td>недостижимо — QR физически не декодируется</td></tr>
  </tbody>
</table>
'''

# roadmap
roadmap_html = '''
<table class="data-table">
  <thead><tr><th>День</th><th>Дата</th><th>Цель</th><th>Статус</th></tr></thead>
  <tbody>
    <tr><td>Д1-2</td><td>13-14.05</td><td>Репо, EDA, baseline детектор, QR-эксперименты</td><td>✅ Done</td></tr>
    <tr><td>Д3</td><td>15.05</td><td>End-to-end pipeline видео→CSV + первая метрика</td><td>✅ Done (mean=33%)</td></tr>
    <tr class="bg-yellow-50"><td>Д4</td><td>16.05</td><td>Улучшение парсера: product_name, prices, barcode → достичь qualified > 30%</td><td>🟡 In progress</td></tr>
    <tr><td>Д5</td><td>17.05</td><td>Gradio UI + (опц.) fine-tune YOLO детектора</td><td>⏳</td></tr>
    <tr><td>Д6</td><td>18.05</td><td>Деплой на HF Spaces + презентация</td><td>⏳</td></tr>
    <tr class="bg-red-50"><td>Д7</td><td><b>19.05 до 15:00</b></td><td><b>САБМИТ</b></td><td>⏳</td></tr>
  </tbody>
</table>
'''

# tested vs untested
exp_table = '''
<h3 class="font-semibold mt-4 mb-2">Гипотезы по технологиям:</h3>
<table class="data-table">
  <thead><tr><th>Технология</th><th>Лицензия</th><th>Результат</th></tr></thead>
  <tbody>
    <tr><td><b>OpenFoodFacts YOLOv11x</b> (детектор ценников)</td><td>AGPL ❌</td><td>0% recall на наших видео (обучен на других ценниках)</td></tr>
    <tr><td><b>red-mask HSV</b> детектор</td><td>—</td><td>✅ 60% recall, текущий выбор</td></tr>
    <tr><td><b>YOLOX-Nano</b> (план для production)</td><td>Apache 2.0 ✅</td><td>Не пробовали, в плане fine-tune</td></tr>
    <tr><td><b>pyzbar / zxing-cpp / PyBoof / WeChat QR</b></td><td>MIT/Apache ✅</td><td>0/100 для QR Ленты (слишком мелкий)</td></tr>
    <tr><td><b>QReader</b> (YOLOv8 + autopipeline)</td><td>MIT ✅</td><td>0/100, та же причина</td></tr>
    <tr><td><b>Real-ESRGAN x4</b> для QR</td><td>BSD-3 ✅</td><td>0/100, галлюцинирует структуру</td></tr>
    <tr><td><b>RapidOCR</b> (русский+латиница)</td><td>Apache 2.0 ✅</td><td>Работает на правильных crops</td></tr>
    <tr><td><b>ByteTrack</b> между кадрами</td><td>MIT ✅</td><td>Не используем (свой простой IoU-tracker)</td></tr>
  </tbody>
</table>
'''

# next steps
next_steps = '''
<ol class="list-decimal pl-6 space-y-2">
  <li><b>Pipeline v7</b>: padding +25% во все стороны (вместо extension +60% вниз) + upscale x3 + дефолт "нет" → ожидаем mean 30-40% на реальных детекциях</li>
  <li><b>Фикс discount_amount</b>: разобраться почему теряем на 60+ ценниках, OCR-output не всегда содержит "-XX%"</li>
  <li><b>Парсер product_name</b>: текущий слишком строг, на правильных crops может вытаскивать имена</li>
  <li><b>Парсер price_card / price_default</b>: различение по высоте шрифта в OCR-output</li>
  <li><b>Парсер barcode</b>: склейка соседних цифровых сегментов в 13-значную строку</li>
  <li><b>Gradio UI</b>: загрузка видео → запуск пайплайна → выдача CSV</li>
  <li><b>Деплой на HF Spaces</b></li>
  <li><b>Презентация</b>: главные слайды + демо</li>
</ol>
'''

# build HTML
html = f'''<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Lenta Shelf Control — Dashboard</title>
<script src="https://cdn.tailwindcss.com"></script>
<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({{startOnLoad:true, theme:'neutral'}});
</script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
  .img-thumb {{ max-width: 100%; max-height: 400px; border-radius: 6px; border: 1px solid #e5e7eb; }}
  .placeholder {{ padding: 20px; background: #f3f4f6; color: #6b7280; border-radius: 6px; text-align: center; }}
  .data-table {{ width: 100%; border-collapse: collapse; }}
  .data-table th, .data-table td {{ padding: 6px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top; }}
  .data-table th {{ background: #f9fafb; font-weight: 600; }}
  .data-table tr.total {{ background: #eff6ff; font-weight: 600; }}
  code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 3px; font-size: 0.9em; }}
</style>
</head>
<body class="bg-slate-100 p-4 md:p-8">

<div class="max-w-6xl mx-auto space-y-6">

  <header class="bg-gradient-to-r from-blue-600 to-indigo-700 text-white p-8 rounded-lg shadow-md">
    <h1 class="text-3xl font-bold">Lenta Shelf Control</h1>
    <p class="text-blue-100 mt-2">Хакатон Lenta Tech Life Hack — распознавание ценников с видео робота</p>
    <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3">
      <div class="bg-white/10 rounded p-3"><div class="text-xs">Дедлайн</div><div class="text-xl font-bold">19.05 15:00</div></div>
      <div class="bg-white/10 rounded p-3"><div class="text-xs">Дней осталось</div><div class="text-xl font-bold">~4</div></div>
      <div class="bg-white/10 rounded p-3"><div class="text-xs">Команда</div><div class="text-xl font-bold">4 чел</div></div>
      <div class="bg-white/10 rounded p-3"><div class="text-xs">Призовой фонд</div><div class="text-xl font-bold">600 000 ₽</div></div>
    </div>
  </header>

  {section("📊 Текущие метрики", metrics_html)}

  {section("🗓 Roadmap", roadmap_html)}

  {section("🔬 Архитектура пайплайна", pipeline_diagram)}

  {section("📦 Датасет", dataset_table + '<p class="text-sm text-slate-600 mt-3">Видео 4K @ 20 fps, фишай-объектив, мобильный робот. На public-наборе все ценники <b>красные</b> кроме 49_5 (есть жёлтые). Финальная проверка организаторами — на private-видео.</p>')}

  {section("🎯 Гипотезы и результаты", hypotheses_html)}

  {section("🔍 Визуальные примеры", ocr_example + qr_rescue + detector_example)}

  {section("🛠 Технологии (audit лицензий)", exp_table)}

  {section("➡️ Следующие шаги", next_steps)}

  <footer class="text-center text-xs text-slate-500 py-6">
    Сгенерировано <code>scripts/build_dashboard.py</code> · Репо: <a class="text-blue-600 hover:underline" href="https://github.com/letmly/lenta-shelf-control">github.com/letmly/lenta-shelf-control</a>
  </footer>
</div>

</body>
</html>
'''

OUT.write_text(html, encoding="utf-8")
print(f"[OK] Dashboard generated: {OUT}")
print(f"     Size: {OUT.stat().st_size / 1024:.1f} KB")
