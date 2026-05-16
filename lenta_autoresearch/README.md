# Lenta autoresearch

Автономный exp loop по мотивам [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
Агент сам редактирует `pipeline_experiment.py`, гоняет на фиксированном subset, логирует.

## Файлы

- `prepare_lenta.py` — **read-only** eval harness. Возвращает `qualified_pct`, `mean_score`, `matched_pct`.
- `pipeline_experiment.py` — **агент модифицирует**. Полный pipeline на 1 видео.
- `program.md` — инструкции агенту (skill).
- `results.tsv` — лог экспериментов.
- `run.log` — stdout последнего запуска (untracked).
- `_runs/` — pred CSVs последнего запуска (untracked).

## Subset

- Video: `26_12-20.mp4` (71 GT-ценник)
- Меньшее видео из 5 → быстрее iter (~5-8 мин/run)

## Метрика

- **Primary**: `qualified_pct` — % ценников где ≥80% полей правильно. Главная цель.
- **Tie-break**: `mean_score` — средний % полей.

## Запуск автономно (агент)

```bash
# 1. Создать ветку
cd D:/.dev/Products/hacks/lenta
git checkout -b autoresearch/<tag>

# 2. Запустить агент с program.md
# В новой сессии Claude Code:
"Hi! Read lenta_autoresearch/program.md and kick off the experiment loop. Setup first."
```

Агент будет:
- Менять один knob за раз
- Делать commit
- Запускать `pipeline_experiment.py`
- Grep метрику
- Если улучшилась → keep + продолжить
- Иначе → `git reset --hard HEAD~1`
- Логировать в `results.tsv`
- Повторять (12+ экспериментов в час)

## Запуск вручную (один тест)

```bash
D:/.dev/Products/hacks/lenta/.venv/Scripts/python.exe lenta_autoresearch/pipeline_experiment.py
```

Output формат:
```
---
matched:       29/71 (0.4085)
mean_score:    0.1605
qualified:     0/71 (0.0000)
n_pred:        152
main_metric:   0.000000
tie_break:     0.160457
```
