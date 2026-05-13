"""Gradio UI: загрузка видео → CSV с результатом распознавания."""
from __future__ import annotations

from pathlib import Path

# import gradio as gr   # TODO Д5: подключить
# from pipeline.pipeline import process_video


def run(video_path: str) -> str:
    """Запускает пайплайн и возвращает путь к CSV."""
    raise NotImplementedError("Д5: подключить process_video и Gradio")


def build_app():  # noqa: ANN201
    """Собирает Gradio-интерфейс."""
    raise NotImplementedError("Д5")


if __name__ == "__main__":
    # build_app().launch(server_name="0.0.0.0", server_port=7860)
    print("Stub. Реализация в Д5.")
