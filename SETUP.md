# SETUP — окружение разработки команды

> Шаги для разово настроить машину под проект. Делается каждым участником один раз.

## 1. Клон репозитория

```bash
git clone https://github.com/letmly/lenta-shelf-control.git
cd lenta-shelf-control
```

## 2. Python-окружение

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

> На Windows может потребоваться поставить **Visual C++ Build Tools** для сборки `pyzbar` и `paddleocr`. Либо использовать WSL/Docker.

## 3. Скачать видео

Видео не в репе. Ссылка — в `data/README.md` (TBD).

```bash
python scripts/fetch_data.py
```

## 4. (Опционально) Claude Code + Notion MCP

Тимлид настраивает Notion-доску команды. Чтобы можно было перетаскивать задачи прямо из Claude Code:

```bash
# Установить Notion MCP (вариант: notion-mcp-server)
claude mcp add notion -- npx -y @notionhq/notion-mcp-server
```

Затем:
1. Создаёшь integration на https://www.notion.so/profile/integrations
2. Копируешь internal integration token
3. В Notion: расшариваешь workspace/page на эту integration (`···` → `Connect to` → ваш integration)
4. Экспортируешь токен: `export NOTION_TOKEN=secret_xxx` (или прописываешь в `~/.claude.json` под mcpServers)
5. Перезапускаешь Claude Code: `/restart` или новый сеанс
6. Проверяешь: `/mcp` → видишь `notion`

После этого Claude сможет:
- читать/писать страницы Notion
- двигать задачи в kanban
- создавать страницы по шаблонам из `notion/*.md`

## 5. (Опционально) Telegram MCP для парсинга чата хака

Только если согласен на риск антифрода:
1. Завести **отдельный** TG-аккаунт (не основной).
2. Войти в чат хакатона.
3. Установить MCP:
   ```bash
   claude mcp add telegram -- npx -y telegram-mcp
   # или: chaindead/telegram-mcp (см. README)
   ```
4. Авторизация — через session string (выдаётся при первом логине).
5. Polling редкий (раз в 5–10 минут), только чтение.

## 6. Установка `gh` CLI (для PR'ов)

```bash
# Windows:
winget install GitHub.cli
# macOS:
brew install gh
# Linux:
sudo apt install gh

gh auth login
```

## 7. Pre-commit (опционально)

```bash
pip install pre-commit
pre-commit install
```

(Хук-файлы добавим позже, если потребуется ruff/black-формат.)
