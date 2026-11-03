# Installation

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - installs a supported Python for you (`pyproject.toml` requires `>=3.11,<3.14`)
- Node.js 20 and npm, for the Electron desktop app (CI uses Node 20)
- `make` on macOS/Linux
- Optional: [Ollama](https://ollama.com/download), a local LLM used for questions the pattern matcher cannot handle

Setup needs internet access: it downloads Python and Node packages, and the first run downloads an Ollama model (`llama3.1:8b`, fallback `llama3.2:3b`) and the `all-MiniLM-L6-v2` embedding model from Hugging Face. After that the app runs locally.

## Install from source

```bash
git clone https://github.com/jtouley/md_data_explorer.git
cd md_data_explorer
make install-dev                 # uv sync --extra dev --group dev
cd electron && npm ci && cd ..
```

On Windows without `make`, run `uv sync --extra dev --group dev` instead of `make install-dev`.

## Run

| Platform | How |
|----------|-----|
| macOS | `./scripts/run_app.sh` starts Ollama (installing it with Homebrew if missing), the API and the desktop app |
| Linux | Install Ollama yourself from <https://ollama.com/download> (the script will not), then `./scripts/run_app.sh` |
| Windows, or any OS by hand | Terminal 1: `uv run uvicorn clinical_analytics.api.main:app --host 127.0.0.1 --port 8000`. Terminal 2: `cd electron && npm start`. Start Ollama with `ollama serve` and `ollama pull llama3.1:8b`. |

`make run` (macOS/Linux) starts the API with auto-reload and, if Ollama is running, pulls a missing model. The desktop window needs the API at `http://127.0.0.1:8000/health` to respond.

## Where your data lives

Uploads, the analytics database, chat sessions and NL prompt-learning files are stored in a per-user data home, never in the repository (`src/clinical_analytics/core/data_home.py`):

| OS | Default |
|----|---------|
| macOS | `~/Library/Application Support/ClinicalAnalytics` |
| Windows | `%LOCALAPPDATA%\ClinicalAnalytics` |
| Linux | `$XDG_DATA_HOME/clinical-analytics` (default `~/.local/share/clinical-analytics`) |

- `CLINICAL_ANALYTICS_DATA_HOME` moves the whole data home.
- `DATABASE_URL` moves only the chat-session database (default SQLite at `<data home>/data/clinical_analytics.db`).
- `OLLAMA_BASE_URL` (default `http://localhost:11434`) must stay a local address: LLM prompts can include column names, category labels and summary results.
- The app does not encrypt data at rest. Use FileVault (macOS), BitLocker (Windows) or full-disk encryption (Linux).
- Do not copy patient files into the repository; `make check-data` fails if git tracks data files.

This is local software, not a compliance certification.

## Troubleshooting

### Port already in use

Stop the other process on port 8000. The desktop app only talks to `127.0.0.1:8000`; for API-only use you can pick another port:

```bash
uv run uvicorn clinical_analytics.api.main:app --host 127.0.0.1 --port 8001
```

### Missing dependencies

If you see import errors, run `make install-dev` again (or `uv sync --extra dev --group dev --reinstall`).

### Questions are not understood

Check `http://127.0.0.1:8000/health`: `ollama_reachable` and `ollama_default_model_available` should be `true`. Without Ollama, simpler questions still work.
