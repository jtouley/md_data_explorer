# Installation

## Prerequisites

- macOS 13 (Ventura) or later on Apple silicon or Intel; Windows 10/11 x64; Linux x86_64/aarch64 with glibc 2.28+. These are the platforms ONNX Runtime 1.23.2, the embedding runtime, ships wheels for (macOS 12 has none for Python 3.13). See `docs/development/packaging.md`.
- [uv](https://docs.astral.sh/uv/) - installs a supported Python for you (`pyproject.toml` requires `>=3.11,<3.14`)
- Node.js 22.12 or later and npm, for the Electron desktop app (Electron 43 needs it; CI uses Node 22)
- `make` on macOS/Linux
- Optional: [Ollama](https://ollama.com/download), a local LLM used for questions the pattern matcher cannot handle

Setup needs internet access: it downloads Python and Node packages, the `all-MiniLM-L6-v2` ONNX embedding model (`make install-dev` fetches it; pinned, SHA-256 checked; bundled with the app, never downloaded at runtime) and, if you use Ollama, a model (the app prefers `llama3.1:8b` and falls back to `llama3.2:3b`). After that the app runs locally.

## Install from source

```bash
git clone https://github.com/jtouley/md_data_explorer.git
cd md_data_explorer
make install-dev                 # uv sync --extra dev --group dev, then fetch the embedding model
cd electron && npm ci && cd ..
```

On Windows without `make`, run these two commands instead of `make install-dev`. The second matters: without the embedding model the backend refuses to start.

```powershell
uv sync --extra dev --group dev
uv run python scripts/fetch_embedding_model.py
```

## Run

| Platform | How |
|----------|-----|
| Any OS, including Windows | `cd electron && npm start`. The desktop app starts its own backend on `127.0.0.1` (port 8000, or a free port if 8000 is taken) with a fresh API token, and stops it when you quit. Start Ollama yourself if you want it: `ollama serve`, then `ollama pull llama3.1:8b` once. |
| macOS | `./scripts/run_app.sh` starts Ollama (installing it with Homebrew if missing), the API and the desktop app |
| Linux | Install Ollama yourself from <https://ollama.com/download> (the script will not), then `./scripts/run_app.sh` |

`./scripts/run_app.sh` uses port 8000 and stops if another program already holds it. If Ollama has no models, it downloads `llama3.2:3b`; run `ollama pull llama3.1:8b` for the app's preferred model.

For development, `make run` (macOS/Linux) starts the API on `http://127.0.0.1:8000` with auto-reload and **no authentication** (never use it with real patient data) and, if Ollama is running, pulls a missing model. To point the desktop app at it: `cd electron && CLINICAL_ANALYTICS_EXTERNAL_BACKEND=1 CLINICAL_ANALYTICS_DEV_NO_AUTH=1 npm start`. Started on its own, `uv run python -m clinical_analytics.api.serve` refuses to start without `CLINICAL_ANALYTICS_API_TOKEN` or `CLINICAL_ANALYTICS_DEV_NO_AUTH=1`.

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

`cd electron && npm start` picks a free port by itself. `./scripts/run_app.sh` and `make run` use port 8000: stop the other process first. For an API-only development server on another port (no authentication, never with real patient data):

```bash
CLINICAL_ANALYTICS_DEV_NO_AUTH=1 uv run python -m clinical_analytics.api.serve --port 8001
```

### Missing dependencies

If you see import errors, run `make install-dev` again (or `uv sync --extra dev --group dev --reinstall`).

### Questions are not understood

Check the backend's `/health` (for example `http://127.0.0.1:8000/health`): `ollama_reachable` and `ollama_default_model_available` should be `true`. Without Ollama, simpler questions still work.
