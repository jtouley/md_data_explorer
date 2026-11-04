# Clinical Analytics Platform

Ask questions about your own clinical dataset in plain English ("compare outcome by diabetes") and get the right statistical analysis back, on your own computer, without writing code.

You upload a CSV, Excel or SPSS file; the app detects patient IDs, outcomes and variable types, picks the analysis (descriptive statistics, group comparisons, survival, regression, correlations) and explains the result. It is a desktop app (Electron) talking to a local Python server (FastAPI) on `127.0.0.1:8000`.

## Where your data lives & privacy

The app runs locally after setup. It is **not a compliance certification**: whether you may use it with a given dataset is up to your institution's rules.

- **Your data is stored outside this repository**, in a per-user data home (`src/clinical_analytics/core/data_home.py`):

  | OS | Default data home |
  |----|-------------------|
  | macOS | `~/Library/Application Support/ClinicalAnalytics` |
  | Windows | `%LOCALAPPDATA%\ClinicalAnalytics` |
  | Linux | `$XDG_DATA_HOME/clinical-analytics` (default `~/.local/share/clinical-analytics`) |

  Uploads, the analytics database, chat sessions and NL prompt-learning files all go under `<data home>/data/`. Set `CLINICAL_ANALYTICS_DATA_HOME` to use another folder. Set `DATABASE_URL` only if you want chat sessions in a different database (default: SQLite at `<data home>/data/clinical_analytics.db`).
- **Encrypt the disk.** Data at rest is not encrypted by the app; turn on FileVault (macOS) or BitLocker (Windows), or full-disk encryption on Linux.
- **Never copy patient files into this repo.** `make check-data` (also part of `make check-fast` and CI) fails if git tracks data files.
- **The LLM is local.** Harder questions are sent to [Ollama](https://ollama.com) at `OLLAMA_BASE_URL` (default `http://localhost:11434`). Prompts can include column names, category labels and summary results. **Do not point `OLLAMA_BASE_URL` at a remote host** - that would send them off your machine. Without Ollama, simpler questions still work through pattern and embedding matching.
- **Setup needs the internet; daily use does not.** Installing downloads Python and Node packages, and the first run downloads models: an Ollama model (`llama3.1:8b`, fallback `llama3.2:3b`, from `src/clinical_analytics/resources/config/ollama.yaml`) and the `all-MiniLM-L6-v2` sentence-transformers model from Hugging Face.
- **Small counts are suppressed.** Patient counts from 1 to 10 are shown as `<11` in answers (`small_cell_threshold` in `core/config_loader.py`).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it installs a Python between 3.11 and 3.13 for you; `pyproject.toml` requires `>=3.11,<3.14`)
- Node.js 20 and npm (the version CI uses, `.github/workflows/ci.yml`)
- `make` (macOS/Linux; see Windows below)
- Optional: [Ollama](https://ollama.com/download) for questions the pattern matcher cannot handle

## Install

```bash
git clone https://github.com/jtouley/md_data_explorer.git
cd md_data_explorer
make install-dev          # uv sync --extra dev --group dev
cd electron && npm ci && cd ..
```

## Launch

**macOS:** one command starts Ollama (installing it with Homebrew if missing), the API and the desktop window:

```bash
./scripts/run_app.sh
```

**Linux:** the same script works, but it will not install Ollama for you. Install Ollama from <https://ollama.com/download> first (or skip it), then run `./scripts/run_app.sh`.

**Any OS, two terminals** (this is the Windows path; `run_app.sh` is a bash script):

```bash
# Terminal 1: the API. On macOS/Linux, `make run` does this and, if Ollama is running, pulls a missing model.
uv run python -m clinical_analytics.api.serve --port 8000

# Terminal 2: the desktop app (needs the API running)
cd electron
npm start
```

On Windows without `make`, install with `uv sync --extra dev --group dev` instead of `make install-dev`, and start Ollama yourself (`ollama serve`, then `ollama pull llama3.1:8b`).

## Your first question

1. In the desktop app, click **Load sample**. This loads a synthetic sample cohort that ships with the app (`src/clinical_analytics/resources/demo/sample_cohort.csv`: patient ID, age, sex, diabetes, hypertension and a binary outcome). It contains no real patients.
2. Ask: `compare outcome by diabetes`
3. Then try `what predicts outcome`.

When you are ready, use **Upload dataset** for your own `.csv`, `.xlsx`, `.xls` or `.sav` file. See [Uploading data](docs/getting-started/uploading-data.md).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8000 in use | Stop the other process. The desktop app only talks to `127.0.0.1:8000`. |
| `Module not found` | `make install-dev` |
| Desktop window shows no data | Check the API is up: `http://127.0.0.1:8000/health` |
| Complex questions are not understood | Start Ollama; `http://127.0.0.1:8000/health` shows `ollama_reachable` and `ollama_default_model_available` |

## Developers

```bash
make install-dev          # Python deps incl. dev tools
make test-fast            # Fast tests (matches CI test job)
make check-fast           # Lint, format check, tracked-data guard (no tests)
make check                # Lint, format check, type check, tests
make format && make lint-fix
make run                  # API at http://127.0.0.1:8000 with --reload (alias of run-api)
```

Run tests through `make` targets, not `pytest` directly. Extra pytest flags: `make test-ui-serial PYTEST_ARGS='-k foo -xvs'`. See [tests/AGENTS.md](tests/AGENTS.md) and [docs/development/testing.md](docs/development/testing.md). API docs (OpenAPI): `http://127.0.0.1:8000/api/docs`.

Architecture: [As-built architecture](docs/architecture/AS_BUILT_ARCHITECTURE.md) · [Module boundaries](docs/architecture/ARCHITECTURE_BOUNDARIES.md) · [Threat model](docs/security/THREAT_MODEL.md) · [Docs index](docs/index.md)

```
src/clinical_analytics/
├── api/            # FastAPI app and routes
├── core/           # Semantic layer, NL engine, schemas, query service, data home
├── datasets/       # Uploaded-dataset definitions, registry, bundled sample cohort
├── analysis/       # Statistical methods
├── ui/             # Upload storage and enrichment helpers (no UI framework)
└── storage/        # Shared storage utilities
electron/           # Desktop shell (Vite + Electron)
tests/              # Pytest suites (see tests/AGENTS.md)
```

## Project history

- The Streamlit UI was removed in favour of Electron + FastAPI: see the [omnibus spec](docs/specs/omnibus_electron_streamlit_cutover.md).
- The built-in COVID-MS, Sepsis and MIMIC-III dataset loaders from older docs are gone. The only bundled data is the synthetic sample cohort above; everything else is data you upload. Details: [dataset registry](docs/architecture/dataset-registry.md), [implementation status (historical)](docs/specs/IMPLEMENTATION_STATUS.md).

## License

[MIT](LICENSE). Dataset files and user-uploaded content remain under their own licenses and terms; the application license does not relicense those materials.
