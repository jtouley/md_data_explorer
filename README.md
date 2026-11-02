# Clinical Analytics Platform

Polars-first clinical analytics: user-uploaded datasets, semantic layer, natural-language queries, and statistical workflows. The **desktop UI** is **Electron** (under `electron/`), backed by a local **FastAPI** server; Streamlit has been removed—see the [omnibus spec](docs/specs/omnibus_electron_streamlit_cutover.md).

**Reviewer / architect entry points:** [As-built architecture](docs/architecture/AS_BUILT_ARCHITECTURE.md) · [Module boundaries](docs/architecture/ARCHITECTURE_BOUNDARIES.md) · [Threat model](docs/security/THREAT_MODEL.md)

---

## Developers

```bash
make install-dev          # uv sync --extra dev --group dev
make test-fast            # Fast tests (matches CI test job)
make check                # lint, format-check, type-check, tests
make run                  # FastAPI backend at http://127.0.0.1:8000 (alias of run-api)
cd electron && npm ci && npm start   # Electron desktop app (needs the backend running)
```

Optional pytest flags: `make test-ui-serial PYTEST_ARGS='-k foo -xvs'`. See [tests/AGENTS.md](tests/AGENTS.md) and [docs/development/testing.md](docs/development/testing.md).

**Built-in demo datasets** (COVID-MS, Sepsis, MIMIC-III) described in older marketing copy are **removed**; the app is **user-uploaded data** only. Details: [docs/specs/IMPLEMENTATION_STATUS.md](docs/specs/IMPLEMENTATION_STATUS.md) (historical) and [docs/architecture/dataset-registry.md](docs/architecture/dataset-registry.md).

---

## Overview

- **Upload-driven workflows:** CSV/Excel and related paths through the dataset registry and semantic layer
- **Question-driven analysis:** NL query engine with pattern, embedding, and LLM fallbacks (see `docs/`)
- **Statistical engine:** Descriptive, group comparison, survival, prediction, correlations (see `src/clinical_analytics/analysis/`)
- **Tech stack:** Python 3.11+, Polars, DuckDB, Ibis, FastAPI; Electron desktop UI

---

## User guide

### Quick start

**Mac/Linux** (sets up Ollama, starts the API, opens the desktop app):

```bash
./scripts/run_app.sh
```

**Or, in two terminals:**

```bash
make run                           # FastAPI backend on http://127.0.0.1:8000
cd electron && npm ci && npm start # Electron desktop app
```

API docs: `http://127.0.0.1:8000/api/docs`

### Interface (conceptual)

- **Dataset bar:** Upload a file or pick an uploaded dataset
- **Sidebar:** Chat sessions per dataset
- **Main area:** Dataset overview and preview, natural-language questions and results

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8000 in use | Stop the other process, or run `uv run uvicorn clinical_analytics.api.main:app --port 8001` (the desktop app expects 8000) |
| Module not found | `make install-dev` |
| Data not loading | Confirm uploads completed; see `docs/getting-started/uploading-data.md` |

### Getting help

- **Docs index:** [docs/index.md](docs/index.md)
- **Implementation status (historical):** [docs/specs/IMPLEMENTATION_STATUS.md](docs/specs/IMPLEMENTATION_STATUS.md)

---

## Installation

This project uses [uv](https://github.com/astral-sh/uv):

```bash
make install-dev
```

## Project structure

```
src/clinical_analytics/
├── api/            # FastAPI app and routes
├── core/           # Semantic layer, NL engine, schemas, query service
├── datasets/       # Dataset definitions (uploaded + registry)
├── analysis/       # Statistical methods
├── ui/             # Upload storage and enrichment helpers (no UI framework)
└── storage/        # Shared storage utilities
electron/           # Desktop shell (Vite + Electron)
tests/              # Pytest suites (see tests/AGENTS.md)
```

## License

TBD
