# Clinical Analytics Platform

Polars-first clinical analytics: user-uploaded datasets, semantic layer, natural-language queries, and statistical workflows. The **interactive UI** is primarily **Streamlit**; **FastAPI** backs API clients; **Electron** (under `electron/`) is an in-progress desktop shell—see the [omnibus spec](docs/specs/omnibus_electron_streamlit_cutover.md).

**Reviewer / architect entry points:** [As-built architecture](docs/architecture/AS_BUILT_ARCHITECTURE.md) · [Module boundaries](docs/architecture/ARCHITECTURE_BOUNDARIES.md) · [Threat model](docs/security/THREAT_MODEL.md)

---

## Developers

```bash
make install-dev          # uv sync --extra dev --group dev
make test-fast            # Fast tests (matches CI test job)
make check                # lint, format-check, type-check, tests
make run                  # Streamlit (http://localhost:8501)
make run-api              # FastAPI (see Makefile / api for port)
```

Optional pytest flags: `make test-ui-serial PYTEST_ARGS='-k foo -xvs'`. See [tests/AGENTS.md](tests/AGENTS.md) and [docs/development/testing.md](docs/development/testing.md).

**Built-in demo datasets** (COVID-MS, Sepsis, MIMIC-III) described in older marketing copy are **removed**; the app is **user-uploaded data** only. Details: [docs/specs/IMPLEMENTATION_STATUS.md](docs/specs/IMPLEMENTATION_STATUS.md) (historical) and [docs/architecture/dataset-registry.md](docs/architecture/dataset-registry.md).

---

## Overview

- **Upload-driven workflows:** CSV/Excel and related paths through the dataset registry and semantic layer
- **Question-driven analysis:** NL query engine with pattern, embedding, and LLM fallbacks (see `docs/`)
- **Statistical engine:** Descriptive, group comparison, survival, prediction, correlations (see `src/clinical_analytics/analysis/`)
- **Tech stack:** Python 3.11+, Polars, DuckDB, Ibis, Streamlit; FastAPI for HTTP; Electron migration tracked in docs above

---

## User guide (Streamlit-oriented)

The section below describes a **sidebar dataset picker** and named public datasets. That matches **legacy Streamlit UX**; today you should expect **upload-first** flows. Keep the procedural tips (ports, troubleshooting) that still apply.

### Quick start (Streamlit)

**Mac/Linux:**

```bash
./scripts/run_app.sh
```

**Or:**

```bash
make run
```

Default URL: `http://localhost:8501`

### Interface (conceptual)

- **Sidebar:** Dataset / session selection (uploaded data in current product)
- **Main area:** Overview, preview, analyses, NL query where enabled

### Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 8501 in use | Stop other Streamlit apps or change port in Streamlit options |
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
├── ui/             # Streamlit app, pages, components, storage helpers
└── storage/        # Shared storage utilities
electron/           # Desktop shell (Vite + Electron)
tests/              # Pytest suites (see tests/AGENTS.md)
```

## License

TBD
