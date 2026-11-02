# Installation

## Prerequisites

- **Python 3.11 or higher**
- **[uv](https://docs.astral.sh/uv/)** (required — do not use bare `pip` for this project)
- **make**

## Install from source

```bash
git clone https://github.com/jtouley/md_data_explorer.git
cd md_data_explorer
make install-dev
```

`make install-dev` creates `.venv` via uv and installs the package in editable mode with dev extras.

## Run

| Command | What it starts |
|---------|----------------|
| `make run` | Streamlit UI (primary) at `http://localhost:8501` |
| `make run-api` | FastAPI at `http://127.0.0.1:8000` (experimental / Electron) |
| `./scripts/run_app.sh` | Streamlit with optional Ollama lifecycle helpers |

## Verify without the UI

```bash
make headless-preview
```

Prints JSON with `columns` and `row_count` for the bundled sample cohort (or pass `ARGS='--csv path/to/file.csv'`).

## Dependencies (high level)

- **Data:** Polars, DuckDB, Ibis
- **Stats:** scipy, statsmodels, lifelines
- **UI:** Streamlit (V1)
- **NL query:** sentence-transformers / optional Ollama

Pandas is not used in new application code.

## Troubleshooting

### Port 8501 in use

```bash
uv run streamlit run src/clinical_analytics/ui/app.py --server.port 8502
```

### Missing environment

```bash
make install-dev
```

### Import errors after pull

```bash
uv sync
```
