# Installation

## Prerequisites

- Python 3.11 or higher
- Node.js 20+ (for the Electron desktop app)
- pip or uv package manager

## Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/md_data_explorer.git
cd md_data_explorer

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

## Verify Installation

```bash
make run                            # FastAPI backend on http://127.0.0.1:8000
cd electron && npm ci && npm start  # Electron desktop app (separate terminal)
```

Or run `./scripts/run_app.sh`, which also sets up Ollama. The desktop window opens once the API at `http://127.0.0.1:8000/health` responds.

## Dependencies

The platform requires:

- **Data Processing**: pandas, polars, duckdb
- **Statistical Analysis**: scipy, statsmodels, lifelines
- **API**: fastapi, uvicorn
- **Desktop UI**: Electron (Node, under `electron/`)
- **NL Query Engine**: sentence-transformers, scikit-learn
- **Semantic Layer**: ibis-framework

All dependencies are automatically installed during setup.

## Troubleshooting

### Port Already in Use

If port 8000 is already in use, stop the other process. The desktop app expects the API on 8000; for API-only use you can run:

```bash
uv run uvicorn clinical_analytics.api.main:app --host 127.0.0.1 --port 8001
```

### Missing Dependencies

If you see import errors:

```bash
uv sync --reinstall
```

### DuckDB Issues

If DuckDB fails to load:

```bash
pip install --upgrade duckdb
```
