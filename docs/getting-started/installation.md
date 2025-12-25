# Installation

## Prerequisites

- Python 3.10 or higher
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
# Run the application
streamlit run src/clinical_analytics/ui/app.py

# Or use the CLI
clinical-analytics serve
```

The application should open in your browser at `http://localhost:8501`.

## Dependencies

The platform requires:

- **Data Processing**: pandas, polars, duckdb
- **Statistical Analysis**: scipy, statsmodels, lifelines
- **UI**: streamlit
- **NL Query Engine**: sentence-transformers, scikit-learn
- **Semantic Layer**: ibis-framework

All dependencies are automatically installed during setup.

## Troubleshooting

### Port Already in Use

If port 8501 is already in use:

```bash
streamlit run src/clinical_analytics/ui/app.py --server.port 8502
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
