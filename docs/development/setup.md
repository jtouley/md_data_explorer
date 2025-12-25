# Development Setup

## Environment Setup

### 1. Install Prerequisites

```bash
# Python 3.10+
python --version

# uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip
pip install --upgrade pip
```

### 2. Clone Repository

```bash
git clone https://github.com/yourusername/md_data_explorer.git
cd md_data_explorer
```

### 3. Create Virtual Environment

```bash
# With uv (recommended)
uv venv
source .venv/bin/activate  # Linux/Mac
# Or: .venv\Scripts\activate  # Windows

# Or with venv
python -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
# Install all dependencies including dev dependencies
uv sync

# Or with pip
pip install -e ".[dev]"
```

## IDE Configuration

### VS Code

Install extensions:

- Python (Microsoft)
- Pylance
- Black Formatter
- isort

Settings (`.vscode/settings.json`):

```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.sortImports.args": ["--profile", "black"]
}
```

### PyCharm

1. Set Python interpreter to `.venv/bin/python`
2. Enable "Black" as code formatter
3. Enable "isort" for import sorting
4. Configure mypy as external tool

## Running the Application

### Streamlit UI

```bash
streamlit run src/clinical_analytics/ui/app.py
```

Opens at `http://localhost:8501`

### CLI (if implemented)

```bash
clinical-analytics serve
clinical-analytics analyze --dataset covid-ms --query "compare mortality by treatment"
```

## Development Tools

### Code Formatting

```bash
# Format all code
black src/ tests/
isort src/ tests/

# Check formatting without changes
black --check src/ tests/
```

### Type Checking

```bash
# Run mypy
mypy src/

# With strict mode
mypy --strict src/
```

### Linting

```bash
# Run flake8
flake8 src/ tests/

# Run pylint
pylint src/
```

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/clinical_analytics --cov-report=html

# Run specific test
pytest tests/test_nl_query_engine.py::test_pattern_matching

# Run with verbose output
pytest tests/ -v -s
```

### Documentation

```bash
# Serve documentation locally
mkdocs serve

# Build documentation
mkdocs build

# Deploy to GitHub Pages
mkdocs gh-deploy
```

## Directory Structure

```
md_data_explorer/
├── src/
│   └── clinical_analytics/
│       ├── core/              # Core logic
│       ├── datasets/          # Built-in datasets
│       ├── ui/                # Streamlit UI
│       └── utils/             # Utilities
├── tests/
│   ├── fixtures/              # Test data
│   ├── test_core/             # Core tests
│   └── test_ui/               # UI tests
├── docs/                      # MkDocs documentation
├── data/                      # Sample datasets
├── .github/                   # GitHub Actions workflows
├── pyproject.toml             # Project metadata and dependencies
├── mkdocs.yml                 # Documentation config
└── README.md                  # Project README
```

## Database Setup

### DuckDB

No setup required - DuckDB runs in-memory by default.

For persistent storage:

```python
import duckdb

# Create persistent database
conn = duckdb.connect('clinical_data.db')

# Use in semantic layer
semantic_layer = SemanticLayer(df, config, backend='duckdb', db_path='clinical_data.db')
```

## Environment Variables

Create `.env` file (not committed to git):

```bash
# Optional: Anthropic API key for Tier 3 LLM fallback
ANTHROPIC_API_KEY=your_key_here

# Optional: Logging level
LOG_LEVEL=INFO

# Optional: Data directory
DATA_DIR=./data
```

Load with:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Troubleshooting

### Import Errors

```bash
# Reinstall in editable mode
pip install -e .
```

### DuckDB Issues

```bash
# Upgrade DuckDB
pip install --upgrade duckdb
```

### Streamlit Port Conflicts

```bash
# Use different port
streamlit run src/clinical_analytics/ui/app.py --server.port 8502
```

### Test Failures

```bash
# Clear pytest cache
pytest --cache-clear

# Run with verbose output
pytest tests/ -v -s --tb=short
```

### Type Checking Errors

```bash
# Ignore specific errors
mypy src/ --no-strict-optional
```

## Performance Profiling

### Profile Code

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
run_analysis(df, query)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Profile Memory

```python
from memory_profiler import profile

@profile
def analyze_large_dataset(df):
    # Memory-intensive operations
    pass
```

## Debugging

### Streamlit Debugging

```python
import streamlit as st

# Add debug info
st.write("Debug:", variable)

# Use expander for verbose output
with st.expander("Debug Info"):
    st.json(debug_dict)
```

### Pytest Debugging

```bash
# Drop into debugger on failure
pytest tests/ --pdb

# Drop into debugger at start of test
pytest tests/ --trace
```

### VS Code Debugging

Add `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Streamlit",
      "type": "python",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "src/clinical_analytics/ui/app.py"]
    },
    {
      "name": "Python: Pytest",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": ["tests/", "-v"]
    }
  ]
}
```

## Next Steps

- Review [Contributing Guidelines](contributing.md)
- Learn about [Testing](testing.md)
- Explore the [Architecture](../architecture/overview.md)
