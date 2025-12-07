# Clinical Analytics Platform

A multi-dataset clinical analytics platform for processing and analyzing clinical research data from various sources.

## Overview

This platform provides a unified interface for loading, processing, and analyzing clinical datasets including:

- **COVID-MS Dataset**: COVID-19 and multiple sclerosis patient data
- **Sepsis Dataset**: Sepsis patient cohort data
- **MIMIC-III** (future): Medical Information Mart for Intensive Care

The platform harmonizes data from different sources into a unified schema, enabling consistent statistical analysis across datasets.

## Features

- **Extensible Architecture**: Abstract base classes support both file-based (CSV/PSV) and SQL-based (DuckDB/Postgres) backends
- **Unified Schema**: Standardized cohort representation across different data sources
- **Statistical Analysis**: Built-in support for logistic regression and other statistical methods
- **Interactive UI**: Streamlit-based interface for dataset selection and visualization
- **Agent Orchestration**: Taskmaster-based workflow management

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

## Project Structure

```
src/clinical_analytics/
├── core/           # Core abstractions and schemas
├── datasets/       # Dataset-specific implementations
├── analysis/       # Statistical analysis modules
└── ui/            # Streamlit user interface
```

## Usage

Run the Streamlit application:

```bash
./scripts/run_app.sh
```

## Development

Install development dependencies:

```bash
uv sync --extra dev
```

Run tests:

```bash
pytest
```

## Requirements

- Python 3.11+
- pandas
- duckdb
- streamlit
- statsmodels

## License

TBD
