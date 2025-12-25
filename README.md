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
- **Polars-Optimized**: Fast data loading and transformation (5-10x faster than pandas)

---

## 📚 User Guide for Non-Technical Users

### Quick Start

#### Step 1: Launch the Application

**On Mac/Linux:**
```bash
./scripts/run_app.sh
```

**On Windows:**
```bash
bash scripts/run_app.sh
```

Or if you have the terminal open:
```bash
streamlit run src/clinical_analytics/ui/app.py
```

#### Step 2: The application will automatically open in your web browser

If it doesn't open automatically, look for a message in the terminal like:
```
Local URL: http://localhost:8501
```

Copy and paste that URL into your web browser.

### Understanding the Interface

#### 📊 Main Screen

1. **Sidebar (Left)**: This is where you select which dataset to analyze
   - **COVID-MS**: COVID-19 and Multiple Sclerosis patient data
   - **Sepsis**: Intensive care unit sepsis data
   - **MIMIC-III**: Coming soon!

2. **Main Area (Center)**: Shows your data and results
   - **Dataset Overview**: Summary statistics (number of patients, outcome rates, etc.)
   - **Data Preview**: First few rows of your data
   - **Analysis Section**: Where you run statistical analyses

#### 🔍 Running an Analysis

1. **Select Predictor Variables**: Choose which factors you want to analyze
   - Example: age_group, sex, comorbidities
   - You can select multiple variables by clicking on them

2. **Click "Run Logistic Regression"**: This button starts the statistical analysis
   - The analysis shows which factors are associated with the outcome
   - Results include odds ratios and p-values

3. **View Results**:
   - **Odds Ratio**: How much a factor increases/decreases risk
     - > 1.0 = increased risk
     - < 1.0 = decreased risk
   - **P-Value**: Statistical significance (< 0.05 is usually significant)
   - **Confidence Intervals**: Range of plausible values

#### 💾 Export Results (Coming Soon)

Click the "Export Results" button to download your analysis results as:
- CSV file (for Excel)
- JSON file (for programmers)
- PDF report (for presentations)

### Common Questions

**Q: The app won't start. What should I do?**
A: Make sure you have:
1. Installed dependencies: `uv sync`
2. Activated the virtual environment
3. Python 3.11+ installed

**Q: I see "Data not available" for a dataset. What does that mean?**
A: The data files for that dataset are not in the `data/raw/` folder. You need to download them first.

**Q: What does "Validation passed with warnings" mean?**
A: It means the platform works correctly, but some optional datasets aren't available (usually Sepsis data).

**Q: How do I interpret the results?**
A:
- Look at the **Odds Ratio** column - this tells you the effect size
- Check the **P-Value** - values < 0.05 are statistically significant
- Read the **Confidence Intervals** - narrow intervals mean more precise estimates

**Q: Can I use my own data?**
A: Yes! You can add custom datasets by:
1. Creating a new folder in `data/raw/your_dataset/`
2. Implementing a new dataset class following the examples in `src/clinical_analytics/datasets/`
3. Adding it to the UI dropdown

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Port 8501 is already in use" | Another Streamlit app is running. Close it or use `streamlit run --server.port 8502 src/clinical_analytics/ui/app.py` |
| "Module not found" error | Run `uv sync` to install dependencies |
| Data not loading | Check that CSV/PSV files are in `data/raw/` folder |
| Regression fails | Make sure you have at least 10 data points with no missing values |
| Browser doesn't open | Manually go to http://localhost:8501 |

### Getting Help

- **Validation Issues**: Run `python scripts/validate_platform.py` to diagnose problems
- **Technical Documentation**: See `docs/specs/` for detailed specifications
- **Implementation Status**: See `docs/specs/IMPLEMENTATION_STATUS.md`

---

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
