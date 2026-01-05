# Multi-Dataset Platform Scaffolding Plan

## 1. Project Initialization

- [ ] Create `pyproject.toml` with `uv` dependencies (`pandas`, `duckdb`, `streamlit`, `statsmodels`).
- [ ] Create `taskmaster.yaml` for agent orchestration.
- [ ] Create directory structure: `src/clinical_analytics/core`, `src/clinical_analytics/datasets`, `src/clinical_analytics/analysis`.

## 2. Core Architecture (Extensible Design)

- [ ] Implement `src/clinical_analytics/core/dataset.py`: Define `ClinicalDataset` abstract base class.
    - **Critical for Extensibility**: Ensure the interface supports both file-based (CSV/PSV for COVID/Sepsis) and SQL-based (DuckDB/Postgres for MIMIC-III) backends.
    - Define methods: `load()`, `validate()`, `get_cohort()`.
- [ ] Implement `src/clinical_analytics/core/schema.py`: Define `UnifiedCohort` schema constants to harmonize data across different sources.

## 3. COVID-MS Module (File-Based Implementation)

- [ ] Implement `src/clinical_analytics/datasets/covid_ms/loader.py`:
    - Read `data/raw/covid_ms/GDSI_OpenDataset_Final.csv`.
    - Map `age_in_cat`, `sex`, `covid19_admission_hospital` to standardized fields.
- [ ] Implement `src/clinical_analytics/datasets/covid_ms/definition.py`: Concrete `CovidMSDataset` class.

## 4. Sepsis Module (Complex File-Based Implementation)

- [ ] Implement `src/clinical_analytics/datasets/sepsis/loader.py`:
    - Logic to detect/read `.psv` files from `data/raw/sepsis`.
    - Handle missing data gracefully.
- [ ] Implement `src/clinical_analytics/datasets/sepsis/definition.py`: Concrete `SepsisDataset` class.

## 5. Analysis & UI

- [ ] Implement `src/clinical_analytics/analysis/stats.py`: Generic `run_logistic_regression` that works on `UnifiedCohort` regardless of source.
- [ ] Create `src/clinical_analytics/ui/app.py`: Streamlit app with dataset selector.
- [ ] Create `scripts/run_app.sh`.

## 6. Validation

- [ ] Run `uv sync`.
- [ ] Test COVID-MS ingestion and simple analysis.
