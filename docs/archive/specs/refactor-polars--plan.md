# Multi-Dataset Platform Scaffolding Plan (Polars-Refactor)

## 1. Project Initialization
- [ ] Create `pyproject.toml` with `uv` dependencies (`pandas`, `duckdb`, `streamlit`, `statsmodels`, `polars`).
- [ ] Create `taskmaster.yaml` for agent orchestration.
- [ ] Create directory structure: `src/clinical_analytics/core`, `src/clinical_analytics/datasets`, `src/clinical_analytics/analysis`.

## 2. Core Architecture (Extensible Design)
- [ ] Implement `src/clinical_analytics/core/dataset.py`: Define `ClinicalDataset` abstract base class.
    - **Extensibility Note**: Interface supports Polars for ETL.
    - Define methods: `load()`, `validate()`, `get_cohort()`.
- [ ] Implement `src/clinical_analytics/core/schema.py`: Define `UnifiedCohort` schema constants.

## 3. COVID-MS Module (Polars Implementation)
- [ ] Refactor `src/clinical_analytics/datasets/covid_ms/loader.py`:
    - Use `polars` for CSV reading and cleaning.
- [ ] Update `src/clinical_analytics/datasets/covid_ms/definition.py`:
    - Store data as `pl.DataFrame`.
    - `get_cohort()` returns `pd.DataFrame` for stats compatibility.

## 4. Sepsis Module (Polars Implementation)
- [ ] Refactor `src/clinical_analytics/datasets/sepsis/loader.py`:
    - Use `polars.scan_csv` or `read_csv` for fast PSV ingestion.
    - Implement efficient aggregation using Polars lazy API.
- [ ] Update `src/clinical_analytics/datasets/sepsis/definition.py`.

## 5. Analysis & UI
- [ ] Implement `src/clinical_analytics/analysis/stats.py`: Generic `run_logistic_regression`.
- [ ] Create `src/clinical_analytics/ui/app.py`: Streamlit app with dataset selector.
- [ ] Create `scripts/run_app.sh`.

## 6. Validation
- [ ] Run `uv sync`.
- [ ] Test Polars ingestion and analysis.