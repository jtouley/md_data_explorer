# Plan: MIMIC-III Integration & QA Hardening

## 1. Specification Alignment

- Update [docs/specs/spec_clinical_analytics_platform.md](docs/specs/spec_clinical_analytics_platform.md) to formally define "Phase 5: MIMIC-III & Advanced Analytics".
- Update [docs/specs/IMPLEMENTATION_STATUS.md](docs/specs/IMPLEMENTATION_STATUS.md) to move target features from "Future" to "In Progress".

## 2. Documentation & User Guide

- **Update [README.md](README.md)**:
- Add "User Guide for Non-Technical Users" section.
- Explain how to launch the app simply (e.g., "Double click run_app.sh" or similar).
- Describe the UI tabs in plain language (e.g., "The 'Analysis' tab is where you see the charts").
- Add troubleshooting tips for common issues.
- Update `scripts/run_app.sh` to print friendly start-up messages.

## 3. Orchestration Updates

- Update [taskmaster.yaml](taskmaster.yaml):
- **Add Agent**: `validator` (Responsibilities: `pytest`, `data profiling`).
- **Update Agent**: `ui_manager` (Add tasks: `Export results`, `Data profiling`).
- **Update Agent**: `analyzer` (Add tasks: `Survival Analysis`, `Mixed Effects`).
- **Expand Workflow**: `mimic_iii_analysis` (Define DuckDB connection & SQL extraction steps).
- **New Workflow**: `quality_assurance` (Run tests & validation).

## 4. QA Infrastructure

- Create `tests/` directory with `conftest.py` and initial unit tests for `Core` and `CovidMS`.
- Create `src/clinical_analytics/core/profiling.py` for generating dataset statistics.

## 5. MIMIC-III Scaffolding

- Create [src/clinical_analytics/datasets/mimic3/loader.py](src/clinical_analytics/datasets/mimic3/loader.py):
- Implement DuckDB connection logic.
- Implement SQL query execution.
- Create [src/clinical_analytics/datasets/mimic3/definition.py](src/clinical_analytics/datasets/mimic3/definition.py):
- Concrete `Mimic3Dataset` class.

## 6. Feature Implementation

- **Analysis**: Add `run_survival_analysis` to `src/clinical_analytics/analysis/stats.py`.
- **UI**: Add "Export Results" button and "Data Profiling" tab to `src/clinical_analytics/ui/app.py`.
- **Caching**: Implement simple disk/memory caching for expensive queries.
