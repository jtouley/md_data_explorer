# Clinical Analytics Platform - Multi-Dataset Specification

**Version:** 3.0 (Multi-Modal)
**Status:** In Progress
**Scope:** COVID-19 MS, Sepsis (PhysioNet 2019), MIMIC-III (Future)
**Package Manager:** uv

---

## 1. Project Overview

### Purpose
A unified clinical analytics platform capable of ingesting heterogeneous medical datasets (CSV, PSV, Relational DB), harmonizing them into a common clinical schema, and running standardized statistical analyses (Mortality, LOS, Sepsis Prediction) without code duplication.

### Supported Datasets

| Dataset | Type | Source Format | Key Outcomes |
|---------|------|---------------|--------------|
| **COVID-MS** | Registry | Single CSV | COVID-19 Severity, Hospitalization |
| **Sepsis** | ICU Time Series | Multiple PSV files | Sepsis Onset (Hourly) |
| **MIMIC-III** | EHR | Relational CSVs | 30-day Mortality, Resistance |

---

## 2. Architecture

### Core Abstractions (`src/clinical_analytics/core/`)
- **`ClinicalDataset`**: Abstract Base Class defining `load()`, `validate()`, and `get_cohort()`.
- **`UnifiedCohort`**: Standardized DataFrame format for analysis.
  - `patient_id` (str)
  - `time_zero` (datetime)
  - `features` (dict/json column for flexible covariates)
  - `outcome` (int/float)
  - `outcome_label` (str)

### Directory Structure
```
clinical-analytics-platform/
├── pyproject.toml
├── taskmaster.yaml
├── data/
│   ├── raw/
│   │   ├── covid_ms/       # GDSI_OpenDataset_Final.csv
│   │   ├── sepsis/         # PhysioNet 2019 PSVs
│   │   └── mimic3/         # Future use
│   └── processed/          # Parquet/DuckDB for standardized data
├── src/
│   └── clinical_analytics/
│       ├── core/           # Base classes & interfaces
│       ├── datasets/       # Dataset-specific implementations
│       │   ├── covid_ms/
│       │   └── sepsis/
│       ├── analysis/       # Generic statistical tools
│       └── ui/             # Streamlit app
└── scripts/
```

---

## 3. Phase 1: Core Foundation

### Deliverables
1. `pyproject.toml` with dependencies: `pandas`, `duckdb`, `scikit-learn`, `statsmodels`, `streamlit`, `polars` (fast CSV reading).
2. `src/clinical_analytics/core/dataset.py`: Base class definition.

### Interface
```python
class ClinicalDataset(ABC):
    @abstractmethod
    def ingest(self, raw_path: Path) -> None:
        """Convert raw data to internal representation."""
        pass

    @abstractmethod
    def get_cohort(self, **filters) -> pd.DataFrame:
        """Return standardized analysis dataframe."""
        pass
```

---

## 4. Phase 2: COVID-MS Module

### Agent ID: `COVID_MS_AGENT`
**Source:** `data/raw/covid_ms/GDSI_OpenDataset_Final.csv`

### Tasks
1. **Ingestion**: Read CSV, map columns to standard schema.
   - `age_in_cat` -> `age_group`
   - `sex` -> `sex`
   - `covid19_admission_hospital` -> `outcome_hospitalized` (Binary)
   - `covid19_confirmed_case` -> Filter for 'confirmed' only?
2. **Analysis**:
   - Logistic Regression: `outcome_hospitalized ~ age + sex + dmt_type`

---

## 5. Phase 3: Sepsis Module

### Agent ID: `SEPSIS_AGENT`
**Source:** `data/raw/sepsis/` (PhysioNet 2019 Challenge)

### Tasks
1. **Ingestion**:
   - Handle zip/PSV file structure.
   - Aggregate time-series (Min/Max/Mean features per patient).
   - Label: `SepsisLabel` (1 if ever present).
2. **Analysis**:
   - Feature engineering (SOFA components).
   - Cohort creation: Sepsis vs Non-Sepsis.

---

## 6. Phase 4: Unified Analysis & UI

### Agent ID: `PLATFORM_AGENT`

### Tasks
1. **Generic Analysis Library**:
   - `run_logistic_regression(df, formula)`
   - `generate_summary_table(df, groupings)`
2. **Streamlit UI**:
   - Sidebar: Select Dataset (COVID-MS / Sepsis).
   - Main: Dynamic filter widgets based on dataset.
   - Output: Formatted tables & plots.

---

## 7. Execution Plan

1. **Scaffold**: Create directory structure and `pyproject.toml`.
2. **Core**: Implement `ClinicalDataset` interface.
3. **COVID-MS**: Implement `CovidMSDataset` class.
4. **Validation**: Verify COVID-MS analysis works.
5. **Sepsis**: Implement `SepsisDataset` class (placeholder if data missing).
6. **UI**: Connect both to Streamlit.
