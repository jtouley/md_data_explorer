# Clinical Analytics Platform - Multi-Dataset Specification

**Version:** 3.1 (Semantic Layer Enhanced)
**Status:** Historical (Scope Changed)
**Scope:** User-Uploaded Datasets Only
**Package Manager:** uv

> **⚠️ Scope Change:** Built-in datasets (COVID-MS, Sepsis, MIMIC-III) referenced below have been **removed**. The platform now supports **user-uploaded datasets only** with automatic schema inference. See [dataset-registry.md](../architecture/dataset-registry.md) for current architecture.

**Related Documentation:**
- **[vision/UNIFIED_VISION.md](../vision/UNIFIED_VISION.md)** - Strategic direction and semantic NL query vision
- **[architecture/IBIS_SEMANTIC_LAYER.md](../architecture/IBIS_SEMANTIC_LAYER.md)** - Semantic layer implementation details
- **[architecture/ARCHITECTURE_OVERVIEW.md](../architecture/ARCHITECTURE_OVERVIEW.md)** - System architecture overview
- **[vision/SPECS_EVOLUTION_ANALYSIS.md](../vision/SPECS_EVOLUTION_ANALYSIS.md)** - Evolution from original specs

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

### Semantic Layer (`src/clinical_analytics/core/semantic.py`)

**Status:** ✅ Implemented

The platform uses an **Ibis-based semantic layer** that generates SQL dynamically from configuration, eliminating hardcoded transformations.

**Key Components:**
- **`SemanticLayer`**: Config-driven SQL generation via Ibis
- **`DatasetRegistry`**: Auto-discovery of dataset implementations
- **Configuration**: `data/configs/datasets.yaml` - Single source of truth

**Capabilities:**
- ✅ Config-driven outcomes, metrics, dimensions
- ✅ Dynamic SQL generation from semantic understanding
- ✅ Zero-code dataset addition (add config entry + implement class)
- ✅ Query builder UI reads from config
- ✅ Filter application via config definitions

**Benefits:**
- No hardcoded transformations in loaders
- Consistent behavior across all datasets
- Easy to extend with new datasets
- SQL generation transparent and debuggable

**See:** [IBIS_SEMANTIC_LAYER.md](../IBIS_SEMANTIC_LAYER.md) for detailed implementation.

### Natural Language Query Enhancement (Optional)

**Status:** 🔄 Planned (Phase 3 in implementation plan)

The platform architecture supports enhancement with **natural language query capabilities** that leverage the semantic layer for context-aware understanding.

**Proposed Features:**
- Free-form natural language input (primary)
- Structured questions as fallback
- Semantic embeddings for intent classification
- RAG pattern using semantic layer metadata
- Automatic variable matching from queries

**Architecture:**
```
User Query (NL or Structured)
    ↓
Semantic Understanding (Embeddings + RAG)
    ↓
Semantic Layer (Ibis SQL Generation)
    ↓
Results
```

**Note:** This is an **optional enhancement** that builds on the existing semantic layer architecture. The current menu-driven UI remains functional, with NL queries as an additional interface option.

**See:** [vision/UNIFIED_VISION.md](../vision/UNIFIED_VISION.md) for complete strategic vision and implementation approach.

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
│   ├── configs/
│   │   └── datasets.yaml    # Semantic layer configuration
│   ├── dictionaries/       # Data dictionary PDFs (for NL queries)
│   └── processed/          # Parquet/DuckDB for standardized data
├── src/
│   └── clinical_analytics/
│       ├── core/           # Base classes & interfaces
│       │   ├── semantic.py  # Semantic layer (Ibis-based)
│       │   ├── registry.py  # Dataset auto-discovery
│       │   └── mapper.py    # Column mapping engine
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
   - Sidebar: Select Dataset (auto-discovered via Registry).
   - Main: Dynamic filter widgets based on dataset config.
   - Query Builder: Config-driven metrics and dimensions selection.
   - Output: Formatted tables & plots.
   - **Future Enhancement:** Natural language query input (see [vision/UNIFIED_VISION.md](../vision/UNIFIED_VISION.md))

---

## 7. Phase 5: MIMIC-III & Advanced Analytics

### Agent ID: `MIMIC_III_AGENT`

**Status:** In Progress
**Source:** MIMIC-III Clinical Database

### Tasks

1. **DuckDB Backend Implementation**:
   - Implement `Mimic3Dataset` class with DuckDB connection support
   - SQL-based cohort extraction from relational tables
   - Support for ADMISSIONS, PATIENTS, DIAGNOSES_ICD tables
   - Efficient query execution and data loading

2. **Advanced Statistical Analysis**:
   - `run_survival_analysis()`: Kaplan-Meier curves and Cox regression
   - `run_mixed_effects()`: Hierarchical models for clustered data
   - Time-series analysis for ICU vital signs
   - Risk stratification models

3. **Data Quality & Profiling**:
   - `src/clinical_analytics/core/profiling.py`: Dataset profiling module
   - Missing data visualization and reporting
   - Distribution analysis for numeric features
   - Categorical feature frequency tables
   - Data quality metrics and validation

4. **Enhanced UI Features**:
   - **Export Results**: CSV, Excel, JSON export functionality
   - **Data Profiling Tab**: Interactive data quality dashboard
   - **Advanced Filters**: Multi-level filtering and cohort building
   - **Caching**: Disk/memory caching for expensive queries
   - **Visualization**: Enhanced plots with Plotly/Altair

5. **Testing & QA Infrastructure**:
   - Unit tests for core modules with `pytest`
   - Integration tests for dataset loading
   - Regression tests for statistical analysis
   - Continuous validation suite
   - Code coverage reporting

### Deliverables

- `src/clinical_analytics/datasets/mimic3/`
  - `loader.py`: DuckDB connection and SQL query execution
  - `definition.py`: `Mimic3Dataset` implementation
  - `queries.py`: Common SQL queries for cohort extraction

- `src/clinical_analytics/core/profiling.py`
- `src/clinical_analytics/analysis/survival.py`
- `tests/` directory with comprehensive test suite
- Enhanced `ui/app.py` with export and profiling features
- Updated documentation and user guide

---

## 8. Execution Plan

1. **Scaffold**: Create directory structure and `pyproject.toml`. ✅
2. **Core**: Implement `ClinicalDataset` interface. ✅
3. **COVID-MS**: Implement `CovidMSDataset` class. ✅
4. **Validation**: Verify COVID-MS analysis works. ✅
5. **Sepsis**: Implement `SepsisDataset` class (placeholder if data missing). ✅
6. **UI**: Connect both to Streamlit. ✅
7. **Semantic Layer**: Implement Ibis-based semantic layer. ✅
8. **Config-Driven**: Refactor to fully config-driven architecture. ✅
9. **QA**: Implement testing infrastructure. 🔄 In Progress
10. **MIMIC-III**: Implement DuckDB backend and dataset. 🔄 In Progress
11. **Advanced Analytics**: Add survival analysis and profiling. 🔄 In Progress
12. **NL Query Enhancement**: Natural language query interface (optional). ⏳ Planned

---

## 9. Evolution Notes

### From Hardcoded to Config-Driven

The platform has evolved from hardcoded transformations to a fully config-driven semantic layer:

**Before:**
- Hardcoded column mappings in loaders
- Manual if/else chains for dataset loading
- Hardcoded filter logic
- Fixed analysis configurations

**After:**
- Config-driven semantic layer (Ibis-based)
- Auto-discovery via DatasetRegistry
- Config-driven filters, outcomes, metrics
- Zero-code dataset addition

**See:** [ARCHITECTURE_REFACTOR.md](../ARCHITECTURE_REFACTOR.md) for detailed evolution documentation.

### Future Enhancements

**Natural Language Query Interface** (Optional):
- Enhances user experience with free-form queries
- Builds on existing semantic layer architecture
- Uses RAG patterns with semantic layer metadata
- Maintains backward compatibility (structured questions remain)

**See:** [UNIFIED_VISION.md](../UNIFIED_VISION.md) for complete strategic vision and implementation roadmap.
