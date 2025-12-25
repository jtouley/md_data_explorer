# Ibis Semantic Layer - Config-Driven Architecture

**Status:** ✅ Implemented  
**Date:** 2025-01-XX  
**Principle:** DRY, Config-Driven, Extensible

---

## 🎯 Overview

The platform now uses **Ibis** to generate SQL dynamically behind the scenes, replacing the brittle custom Python mapping logic. Everything is driven by `datasets.yaml` configuration - no hardcoded logic.

---

## ✅ What's Config-Driven

### 1. **Dataset Discovery**
- ✅ Auto-discovery via `DatasetRegistry`
- ✅ No hardcoded dataset lists in UI
- ✅ Add new dataset = add config entry + implement class

### 2. **Column Mappings**
- ✅ All column renames in `column_mapping` section
- ✅ No hardcoded `pl.col().alias()` logic
- ✅ Semantic layer reads config and builds SQL

### 3. **Outcome Definitions**
- ✅ Binary mappings (yes/no → 1/0) in `outcomes` section
- ✅ CASE WHEN expressions generated from config
- ✅ Multiple outcomes per dataset supported

### 4. **Filters**
- ✅ Filter definitions in `filters` section
- ✅ Filter types: `equals`, `in`, `range`, `exists`
- ✅ Default filters applied automatically

### 5. **Metrics & Dimensions** (NEW!)
- ✅ Metrics defined in `metrics` section
- ✅ Dimensions defined in `dimensions` section
- ✅ Query builder UI reads from config
- ✅ SQL generated dynamically based on selections

### 6. **Time Zero**
- ✅ Static values or source columns from config
- ✅ No hardcoded date logic

### 7. **Analysis Defaults**
- ✅ Default outcomes, predictors, categorical variables
- ✅ All from config

---

## 📁 File Structure

```
src/clinical_analytics/
├── core/
│   ├── semantic.py          # DRY semantic layer (reusable!)
│   ├── dataset.py            # Abstract base class
│   ├── schema.py            # UnifiedCohort schema
│   └── registry.py           # Auto-discovery
├── datasets/
│   ├── covid_ms/
│   │   └── definition.py     # Thin wrapper around SemanticLayer
│   └── sepsis/
│       └── definition.py     # Thin wrapper around SemanticLayer
└── ui/
    └── app.py                # UI with query builder

data/configs/
└── datasets.yaml             # Single source of truth
```

---

## 🔧 How to Add a New Dataset

### Step 1: Add Config Entry

```yaml
# data/configs/datasets.yaml
my_new_dataset:
  name: "My Dataset"
  display_name: "My New Dataset"
  description: "Description here"
  source: "Source name"
  status: "available"
  
  init_params:
    source_path: "data/raw/my_dataset/data.csv"
  
  time_zero:
    value: "2020-01-01"
  
  column_mapping:
    id: patient_id
    age: age
    outcome_col: outcome
  
  outcomes:
    outcome:
      source_column: outcome_col
      type: binary
      mapping:
        yes: 1
        no: 0
  
  metrics:
    outcome_rate:
      expression: "outcome.mean()"
      type: "rate"
      label: "Outcome Rate"
  
  dimensions:
    age:
      label: "Age"
      type: "continuous"
  
  filters:
    active_only:
      type: equals
      column: status
      description: "Active patients only"
  
  default_filters:
    active_only: true
  
  analysis:
    default_outcome: outcome
    default_predictors:
      - age
```

### Step 2: Implement Dataset Class

```python
# src/clinical_analytics/datasets/my_dataset/definition.py
from clinical_analytics.core.dataset import ClinicalDataset
from clinical_analytics.core.semantic import SemanticLayer
from clinical_analytics.core.mapper import load_dataset_config

class MyDataset(ClinicalDataset):
    def __init__(self, source_path=None):
        self.config = load_dataset_config('my_new_dataset')
        if source_path is None:
            source_path = self.config['init_params']['source_path']
        super().__init__(name="my_new_dataset", source_path=source_path)
        self.semantic = SemanticLayer('my_new_dataset', config=self.config)
    
    def validate(self) -> bool:
        return self.source_path.exists() if self.source_path else False
    
    def load(self) -> None:
        if not self.validate():
            raise FileNotFoundError(f"Source not found: {self.source_path}")
        print(f"Semantic layer initialized for {self.name}")
    
    def get_cohort(self, **filters) -> pd.DataFrame:
        outcome_col = filters.get("target_outcome")
        filter_only = {k: v for k, v in filters.items() if k != "target_outcome"}
        return self.semantic.get_cohort(
            outcome_col=outcome_col,
            filters=filter_only
        )
```

### Step 3: Register in `__init__.py`

```python
# src/clinical_analytics/datasets/__init__.py
from .my_dataset.definition import MyDataset
```

**That's it!** The registry auto-discovers it, UI shows it, query builder works.

---

## 🔍 Query Builder Usage

The query builder is **fully config-driven**:

1. **Metrics** come from `metrics` section in config
2. **Dimensions** come from `dimensions` section in config
3. **Filters** come from `filters` section in config
4. **SQL is generated** behind the scenes via Ibis

### Example Query

User selects:
- Metrics: `hospitalization_rate`, `patient_count`
- Dimensions: `age_group`, `sex`
- Filters: `confirmed_only = true`

**Generated SQL** (behind the scenes):
```sql
SELECT 
  age_group,
  sex,
  AVG(outcome_hospitalized) as "Hospitalization Rate",
  COUNT(*) as "Patient Count"
FROM covid_ms_raw
WHERE covid19_confirmed_case = 'confirmed'
GROUP BY age_group, sex
```

---

## 🎨 Extensibility Points

### 1. **Add New Metric Type**

Extend `_build_metric_expression()` in `semantic.py`:

```python
elif agg_func == 'median()':
    return col_expr.median()
```

### 2. **Add New Filter Type**

Extend `apply_filters()` in `semantic.py`:

```python
elif filter_type == 'regex':
    view = view.filter(_[column].re_match(filter_value))
```

### 3. **Add New Outcome Type**

Extend `get_base_view()` in `semantic.py`:

```python
elif outcome_def.get('type') == 'continuous':
    mutations[outcome_name] = _[source_col].cast('float64')
```

---

## 🚀 Benefits

| Before (Custom Mapper) | After (Ibis Semantic Layer) |
|------------------------|----------------------------|
| 450+ lines of Python mapping logic | ~100 lines of config |
| Hardcoded transformations | Config-driven |
| Brittle filter logic | Standard SQL compilation |
| Can't scale beyond CSV | Works with DuckDB, Snowflake, etc. |
| Fixed endpoints | Dynamic SQL generation |
| Hard to extend | Add config entry = done |

---

## 📊 SQL Transparency

The semantic layer can show generated SQL for debugging:

```python
cohort = semantic.get_cohort(show_sql=True)
# Prints:
# Generated SQL for covid_ms:
# SELECT patient_id, time_zero, outcome, outcome_label, ...
# FROM covid_ms_raw
# WHERE covid19_confirmed_case = 'confirmed'
```

---

## ✅ DRY Principles

1. **Single SemanticLayer class** - reused by all datasets
2. **Single config file** - `datasets.yaml` is source of truth
3. **No code duplication** - transformations defined once in config
4. **Extensible** - add new datasets/metrics/filters via config only

---

## 🔮 Future Enhancements

- [ ] Support for SQL-based datasets (MIMIC-III) via semantic layer
- [ ] Config-driven aggregation strategies
- [ ] Config-driven visualization types
- [ ] Multi-dataset joins via config
- [ ] Config-driven data quality checks
- [ ] Automatic data dictionary parsing from README files (MIMIC, COVID-MS pattern)
  - Extract table/column descriptions
  - Infer relationships from documentation
  - Auto-generate semantic layer config from data dictionary

